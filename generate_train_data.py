import asyncio
import aiohttp
import pandas as pd
import os
import yaml
from datasets import load_dataset
from utils import _async_query, extract_components, label_map, NOVITA_API_KEY

# Load prompt from YAML file
with open('/teamspace/studios/this_studio/prompts.yaml', 'r') as file:
    prompts_data = yaml.safe_load(file)
prompt_template = prompts_data["prompt_inference_template"]

# Parameters
BATCH_SIZE = 50
TARGET_CORRECT_PER_LABEL = 500

async def process_batch(batch_df):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks_info = []
        for _, row in batch_df.iterrows():
            prompt = prompt_template.format(row['text'])
            tasks_info.append((row, prompt))
        tasks = [ _async_query(prompt, session=session) for _, prompt in tasks_info ]
        print(f"Sending {len(tasks)} asynchronous requests...")
        api_responses = await asyncio.gather(*tasks)
        
        # Retry logic for None responses
        for idx, response in enumerate(api_responses):
            row, prompt = tasks_info[idx]
            if response is None:
                print(f"Error encountered for row {idx+1}")
                await asyncio.sleep(60)
                retry_response = await _async_query(prompt, api_key=NOVITA_API_KEY, session=session)
                if retry_response is not None:
                    response = retry_response
                else:
                    print(f"Retry failed for row {idx+1}. Skipping this request.")
            
            if response and isinstance(response, list) and 'generated_text' in response[0]:
                components = extract_components(response[0]['generated_text'])
                predicted = components.get('predicted_label', None)
                if predicted == row['label']:
                    results.append({
                        'text': row['text'],
                        'label': row['label'],
                        'thinking': components.get('thinking', ""),
                        'full_responses': components.get('full_response', "")
                    })
        return results

def main():
    print("Loading dataset...")
    ds = load_dataset("bsvaz/hatexplain-processed", split='train', num_proc=-1)
    df = pd.DataFrame(ds)
    
    # Map integer labels to strings using label_map
    df['label'] = df['label'].map(label_map)
    
    # Change "Offensive" label to "Offensive Language"
    df['label'] = df['label'].apply(lambda x: "Offensive Language" if x == "Offensive" else x)
    
    # Create prompts CSV file with two columns: prompt and labels
    print("Creating prompts CSV file...")
    df['prompt'] = df['text'].apply(lambda t: prompt_template.format(t))
    prompts_df = df[['prompt', 'label']].rename(columns={'label': 'labels'})
    prompts_df = prompts_df.sample(frac=1, random_state=42)  # randomize order
    prompts_df.to_csv("prompts.csv", index=False)
    
    # Split prompts.csv into separate CSVs for each label
    for lbl in ["Normal", "Offensive Language", "Hate Speech"]:
        lbl_df = prompts_df[prompts_df['labels'] == lbl]
        lbl_df.to_csv(f"prompts_{lbl.replace(' ', '_')}.csv", index=False)
    
    # Load existing results if available, so as not to repeat processing.
    results_file = "train_reasoning_data.csv"
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        final_results = existing_results.to_dict(orient="records")
        print(f"Loaded {len(final_results)} previously processed results.")
    else:
        final_results = []
    
    # Process inference in separate phases per label
    for lbl in df['label'].unique():
        print(f"Starting phase for label: {lbl}")
        # Filter out rows already processed for this label based on text uniqueness.
        processed_texts = {r['text'] for r in final_results if r['label'] == lbl}
        subset_df = df[df['label'] == lbl].reset_index(drop=True)
        subset_df = subset_df[~subset_df['text'].isin(processed_texts)]
        
        # Get current count from previously processed data.
        correct_count = len([r for r in final_results if r['label'] == lbl])
        processed_indexes = set()
        current_index = 0
        total_rows = len(subset_df)
        print(f"Already processed {correct_count} rows for label: {lbl}")
        
        # Continue processing until target is reached for the label.
        while correct_count < TARGET_CORRECT_PER_LABEL and current_index < total_rows:
            batch_df = subset_df.iloc[current_index:current_index + BATCH_SIZE]
            current_index += BATCH_SIZE
            
            # Filter out rows already processed in this session.
            batch_df = batch_df[~batch_df.index.isin(processed_indexes)]
            if batch_df.empty:
                continue
            
            loop = asyncio.get_event_loop()
            batch_results = loop.run_until_complete(process_batch(batch_df))
            if batch_results:
                for res in batch_results:
                    if correct_count < TARGET_CORRECT_PER_LABEL:
                        final_results.append(res)
                        correct_count += 1
                processed_indexes.update(batch_df.index)
                print(f"Phase {lbl}: correct_count = {correct_count}")
                # Update results csv after each batch for robustness.
                pd.DataFrame(final_results, columns=["text", "label", "thinking", "full_responses"])\
                  .to_csv(results_file, index=False)
            else:
                print(f"No correct responses in this batch for {lbl}.")
                
        print(f"Completed phase for label: {lbl} with {correct_count} correct responses.")
    
    print(f"Processing completed. Results saved to {results_file}")

if __name__ == "__main__":
    main()

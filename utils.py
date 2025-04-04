import re  # Import regular expressions for text parsing
import requests  # Import the requests library to make HTTP requests
from dotenv import load_dotenv
import os
import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from openai import OpenAI, AsyncOpenAI

load_dotenv()

NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")

label_map = {0: 'Hate Speech', 1: 'Normal', 2: 'Offensive Language'}

generation_parameters = {
            "temperature": 0.6,
            "max_new_tokens": 1024
        }

def query(prompt, api_key=NOVITA_API_KEY):
    # Create OpenAI client with custom base URL
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.novita.ai/v3/openai"
    )
    
    # Call the model using the chat completions API
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1",
        messages=[{"role": "user", "content": prompt}],
        temperature=generation_parameters["temperature"],
        max_tokens=generation_parameters["max_new_tokens"]
    )
    
    # Return the response in a format similar to the original function
    return [{"generated_text": response.choices[0].message.content}]


async def _async_query(prompt, api_key=NOVITA_API_KEY, session=None):
    """
    Asynchronously queries the OpenAI API via Novita endpoint.

    Args:
        prompt (str): The prompt to send to the model.
        api_key (str): Novita API key.
        session (aiohttp.ClientSession, optional): Not used with OpenAI async client but kept for compatibility.

    Returns:
        dict: The JSON response from the API.
    """
    try:
        # Create async OpenAI client
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.novita.ai/v3/openai"
        )
        
        # Call the model using the chat completions API
        response = await client.chat.completions.create(
            model="deepseek/deepseek-r1",
            messages=[{"role": "user", "content": prompt}],
            temperature=generation_parameters["temperature"],
            max_tokens=generation_parameters["max_new_tokens"]
        )
        
        # Return the response in the same format as before
        return [{"generated_text": response.choices[0].message.content}]
    except Exception as e:
        print(f"Error during API request: {e}")
        return None


async def analyze_samples_by_index(train_df, prompt_template: str, indexes, hf_api_key=NOVITA_API_KEY):
    """
    Analyze samples from the training dataset at specified indexes using asynchronous API calls.

    Args:
        train_df (pd.DataFrame): Training dataframe containing text and labels.
        prompt_template (str): Template string for formatting prompts.
        indexes: List of integers or slice object specifying which samples to analyze.
        hf_api_key (str): Novita API key.

    Returns:
        pd.DataFrame: DataFrame containing original text, label, and extracted components
    """
    tasks = [] # List to store asynchronous tasks
    samples = train_df.iloc[indexes]
    
    async with aiohttp.ClientSession() as session:
        # Process each sample
        for _, row in samples.iterrows():
            formatted_prompt = prompt_template.format(row['text'])
            task = asyncio.create_task(_async_query(formatted_prompt, hf_api_key, session))
            tasks.append(task)

        # Gather responses from all tasks concurrently
        api_responses = await asyncio.gather(*tasks)

        # Process responses and build DataFrame
        processed_data = []
        for i, response in enumerate(api_responses):
            if response:
                # Extract components from the generated text
                components = extract_components(response[0]['generated_text'])
                # Add original text and label
                components['text'] = samples.iloc[i]['text']
                components['label'] = samples.iloc[i]['label']
                processed_data.append(components)
            else:
                print(f"No valid API response received for sample {i+1}")

        # Create DataFrame from all processed data
        if processed_data:
            return pd.DataFrame(processed_data)[['text', 'label', 'predicted_label', 'thinking', 'full_response']]
        else:
            return pd.DataFrame()

def extract_components(response_text: str) -> dict:
    """Extract components from model response text."""
    # Check if the input is in the new format (list with dictionary)
    if isinstance(response_text, list) and len(response_text) > 0 and 'generated_text' in response_text[0]:
        response_text = response_text[0]['generated_text']
    
    components = {
        'full_response': response_text,
        'thinking': None,
        'predicted_label': None
    }
    
    # Extract thinking
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response_text, re.DOTALL)
    if think_match:
        components['thinking'] = think_match.group(1).strip()
    
    # Get everything after </think>
    after_think_pattern = r'</think>(.*?)$'
    after_think_match = re.search(after_think_pattern, response_text, re.DOTALL)
    if after_think_match:
        after_think_text = after_think_match.group(1).strip()
        
        # Extract the final label directly (no classification tags in new format)
        # Clean up the text and look for known labels
        cleaned_text = after_think_text.strip()
        
        # Check for standard labels in the final text
        if 'hate speech' in cleaned_text.lower():
            components['predicted_label'] = 'Hate Speech'
        elif 'offensive language' in cleaned_text.lower() or 'offensive' in cleaned_text.lower():
            components['predicted_label'] = 'Offensive Language'
        elif 'normal' in cleaned_text.lower():
            components['predicted_label'] = 'Normal'
        else:
            # If no standard label is found, use the entire text after </think>
            components['predicted_label'] = cleaned_text
        
    
    return components

def plot_confusion_matrix(df):
    # Create an explicit copy and then drop NA values
    df = df.copy()
    df = df.dropna(subset=['label', 'predicted_label'])
    
    # Map 'Offensive Language' to 'Offensive' for consistency
    df.loc[:, 'predicted_label'] = df['predicted_label']
    
    # Define the desired order of labels
    labels = ['Normal', 'Offensive Language', 'Hate Speech']
    
    # Create confusion matrix manually
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    skipped = 0
    for true, pred in zip(df['label'], df['predicted_label']):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true]][label_to_idx[pred]] += 1
        else:
            skipped += 1
            print(f"Skipping data point: true='{true}', pred='{pred}'")
    
    # Normalize by row (true labels)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add labels
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add numbers in cells (showing both percentage and raw count)
    for i in range(n_labels):
        for j in range(n_labels):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.2%}\n({int(cm[i, j])})', 
                         ha="center", va="center")
    
    plt.title('Normalized Confusion Matrix' + (f' (skipped {skipped} mislabelled data points)' if skipped else ''))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def calculate_response_accuracy(responses):
    # Filter out None values
    valid_responses = responses.dropna(subset=['predicted_label'])
    
    # Report number of ignored samples
    ignored = len(responses) - len(valid_responses)
    if ignored > 0:
        print(f"Ignored {ignored} samples with None predictions")
    
    # Calculate accuracy on valid responses
    total = len(valid_responses)
    correct = sum((valid_responses['label'] == valid_responses['predicted_label']) | 
                ((valid_responses['label'] == 'Offensive') & 
                 (valid_responses['predicted_label'] == 'Offensive Language')))
    accuracy = correct / total if total > 0 else 0

    print(f"Accuracy: {accuracy:.2%}")

def display_mislabeled_analysis(df):
    def wrap_text(text, width=80):
        return '\n    '.join(textwrap.wrap(str(text), width=width))
    
    print("\nMISLABELED CASES ANALYSIS")
    
    for original_label in sorted(df['label'].unique()):
        print(f"\n{'='*80}")
        original_group = df[df['label'] == original_label]
        print(f"\nORIGINAL LABEL: {original_label}")
        print(f"Total cases: {len(original_group)}")
        
        for pred_label in sorted(original_group['predicted_label'].unique()):
            cases = original_group[original_group['predicted_label'] == pred_label]
            print(f"\n{'-'*40}")
            print(f"Predicted as: {pred_label} ({len(cases)} cases)")
            
            for idx, case in cases.iterrows():
                print(f"\nCase {idx}:")
                print("\nText:")
                print(wrap_text(case['text']))
                print("\nReasoning:")
                print(wrap_text(case['thinking']))
                print(f"\n{'-'*30}")

def extract_classification_and_thinking(df):
    # Create new DataFrame with 'text' column
    new_df = pd.DataFrame({'text': df['text']})
    
    # Function to extract thinking content
    def get_thinking(response):
        think_pred = response.split('<think>\n')[-1]
        think = think_pred.split('</think>')[0]
        return think
    
    # Function to extract final prediction
    def get_prediction(response):
        # Look for the last occurrence of the classification terms after </think>
        response_after_think = response.split('</think>')[-1].lower()
        
        # Search for classification terms
        if 'hate speech' in response_after_think:
            return 'Hate Speech'
        elif 'offensive language' in response_after_think:
            return 'Offensive Language'
        elif 'normal' in response_after_think:
            return 'Normal'
        return None
    
    # Apply extraction functions
    new_df['thinking'] = df['response'].apply(get_thinking)
    new_df['prediction'] = df['response'].apply(get_prediction)
    
    return new_df


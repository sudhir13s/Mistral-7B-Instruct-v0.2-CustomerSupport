import os
import json
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def format_mistral_instruction(instruction, response, intent=None, action=None):
    """
    Formats the input into the Mistral-7B-Instruct-v0.2 prompt template.
    Template: <s>[INST] {instruction} [/INST] {response} </s>
    """
    # Create structured response if components are provided
    if intent or action:
        structured_response = {
            "intent": intent,
            "response": response,
            "action": action or "suggest_next_step"
        }
        response_text = json.dumps(structured_response, indent=2)
    else:
        response_text = response

    return f"<s>[INST] {instruction} [/INST] {response_text} </s>"

def prepare_data():
    config = load_config()
    dataset_config = config['dataset']
    
    print(f"Loading dataset: {dataset_config['hf_path']}...")
    # Load the Bitext dataset
    dataset = load_dataset(dataset_config['hf_path'], split=dataset_config['train_split'])
    
    formatted_data = []
    
    # Process dataset entries
    for entry in dataset:
        # The bitext dataset contains 'instruction', 'intent', and 'response'
        instruction = entry['instruction']
        intent = entry['intent']
        response = entry['response']
        
        # We manually map the action based on intent for better structure
        action = f"handle_{intent}" 
        
        formatted_entry = format_mistral_instruction(
            instruction=instruction,
            response=response,
            intent=intent,
            action=action
        )
        
        formatted_data.append({"text": formatted_entry})

    # Add E-commerce Synthetic Samples for better structure learning
    synthetic_samples = [
        {
            "instruction": "I received my order #12345 but it is damaged. What should I do?",
            "intent": "damaged_item",
            "response": "I'm so sorry to hear that your order arrived damaged. Please send us a photo of the item and we'll process a replacement or refund immediately.",
            "action": "initiate_return_damaged"
        },
        {
            "instruction": "Do you offer international shipping to Germany?",
            "intent": "shipping_query",
            "response": "Yes, we ship to Germany! Delivery typically takes 5-7 business days via DHL Express.",
            "action": "provide_shipping_info"
        },
        {
            "instruction": "I want to cancel my subscription for the SaaS Basic plan.",
            "intent": "cancel_subscription",
            "response": "We're sorry to see you go. You can cancel your subscription in the Billing settings, or I can help you with that right now.",
            "action": "cancel_account_workflow"
        }
    ]

    for sample in synthetic_samples:
        formatted_entry = format_mistral_instruction(
            instruction=sample['instruction'],
            response=sample['response'],
            intent=sample['intent'],
            action=sample['action']
        )
        formatted_data.append({"text": formatted_entry})

    # Split into Train and Test
    train_data, test_data = train_test_split(
        formatted_data, 
        test_size=dataset_config['test_size'],
        random_state=42
    )

    # Save to JSONL
    print(f"Saving processed data to data/processed/...")
    
    with open("data/processed/train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("data/processed/test.jsonl", "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Data Preparation Complete! (Train: {len(train_data)}, Test: {len(test_data)})")

if __name__ == "__main__":
    prepare_data()

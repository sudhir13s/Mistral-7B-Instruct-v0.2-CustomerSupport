import os
import yaml
from huggingface_hub import HfApi, ModelCard, ModelCardData, login
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def push_to_hub():
    config = load_config()
    model_cfg = config['model']
    train_cfg = config['training']
    adapter_path = train_cfg['output_dir']
    
    # --- 1. Authentication ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not found.")
        return
    
    print("Authenticating with Hugging Face...")
    login(token=hf_token)
    
    api = HfApi()
    
    # --- 2. Model Repo Info ---
    # Define repository ID using the environment variable
    hf_user = os.getenv("HF_USER_NAME", "your-hf-username")
    repo_id = train_cfg.get('hub_model_id', f"{hf_user}/mistral-7b-support-adapter")
    
    print(f"Pushing adapter to Hub: {repo_id}...")
    
    # --- 3. Upload Adapter Weights ---
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    
    # We upload the adapter files directly from the output directory
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
    )

    # --- 4. Generate & Push Model Card ---
    print("Generating Model Card...")
    card_data = ModelCardData(
        language='en',
        license='apache-2.0',
        library_name='peft',
        tags=['mistral', 'lora', 'customer-support', 'qlora'],
        model_name='Mistral-7B Customer Support Specialist',
        base_model=model_cfg['base_model'],
    )
    
    content = f"""
# Mistral-7B Customer Support Specialist (Fine-Tuned)

This model is a fine-tuned version of [{model_cfg['base_model']}](https://huggingface.co/{model_cfg['base_model']}) using QLoRA. It is specifically designed to handle customer support inquiries by generating structured JSON responses.

## Model Description
- **Task**: Customer Support Auto-Resolution.
- **Output Format**: Structured JSON including `intent`, `response`, and `action`.
- **E-commerce / SaaS Focused**: Trained on highly specialized support ticket datasets.

## How to use
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "{model_cfg['base_model']}"
adapter_name = "{repo_id}"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
```
"""
    
    card = ModelCard.from_template(card_data, template_str=content)
    card.push_to_hub(repo_id)
    
    print(f"Successfully pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    push_to_hub()

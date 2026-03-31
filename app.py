import os
import torch
import json
import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import argparse

# Parse arguments for runtime flexibility
parser = argparse.ArgumentParser()
parser.add_argument("--hf_user", default=os.getenv("HF_USER_NAME", "your-hf-username"), help="Hugging Face Username")
args, unknown = parser.parse_known_args()

# Model Details (Loaded from environment variables or arguments)
HF_USER_NAME = args.hf_user
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_NAME = f"{HF_USER_NAME}/mistral-7b-support-adapter"

def load_model():
    print(f"Loading {BASE_MODEL} and adapter {ADAPTER_NAME}...")
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load adapter
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    except:
        # Fallback if adapter not yet on Hub
        print("Adapter not found on Hub. Running with base model.")
        model = base_model
        
    model.eval()
    return tokenizer, model

# Load model global
tokenizer, model = load_model()

def generate_response(user_query):
    # Mistral Instruction Template
    prompt = f"<s>[INST] {user_query} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the prediction
    prediction = full_output.split("[/INST]")[-1].strip()
    
    try:
        # Attempt to parse as JSON for pretty highlighting
        structured_json = json.loads(prediction)
        return json.dumps(structured_json, indent=2)
    except:
        # Fallback to raw text
        return prediction

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Customer Support Auto-Resolution Model")
    gr.Markdown("> **A fine-tuned Mistral-7B model for structured e-commerce/SaaS support.**")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Customer Inquiry", 
                placeholder="e.g. My order #12345 hasn't arrived yet.",
                lines=3
            )
            submit_btn = gr.Button("Generate Resolution", variant="primary")
            
        with gr.Column():
            output_json = gr.Code(label="Structured Output (JSON)", language="json")
            
    gr.Examples(
        examples=[
            ["I received a broken item. How do I return it?"],
            ["Do you ship to the UK?"],
            ["I want to upgrade my SaaS subscription to the Pro plan."]
        ],
        inputs=input_text
    )

    submit_btn.click(generate_response, inputs=input_text, outputs=output_json)

if __name__ == "__main__":
    demo.launch()

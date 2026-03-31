import os
import torch
import json
import yaml
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_evaluation():
    config = load_config()
    model_cfg = config['model']
    adapter_path = config['training']['output_dir']
    
    # --- 1. Load Tokenizer & Base Model ---
    print(f"📦 Loading Model & Fine-tuned Adapter...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model in 4-bit (to save VRAM during evaluation)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Merge with adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # --- 2. Load Test Dataset ---
    print(f"📊 Loading test set from data/processed/test.jsonl...")
    dataset = load_dataset("json", data_files={"test": "data/processed/test.jsonl"})["test"]
    
    # Select a subset for evaluation speed (e.g., first 50 samples)
    eval_set = dataset.select(range(min(50, len(dataset))))

    # --- 3. Inference Loop ---
    print(f"✨ Running inference on {len(eval_set)} samples...")
    results = []
    
    for entry in tqdm(eval_set):
        full_text = entry['text']
        # Split into [INST] and answer to evaluate.
        # Mistral format: <s>[INST] {instruction} [/INST] {output} </s>
        parts = full_text.split("[/INST]")
        prompt = parts[0] + "[/INST]"
        reference = parts[1].replace("</s>", "").strip()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.1, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the part AFTER the prompt
        prediction = prediction.split("[/INST]")[-1].strip()
        
        results.append({
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction
        })

    # --- 4. Calculate Metrics ---
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(r['reference'], r['prediction'])['rougeL'].fmeasure for r in results]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    
    # BERTScore
    print(f"📈 Calculating BERTScore...")
    references = [r['reference'] for r in results]
    predictions = [r['prediction'] for r in results]
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    avg_bert_f1 = F1.mean().item()

    # --- 5. Report Results ---
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"🏆 Average ROUGE-L: {avg_rouge_l:.4f}")
    print(f"🧠 Average BERTScore (F1): {avg_bert_f1:.4f}")
    print("="*50)
    
    # Sample Comparison
    print("\n📝 Sample Qualitative Check:")
    print(f"🔹 Prompt: {results[0]['prompt'][:100]}...")
    print(f"🔸 Reference: {results[0]['reference']}")
    print(f"✨ Prediction: {results[0]['prediction']}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()

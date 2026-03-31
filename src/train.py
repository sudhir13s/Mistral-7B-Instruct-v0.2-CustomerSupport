import os
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    model_cfg = config['model']
    train_cfg = config['training']
    lora_cfg = config['lora']
    quant_cfg = config['quantization']
    token_cfg = config['tokenizer']

    # --- 1. Load Tokenizer ---
    print(f"📦 Loading Tokenizer: {model_cfg['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = token_cfg['padding_side']

    # --- 2. BitsAndBytes Config (QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get('load_in_4bit', True),
        bnb_4bit_quant_type=quant_cfg['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=quant_cfg['bnb_4bit_use_double_quant'],
    )

    # --- 3. Load Base Model ---
    print(f"🔥 Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg['base_model'],
        quantization_config=bnb_config,
        device_map=model_cfg['device_map'],
        trust_remote_code=model_cfg['trust_remote_code']
    )
    
    # Pre-train prep
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    # --- 4. LoRA Config ---
    peft_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias=lora_cfg['bias'],
        task_type=lora_cfg['task_type'],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 5. Load Dataset ---
    print(f"📊 Loading processed dataset from data/processed/...")
    dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl", "test": "data/processed/test.jsonl"})

    # --- 6. Training Arguments ---
    print(f"🚀 Setting up TrainingArguments...")
    
    # Handle HF Token for potential Hub pushes (using custom env var as requested)
    hf_token = os.getenv("HF_TOKEN")
    
    training_arguments = TrainingArguments(
        output_dir=train_cfg['output_dir'],
        num_train_epochs=train_cfg['num_train_epochs'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        optim=train_cfg['optimizer'],
        save_strategy=train_cfg['save_strategy'],
        logging_steps=train_cfg['logging_steps'],
        learning_rate=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        fp16=train_cfg['fp16'],
        bf16=train_cfg['bf16'],
        max_grad_norm=train_cfg['max_grad_norm'],
        warmup_steps=train_cfg['warmup_steps'],
        group_by_length=train_cfg['group_by_length'],
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        report_to=train_cfg['report_to'],
        evaluation_strategy=train_cfg['evaluation_strategy'],
        eval_steps=train_cfg['eval_steps'],
        save_total_limit=train_cfg['save_total_limit'],
        gradient_checkpointing=train_cfg['gradient_checkpointing'],
        # Optional Hub parameters
        push_to_hub=train_cfg.get('push_to_hub', False),
        hub_token=hf_token,
        hub_model_id=train_cfg.get('hub_model_id'),
    )

    # --- 7. Initialize SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=token_cfg['max_seq_length'],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config['dataset'].get('packing', False),
    )

    # --- 8. Train! ---
    print(f"✨ Starting training...")
    trainer.train()

    # --- 9. Save Adapter ---
    print(f"💾 Saving fine-tuned adapter weights to {train_cfg['output_dir']}...")
    trainer.model.save_pretrained(train_cfg['output_dir'])
    print(f"✅ Training Complete!")

if __name__ == "__main__":
    train()

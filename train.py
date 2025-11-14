#!/usr/bin/env python3
import argparse
import os
import logging
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant that speaks Hinglish (a natural mix of Hindi and English). You understand code-mixed queries and respond naturally in the same style. Be conversational, friendly, and helpful. If asked to explain code, provide clear, step-by-step explanations in Hinglish."""


def train_model(lora_r, lora_alpha, output_dir, hf_dataset_name, hf_model_repo_name):
    """
    Main function to load data, configure, run training, and upload to Hub.
    """
    logger.info(f"--- Starting training for r={lora_r}, alpha={lora_alpha} ---")
    logger.info(f"Dataset: {hf_dataset_name}")
    logger.info(f"Target Repo: {hf_model_repo_name}")

    # --- 1. Load Model & Tokenizer ---
    model_name = "meta-llama/Llama-3.2-3b-instruct"

    # 4-bit config (stable for RTX 3090)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 for stability
        bnb_4bit_use_double_quant=True,  # Enable for memory efficiency
    )

    logger.info(f"Loading base model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    logger.info("Model loaded and prepared for training.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded.")

    # --- 2. Load and Prepare Dataset ---
    logger.info(f"Loading dataset from {hf_dataset_name}...")
    try:
        dataset = load_dataset(hf_dataset_name)
    except Exception as e:
        logger.error(f"FATAL: Could not load dataset. Error: {e}")
        return

    if "validation" not in dataset:
        logger.warning("No 'validation' split found. Creating 5% validation split.")
        train_val_split = dataset["train"].train_test_split(test_size=0.05, seed=42)
        dataset = DatasetDict({
            "train": train_val_split["train"],
            "validation": train_val_split["test"]
        })

    logger.info(f"Train set: {len(dataset['train'])} examples")
    logger.info(f"Val set: {len(dataset['validation'])} examples")

    # --- 3. Configure LoRA ---
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # --- 4. Configure Training Arguments ---
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,  # Conservative for stability
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        optim="paged_adamw_32bit",
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,  # Use fp16 for stability
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        evaluation_strategy="steps",  # Use old parameter name
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,  # Keep only 3 checkpoints
    )

    # --- 5. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=1024,  # Reasonable sequence length
        packing=False,  # Disable for stability
    )

    # --- 6. Start Training ---
    logger.info("Checking for existing checkpoints...")

    resume_from_checkpoint = None
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(output_dir, latest_checkpoint)
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("No checkpoint found. Starting fresh.")
    else:
        logger.info("Starting training from scratch.")

    logger.info("--- Starting Fine-Tuning ---")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("--- Fine-Tuning Complete ---")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # --- 7. Save & Upload ---
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    try:
        logger.info(f"Uploading to Hub: {hf_model_repo_name}")
        trainer.model.push_to_hub(
            repo_id=hf_model_repo_name,
            commit_message=f"LoRA r={lora_r} alpha={lora_alpha}"
        )
        tokenizer.push_to_hub(
            repo_id=hf_model_repo_name,
            commit_message=f"LoRA r={lora_r} alpha={lora_alpha}"
        )
        logger.info("Upload complete!")
    except Exception as e:
        logger.error(f"Upload failed. Is HF_TOKEN set? Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Script")
    parser.add_argument("--r", type=int, required=True, help="LoRA rank")
    parser.add_argument("--alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset name")
    parser.add_argument("--repo", type=str, required=True,
                        help="HF model repo name")

    args = parser.parse_args()

    train_model(
        lora_r=args.r,
        lora_alpha=args.alpha,
        output_dir=args.output_dir,
        hf_dataset_name=args.dataset,
        hf_model_repo_name=args.repo
    )
#!/usr/bin/env python3
import argparse
import os
import logging
import math
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
# THIS IS THE FIXED IMPORT LINE
from transformers.trainer_utils import get_last_checkpoint

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# This is the Llama-3-style system prompt your model will be trained with.
SYSTEM_PROMPT = """You are a helpful AI assistant that speaks Hinglish (a natural mix of Hindi and English). You understand code-mixed queries and respond naturally in the same style. Be conversational, friendly, and helpful. If asked to explain code, provide clear, step-by-step explanations in Hinglish."""


# --- End Configuration ---
def train_model(lora_r, lora_alpha, output_dir, hf_dataset_name, hf_model_repo_name):
    logger.info(f"--- Starting training for r={lora_r}, alpha={lora_alpha} ---")
    logger.info(f"Dataset: {hf_dataset_name}")
    logger.info(f"Target Repo: {hf_model_repo_name}")
    # --- 1. Load Model & Tokenizer ---
    model_name = "meta-llama/Llama-3.2-3b-instruct"

    # QLoRA 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 is better for T4/3090/4090
        bnb_4bit_use_double_quant=False,
    )
    logger.info(f"Loading base model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    logger.info("Model loaded.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Llama 3.2 uses EOS as pad
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded.")
    # --- 2. Load and Prepare Blended Dataset ---
    logger.info(f"Loading dataset from {hf_dataset_name}...")
    try:
        dataset = load_dataset(hf_dataset_name)
    except Exception as e:
        logger.error(f"FATAL: Could not load dataset. Make sure it's public. Error: {e}")
        return
    # Robust checking for train/val splits
    if "validation" not in dataset:
        logger.warning("No 'validation' split found. Creating a 5% validation split from train.")
        train_val_split = dataset["train"].train_test_split(test_size=0.05, seed=42)
        dataset = DatasetDict({
            "train": train_val_split["train"],
            "validation": train_val_split["test"]
        })

    logger.info(f"Loaded train set: {len(dataset['train'])} examples")
    logger.info(f"Loaded val set: {len(dataset['validation'])} examples")
    # --- 3. Configure LoRA ---
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    # --- 4. Configure Training Arguments (with all fixes) ---
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
    )
    # --- 5. Initialize Trainer (Clean, simple constructor) ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
    )
    # --- 6. Start Training (with *SAFE* checkpoint resuming) ---
    logger.info("Checking for existing checkpoints...")

    # This is the robust logic that checks for a checkpoint
    resume_from_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint
        else:
            logger.info("No checkpoint found. Starting training from scratch.")
    else:
        logger.info("Output directory not found. Starting training from scratch.")
    logger.info("--- Starting Fine-Tuning ---")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("--- Fine-Tuning Complete ---")
    # --- 7. Save Final Model & Upload to Hub ---
    final_model_path = os.path.join(output_dir, "final_champion")
    trainer.save_model(final_model_path)
    logger.info(f"Best model saved locally to {final_model_path}")
    try:
        logger.info(f"--- Uploading model to Hugging Face Hub: {hf_model_repo_name} ---")
        trainer.push_to_hub(commit_message=f"Add r={lora_r} alpha={lora_alpha} champion model",
                            repo_id=hf_model_repo_name)
        logger.info("--- Upload Complete ---")
    except Exception as e:
        logger.error(f"FAILED to upload model. Is HF_TOKEN set? Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Script")
    parser.add_argument("--r", type=int, required=True, help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, required=True, help="Local directory to save checkpoints")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of your HF dataset (e.g., 'YourUser/hinglish-blended-dataset')")
    parser.add_argument("--repo", type=str, required=True,
                        help="Name of your HF model repo to upload to (e.g., 'YourUser/hinglish-llama-r8')")

    args = parser.parse_args()

    train_model(
        lora_r=args.r,
        lora_alpha=args.alpha,
        output_dir=args.output_dir,
        hf_dataset_name=args.dataset,
        hf_model_repo_name=args.repo
    )
#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for Llama 3.2-3B-Instruct on Hinglish Dataset (v2.0 - Optimized for 3090).
Supports multi-turn chat formatting, existing splits, and 5-7 hr runtime.

Usage: nohup python finetune_qlora.py --r 8 --alpha 16 --output_dir ./r8-checkpoints --dataset "omhome/hinglish-blended-dataset" --repo "omhome/hinglish-llama-r8" > r8_log.out 2>&1 &
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def formatting_prompts_func(example, tokenizer):
    """Convert {'messages': [...]} to formatted text using Llama 3.2 chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def train_model(lora_r: int, lora_alpha: int, output_dir: str, hf_dataset_name: str, hf_model_repo_name: str, resume_from_checkpoint: str = None):
    """
    Main training loop: Load model/tokenizer, dataset, format chats, configure PEFT/Trainer, train, save, and upload.
    """
    logger.info(f"--- Starting QLoRA fine-tuning: r={lora_r}, alpha={lora_alpha} ---")
    logger.info(f"Dataset: {hf_dataset_name} | Output: {output_dir} | Target Repo: {hf_model_repo_name}")
    logger.info(f"Resume from: {resume_from_checkpoint or 'None (fresh start)'}")

    model_name = "meta-llama/Meta-Llama-3.2-3B-Instruct"
    max_seq_length = 2048  # Suitable for Hinglish chats; adjust if needed

    # --- Quantization Config (4-bit NF4 for 3090) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # --- Load Model ---
    try:
        logger.info(f"Loading {model_name} with 4-bit quant...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
        model.config.use_cache = False
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Model load failed. Check HF token/CUDA.")
        raise

    # --- Load Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error("Tokenizer load failed.")
        raise

    # --- Load Dataset (Use existing train/val splits) ---
    try:
        dataset = load_dataset(hf_dataset_name)
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({"train": dataset})
    except Exception as e:
        logger.error(f"Dataset load failed: {e}. Ensure public/access.")
        raise

    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Dataset must have 'train' and 'validation' splits.")

    logger.info(f"Train: {len(dataset['train'])} examples | Val: {len(dataset['validation'])} examples")

    # --- LoRA Config ---
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --- Training Args (Optimized for ~5-7 hrs on 3090: Effective batch=64) ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,  # Effective batch=64; ~2,100 steps for 135k
        optim="paged_adamw_8bit",
        logging_steps=50,
        learning_rate=2e-4,
        weight_decay=0.001,
        bf16=torch.cuda.is_bf16_supported(),
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
        eval_steps=500,  # ~4 evals total to save time
        save_steps=500,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        resume_from_checkpoint=resume_from_checkpoint,
        push_to_hub=False,  # Handle manually
        dataloader_num_workers=4,  # Speed up loading
    )

    # --- SFT Trainer with Chat Formatting ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=lambda ex: formatting_prompts_func(ex, tokenizer),  # Applies chat template
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,  # Packs short seqs for efficiency
    )

    # --- Train ---
    logger.info("--- Starting training ---")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("--- Training complete ---")
    except Exception as e:
        logger.error("Training failed.")
        raise

    # --- Save Locally ---
    final_path = Path(output_dir) / "final_model"
    try:
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(final_path)
        logger.info(f"Model saved to {final_path}")
    except Exception as e:
        logger.error(f"Save failed: {e}")

    # --- Upload to HF Hub (Assumes HF_TOKEN set) ---
    try:
        logger.info(f"Uploading to {hf_model_repo_name}...")
        trainer.push_to_hub(
            commit_message=f"QLoRA r={lora_r} alpha={lora_alpha} on Hinglish dataset",
            repo_id=hf_model_repo_name,
        )
        logger.info("Upload complete!")
    except Exception as e:
        logger.error(f"Upload failed: {e}. Check HF_TOKEN.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Hinglish Llama 3.2-3B")
    parser.add_argument("--r", type=int, required=True, help="LoRA rank (e.g., 8)")
    parser.add_argument("--alpha", type=int, required=True, help="LoRA alpha (e.g., 16)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset (e.g., 'omhome/hinglish-blended-dataset')")
    parser.add_argument("--repo", type=str, required=True, help="HF repo to upload")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume checkpoint")
    args = parser.parse_args()

    train_model(
        lora_r=args.r,
        lora_alpha=args.alpha,
        output_dir=args.output_dir,
        hf_dataset_name=args.dataset,
        hf_model_repo_name=args.repo,
        resume_from_checkpoint=args.resume_from,
    )
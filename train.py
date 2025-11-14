#!/usr/bin/env python3
# (rest of imports and config are same as your compat script above)

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_bnb_dtype():
    return torch.float16

def train_model(lora_r, lora_alpha, output_dir, hf_dataset_name, hf_model_repo_name, resume_from_checkpoint=True):
    logger.info(f"Starting training: r={lora_r}, alpha={lora_alpha}")
    logger.info(f"Dataset: {hf_dataset_name}  Repo: {hf_model_repo_name}")

    model_name = "meta-llama/Llama-3.2-3b-instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=safe_bnb_dtype(),
        bnb_4bit_use_double_quant=False,
    )

    # --- Model & Tokenizer ---
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    # --- Dataset load (same as before) ---
    raw = load_dataset(hf_dataset_name)
    if isinstance(raw, DatasetDict):
        dataset = raw
    elif isinstance(raw, dict):
        dataset = DatasetDict(raw)
    else:
        dataset = DatasetDict({"train": raw})

    if "train" not in dataset:
        raise ValueError("Dataset has no 'train' split.")
    if "validation" not in dataset:
        logger.info("No 'validation' split found. Creating a 5% validation split from train.")
        train_ds = dataset["train"]
        val_size = max(1, math.floor(0.05 * len(train_ds)))
        dataset = DatasetDict({
            "train": train_ds.select(range(len(train_ds) - val_size)),
            "validation": train_ds.select(range(len(train_ds) - val_size, len(train_ds))),
        })

    logger.info("Train size: %d, Validation size: %d", len(dataset["train"]), len(dataset["validation"]))

    # --- LoRA config ---
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --- TrainingArguments (safe) ---
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=use_fp16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
    )

    # --- Prepare SFTTrainer arguments (DO NOT pass tokenizer kw) ---
    sft_extra_kwargs = {
        "dataset_packing": True,
        "max_seq_length": 4096,
    }

    trainer = None
    try:
        # Primary attempt: include sft-specific kwargs but NOT tokenizer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            args=training_args,
            **sft_extra_kwargs,
        )
        logger.info("SFTTrainer created with sft_extra_kwargs.")
    except TypeError as e:
        logger.warning("SFTTrainer rejected extra kwargs; retrying with minimal signature. Error: %s", e)
        # Fallback: minimal constructor (no tokenizer, no sft extras)
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            args=training_args,
        )
        logger.info("SFTTrainer created with minimal signature.")

    # --- Attach tokenizer to trainer AFTER construction (fixes TRL 0.9.6 API) ---
    try:
        # Some TRL versions expect a tokenizer attribute; assign it
        trainer.tokenizer = tokenizer
        logger.info("Assigned tokenizer to trainer.tokenizer")
    except Exception as e:
        logger.warning("Failed to set trainer.tokenizer property: %s", e)

    # --- Start training ---
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training complete.")

    # --- Save & push ---
    final_model_path = os.path.join(output_dir, "final_champion")
    try:
        trainer.save_model(final_model_path)
        logger.info("Saved model to %s", final_model_path)
    except Exception as e:
        logger.warning("Could not save model locally: %s", e)

    try:
        trainer.push_to_hub(commit_message=f"Add r={lora_r} alpha={lora_alpha} champion model", repo_id=hf_model_repo_name)
        logger.info("Pushed to hub.")
    except Exception as e:
        logger.warning("Push to hub failed: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tune (compat mode)")
    parser.add_argument("--r", type=int, required=True, help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, required=True, help="Local directory to save checkpoints")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name (e.g., user/dataset)")
    parser.add_argument("--repo", type=str, required=True, help="HF model repo to push to (e.g., user/repo)")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume-from-checkpoint")
    args = parser.parse_args()

    train_model(
        lora_r=args.r,
        lora_alpha=args.alpha,
        output_dir=args.output_dir,
        hf_dataset_name=args.dataset,
        hf_model_repo_name=args.repo,
        resume_from_checkpoint=not args.no_resume,
    )

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import HfApi, HfFolder


def train_model(lora_r, lora_alpha, output_dir, hf_dataset_name, hf_model_repo_name):
    """
    Main function to load data, configure, run training, and upload to Hub.
    """
    print(f"--- Starting training for r={lora_r}, alpha={lora_alpha} ---")
    print(f"Dataset: {hf_dataset_name}")
    print(f"Target Repo: {hf_model_repo_name}")

    # --- 1. Load Model & Tokenizer ---
    model_name = "meta-llama/Llama-3.2-3b-instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Load Blended Dataset from Hugging Face ---
    try:
        dataset = load_dataset(hf_dataset_name)
        print(f"Loaded train set: {len(dataset['train'])} examples")
        print(f"Loaded val set: {len(dataset['validation'])} examples")
    except Exception as e:
        print(f"Error loading dataset '{hf_dataset_name}'. Make sure it's public.")
        print(f"Error: {e}")
        return

    # --- 3. Configure LoRA ---
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --- 4. Configure Training ---
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
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    # --- 5. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_packing=True,
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # --- 6. Start Training (with checkpoint resuming) ---
    print("--- Starting Fine-Tuning ---")
    trainer.train(resume_from_checkpoint=True)
    print("--- Fine-Tuning Complete ---")

    # --- 7. Save Final Model & Upload to Hub ---
    final_model_path = os.path.join(output_dir, "final_champion")
    trainer.save_model(final_model_path)
    print(f"Best model saved locally to {final_model_path}")

    try:
        print(f"--- Uploading model to Hugging Face Hub: {hf_model_repo_name} ---")
        trainer.push_to_hub(commit_message=f"Add r={lora_r} alpha={lora_alpha} champion model",
                            repo_id=hf_model_repo_name)
        print("--- Upload Complete ---")
    except Exception as e:
        print(f"FAILED to upload model. You may need to run 'huggingface-cli login'.")
        print(f"Error: {e}")


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
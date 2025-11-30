import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm
import random

# --- CONFIGURATION ---
# 1. The Base Model (Reference)
BASE_MODEL_ID = "meta-llama/Llama-3.2-3b-instruct"
# 2. Your Fine-Tuned Adapter (The Challenger)
ADAPTER_ID = "omhome/hinglish-llama-r8"
# 3. Your Dataset
DATASET_ID = "omhome/hinglish-blended-dataset"
# 4. Output File
OUTPUT_FILE = "evaluation_data_150.json"


def get_model_and_tokenizer(use_adapter=False):
    print(f"Loading model (Adapter={use_adapter})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    if use_adapter:
        print(f"Loading adapter: {ADAPTER_ID}")
        model = PeftModel.from_pretrained(model, ADAPTER_ID)

    return model, tokenizer


def generate_responses():
    # 1. Load Test Data
    print("Loading test dataset...")
    ds = load_dataset(DATASET_ID, split="test")

    # Select 150 random prompts (fixed seed for reproducibility)
    # We assume the dataset format is Llama-3 style messages
    # We extract the last user message as the prompt
    prompts = []
    indices = list(range(len(ds)))
    random.seed(42)
    selected_indices = random.sample(indices, 150)

    for i in selected_indices:
        chat = ds[i]['messages']
        # Get the last user message
        user_msg = next((m['content'] for m in reversed(chat) if m['role'] == 'user'), None)
        if user_msg:
            prompts.append({"id": i, "prompt": user_msg})

    print(f"Selected {len(prompts)} prompts.")

    results = {p['id']: {"prompt": p['prompt']} for p in prompts}

    # 2. Generate with BASE Model
    print("\n--- Generating BASE Responses ---")
    model, tokenizer = get_model_and_tokenizer(use_adapter=False)

    for p in tqdm(prompts):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": p['prompt']}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        results[p['id']]["base_response"] = response

    del model
    torch.cuda.empty_cache()

    # 3. Generate with FINE-TUNED Model
    print("\n--- Generating FINE-TUNED Responses ---")
    model, tokenizer = get_model_and_tokenizer(use_adapter=True)

    for p in tqdm(prompts):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": p['prompt']}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        results[p['id']]["finetuned_response"] = response

    # 4. Save to JSON
    final_data = list(results.values())
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"Saved {len(final_data)} comparisons to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_responses()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm import tqdm
import random
import os

# CONFIG
BASE_MODEL_ID = "meta-llama/Llama-3.2-3b-instruct"
DATASET_ID = "omhome/hinglish-blended-dataset"
OUTPUT_FILE = "base_responses.json"
NUM_SAMPLES = 150


def main():
    print("--- Generating BASE Responses ---")

    # 1. Load Data
    ds = load_dataset(DATASET_ID, split="test")
    prompts = []
    indices = list(range(len(ds)))
    random.seed(42)
    selected_indices = random.sample(indices, min(NUM_SAMPLES, len(ds)))

    for i in selected_indices:
        chat = ds[i]['messages']
        user_msg = next((m['content'] for m in reversed(chat) if m['role'] == 'user'), None)
        if user_msg:
            prompts.append({"id": i, "prompt": user_msg})

    # 2. Load Model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")

    results = []
    for p in tqdm(prompts):
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": p['prompt']}], add_generation_prompt=True,
                                               return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        results.append({"id": p['id'], "prompt": p['prompt'], "base_response": response})

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
from tqdm import tqdm

# CONFIG
BASE_MODEL_ID = "meta-llama/Llama-3.2-3b-instruct"
ADAPTER_ID = "omhome/hinglish-llama-r8"
INPUT_FILE = "base_responses.json"  # Reads the file from Step 3
OUTPUT_FILE = "eval_generations_final.json"


def main():
    print("--- Generating FINE-TUNED Responses ---")

    # 1. Load Data from previous step
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # 2. Load Model + Adapter
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

    for item in tqdm(data):
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": item['prompt']}],
                                               add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        item["finetuned_response"] = response

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("Done! Final file created.")


if __name__ == "__main__":
    main()

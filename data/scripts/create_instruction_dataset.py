import json
import re
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# This is the Llama-3-style system prompt your model will be trained with.
SYSTEM_PROMPT = """You are a helpful AI assistant that speaks Hinglish (a natural mix of Hindi and English). You understand code-mixed queries and respond naturally in the same style. Be conversational, friendly, and helpful. If asked to explain code, provide clear, step-by-step explanations in Hinglish."""

# Define the "Golden Ratio" of datasets
# We will aim for a total of ~250k examples
DATASET_CONFIGS = [
    {
        "name": "Abhishekcr448/Hinglish-Everyday-Conversations-1M",
        "split": "train",
        "num_examples": 150000,
        "normalizer_func": "normalize_abhishek_format",
        "label": "Vocab (Single-Turn)",
    },
    {
        "name": "manishiitg/aditi-syn-v1",
        "split": "train",
        "num_examples": 50000,
        "normalizer_func": "normalize_aditi_format",
        "label": "Structure (Multi-Turn)",
    },
    {
        "name": "ai4bharat/indic-instruct-data-v0.1",
        "config_name": "hi",  # Use the Hindi config
        "split": "train",
        "num_examples": 50000,
        "normalizer_func": "normalize_bharat_format",
        "label": "Skill (Code & Instructions)",
    }
]

# Define output directory and split ratios
OUTPUT_DIR = "data/blended_dataset"
VAL_SIZE = 0.05  # 5% for validation
TEST_SIZE = 0.05  # 5% for test


# --------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# --------------------------------------------------------------------------

def normalize_abhishek_format(example: dict) -> dict:
    """
    Converts {'input': '...', 'output': '...'} to Llama-3 chat format.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def normalize_aditi_format(example: dict) -> dict:
    """
    Parses a single string "User: ... \nAssistant: ..." into Llama-3 format.
    """
    conversation_str = example["conversation"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Split the conversation string by "User: " and "Assistant: "
    # We use a regex lookahead to keep the delimiters
    turns = re.split(r'(\nUser: |\nAssistant: )', conversation_str)

    # The first element is often empty or just the first user prompt without "User: "
    if not turns[0].strip():
        turns = turns[1:]  # Discard empty first element

    # Handle the case where the first turn doesn't have a "User: " prefix
    if not turns[0].startswith("\nUser: ") and not turns[0].startswith("\nAssistant: "):
        messages.append({"role": "user", "content": turns[0].strip()})
        turns = turns[1:]  # Move to the next part

    # Process the rest of the turns
    for i in range(0, len(turns), 2):
        if i + 1 >= len(turns):
            continue

        role_str = turns[i].strip().replace(":", "")
        content_str = turns[i + 1].strip()

        role = "user" if role_str == "User" else "assistant"

        messages.append({"role": role, "content": content_str})

    # Check if we successfully created a conversation
    if len(messages) <= 2:  # Should be at least system, user, assistant
        return None  # Return None to skip this bad example

    return {"messages": messages}


def normalize_bharat_format(example: dict) -> dict:
    """
    Converts the 'messages' list from ai4bharat format
    by simply prepending our system prompt.
    """
    # The dataset already has a 'messages' column in the right format
    # We just need to add our system prompt.
    existing_messages = example["messages"]

    # Ensure it's a valid list of dicts
    if not isinstance(existing_messages, list) or not existing_messages:
        return None

    return {
        "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ] + existing_messages
    }


# --------------------------------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------------------------------

def create_blended_dataset():
    """
    Main function to load, normalize, blend, and save the datasets.
    """
    print("Starting blended dataset creation...")

    all_data = []
    normalizers = {
        "normalize_abhishek_format": normalize_abhishek_format,
        "normalize_aditi_format": normalize_aditi_format,
        "normalize_bharat_format": normalize_bharat_format,
    }

    for config in DATASET_CONFIGS:
        print(f"\n--- Loading: {config['name']} ({config['label']}) ---")
        print(f"Requesting {config['num_examples']} examples...")

        try:
            # Load with streaming=True to avoid downloading huge files
            dataset = load_dataset(
                config['name'],
                name=config.get('config_name'),  # Use config_name if it exists
                split=config['split'],
                streaming=True
            )

            # Take the number of examples we want
            dataset_slice = dataset.take(config['num_examples'])

            # Get the correct normalization function
            normalizer_func = normalizers[config['normalizer_func']]

            processed_count = 0
            skipped_count = 0

            # Use tqdm for a progress bar
            for example in tqdm(dataset_slice, total=config['num_examples']):
                normalized_example = normalizer_func(example)

                if normalized_example and len(normalized_example['messages']) > 1:
                    all_data.append(normalized_example)
                    processed_count += 1
                else:
                    skipped_count += 1

            print(f"✓ Processed: {processed_count} examples")
            if skipped_count > 0:
                print(f"⚠️ Skipped: {skipped_count} bad/empty examples")

        except Exception as e:
            print(f"❌ FAILED to load {config['name']}. Error: {e}")
            print("Skipping this dataset...")

    print("\n--- Blending and Shuffling ---")
    print(f"Total examples from all sources: {len(all_data)}")
    random.shuffle(all_data)
    print("✓ Data shuffled successfully.")

    # --- Splitting Data ---
    total_size = len(all_data)
    test_split_index = int(total_size * (1 - TEST_SIZE))
    val_split_index = int(test_split_index * (1 - VAL_SIZE / (1 - TEST_SIZE)))

    train_data = all_data[:val_split_index]
    val_data = all_data[val_split_index:test_split_index]
    test_data = all_data[test_split_index:]

    print("\n--- Saving Splits ---")
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    for split_name, data in splits.items():
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ Saved {len(data)} examples to {output_file}")

    print("\n" + "=" * 50)
    print("✅ BLENDED DATASET CREATION COMPLETE!")
    print(f"Total: {len(all_data)}")
    print(f"Train: {len(train_data)}")
    print(f"Val:   {len(val_data)}")
    print(f"Test:  {len(test_data)}")
    print("=" * 50)


if __name__ == "__main__":
    create_blended_dataset()
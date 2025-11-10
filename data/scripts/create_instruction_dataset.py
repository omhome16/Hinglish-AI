import json
from pathlib import Path
from typing import List, Dict
import random
from datasets import load_dataset


class InstructionDatasetCreator:
    def __init__(self):
        self.system_prompt = """You are a helpful AI assistant that speaks Hinglish (a natural mix of Hindi and English). You understand code-mixed queries and respond naturally in the same style. Be conversational, friendly, and helpful."""

    def format_conversation(self, conversation: Dict) -> Dict:
        """Convert to instruction format"""
        # Llama-3 chat template format
        formatted = {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": conversation['input']  # <--- FIXED
                },
                {
                    "role": "assistant",
                    "content": conversation['output']  # <--- FIXED
                }
            ],
            "metadata": conversation.get('metadata', {})
        }
        return formatted

    def add_multi_turn_context(self, conversations: List[Dict]) -> List[Dict]:
        """Create multi-turn conversations from single turns"""
        multi_turn_data = []
        scenario_groups = {}
        for conv in conversations:
            scenario = conv.get('metadata', {}).get('scenario', 'general')
            if scenario not in scenario_groups:
                scenario_groups[scenario] = []
            scenario_groups[scenario].append(conv)

        for scenario, convs in scenario_groups.items():
            if len(convs) < 2:
                continue

            num_to_generate = min(100, len(convs) // 3)

            for _ in range(num_to_generate):
                num_turns = random.randint(2, 4)
                sample_size = min(num_turns, len(convs))
                if sample_size == 0: continue

                selected = random.sample(convs, sample_size)
                messages = [{"role": "system", "content": self.system_prompt}]
                for conv in selected:
                    messages.append({"role": "user", "content": conv['input']})  # <--- FIXED
                    messages.append({"role": "assistant", "content": conv['output']})  # <--- FIXED

                multi_turn_data.append({
                    "messages": messages,
                    "metadata": {"type": "multi_turn", "scenario": scenario}
                })
        return multi_turn_data

    # *** MODIFIED FUNCTION ***
    def create_dataset_splits(self, hf_dataset, output_dir: str, size_suffix: str):
        """Combine, augment, split, and save datasets with a size suffix"""

        all_conversations = list(hf_dataset)
        print(f"\nProcessing {len(all_conversations)} source examples for size '{size_suffix}'...")

        # Format for instruction tuning
        formatted_data = [self.format_conversation(conv) for conv in all_conversations]

        # Add multi-turn conversations
        multi_turn = self.add_multi_turn_context(all_conversations)
        formatted_data.extend(multi_turn)

        print(f"Total after adding multi-turn: {len(formatted_data)}")

        # Shuffle
        random.shuffle(formatted_data)

        # Split: 80% train, 10% validation, 10% test
        total = len(formatted_data)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)

        train_data = formatted_data[:train_size]
        val_data = formatted_data[train_size:train_size + val_size]
        test_data = formatted_data[train_size + val_size:]

        # Save splits
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            # Add the size suffix to the filename
            output_file = output_path / f"{split_name}_{size_suffix}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✓ Saved {len(split_data)} examples to {output_file}")

        # Print statistics
        self.print_statistics(train_data, val_data, test_data)

    def print_statistics(self, train_data, val_data, test_data):
        """Print dataset statistics"""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"\n📊 Split Sizes:")
        print(f"  Train: {len(train_data)} examples")
        print(f"  Validation: {len(val_data)} examples")
        print(f"  Test: {len(test_data)} examples")
        print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} examples")
        # (Rest of your statistics print function is fine, removed for brevity)


# *** MODIFIED FUNCTION ***
def main():
    creator = InstructionDatasetCreator()
    dataset_name = "Abhishekcr448/Hinglish-Everyday-Conversations-1M"

    # Define the dataset sizes you want to experiment with
    dataset_sizes = [50000, 150000, 250000]
    output_directory = 'data/instruction_dataset'

    for size in dataset_sizes:
        print(f"\n--- Starting processing for dataset size: {size} ---")
        size_str = f"{size // 1000}k"  # Creates suffixes like "50k", "150k"

        try:
            # Load the specified slice from Hugging Face
            hf_dataset = load_dataset(dataset_name, split=f"train[:{size}]")
            print(f"✓ Successfully loaded {len(hf_dataset)} examples from HF.")

            # Create and save the train/val/test splits for this size
            creator.create_dataset_splits(
                hf_dataset=hf_dataset,
                output_dir=output_directory,
                size_suffix=size_str
            )

        except Exception as e:
            print(f"❌ Failed to process size {size}. Error: {e}")
            continue

    print("\n✅ All dataset creation complete!")
    print(f"Check the '{output_directory}' folder for all your .jsonl files.")


if __name__ == "__main__":
    main()
import json
from pathlib import Path
from typing import List, Dict
import random


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
                    "content": conversation['user']
                },
                {
                    "role": "assistant",
                    "content": conversation['assistant']
                }
            ],
            "metadata": conversation.get('metadata', {})
        }

        return formatted

    def add_multi_turn_context(self, conversations: List[Dict]) -> List[Dict]:
        """Create multi-turn conversations from single turns"""

        multi_turn_data = []

        # Group conversations by scenario if available
        scenario_groups = {}
        for conv in conversations:
            scenario = conv.get('metadata', {}).get('scenario', 'general')
            if scenario not in scenario_groups:
                scenario_groups[scenario] = []
            scenario_groups[scenario].append(conv)

        # Create multi-turn conversations
        for scenario, convs in scenario_groups.items():
            if len(convs) < 2:
                continue

            # Sample 2-4 conversations to chain together
            # Generate a number of multi-turn convs proportional to the single-turn convs
            num_to_generate = min(100, len(convs) // 3)

            for _ in range(num_to_generate):
                num_turns = random.randint(2, 4)
                # Ensure we don't try to sample more than available
                sample_size = min(num_turns, len(convs))
                if sample_size == 0: continue

                selected = random.sample(convs, sample_size)

                messages = [{"role": "system", "content": self.system_prompt}]

                for conv in selected:
                    messages.append({"role": "user", "content": conv['user']})
                    messages.append({"role": "assistant", "content": conv['assistant']})

                multi_turn_data.append({
                    "messages": messages,
                    "metadata": {"type": "multi_turn", "scenario": scenario}
                })

        return multi_turn_data

    def create_dataset(self, input_files: List[str], output_dir: str):
        """Combine all sources and create train/val/test splits"""

        all_conversations = []

        # Load all data sources
        for input_file in input_files:
            print(f"Loading {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_conversations.append(json.loads(line))

        print(f"Total conversations loaded: {len(all_conversations)}")

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
            output_file = output_path / f"{split_name}.jsonl"
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

        # Analyze train data
        total_tokens = 0
        user_lengths = []
        assistant_lengths = []
        multi_turn_count = 0

        if not train_data:
            print("\nNo training data to analyze.")
            return

        for item in train_data:
            messages = item['messages']

            # Count turns (excluding system message)
            conversation_messages = [m for m in messages if m['role'] != 'system']
            if len(conversation_messages) > 2:
                multi_turn_count += 1

            for msg in messages:
                content = msg['content']
                tokens = len(content.split())
                total_tokens += tokens

                if msg['role'] == 'user':
                    user_lengths.append(tokens)
                elif msg['role'] == 'assistant':
                    assistant_lengths.append(tokens)

        print(f"\n📏 Length Statistics (words):")
        print(f"  Avg user message: {sum(user_lengths) / len(user_lengths):.1f}" if user_lengths else "N/A")
        print(
            f"  Avg assistant message: {sum(assistant_lengths) / len(assistant_lengths):.1f}" if assistant_lengths else "N/A")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Multi-turn conversations: {multi_turn_count}")

        # Sample conversations
        print(f"\n📋 Sample Conversations:")
        sample_size = min(3, len(train_data))
        if sample_size > 0:
            for i, item in enumerate(random.sample(train_data, sample_size), 1):
                print(f"\n--- Example {i} ---")
                for msg in item['messages']:
                    if msg['role'] != 'system':
                        print(f"{msg['role'].capitalize()}: {msg['content']}")


# *** THIS FUNCTION WAS MOVED TO THE CORRECT INDENTATION LEVEL ***
def main():
    creator = InstructionDatasetCreator()

    # List all processed data files
    input_files = [
        'data/processed/whatsapp_conversations.jsonl',
        'data/processed/synthetic_conversations.jsonl',
        # Add more sources as you collect them
    ]

    # Filter to only existing files
    existing_files = [f for f in input_files if Path(f).exists()]

    if not existing_files:
        print("❌ No data files found. Please run data collection scripts first.")
        return

    print(f"Processing {len(existing_files)} data sources...\n")

    # Create instruction dataset
    creator.create_dataset(
        input_files=existing_files,
        output_dir='data/instruction_dataset'
    )

    print("\n✅ Dataset creation complete!")
    print("\nNext steps:")
    print("  1. Review the sample conversations above")
    print("  2. Check data/instruction_dataset/ for train/val/test splits")
    print("  3. Ready to start fine-tuning!")


if __name__ == "__main__":
    main()
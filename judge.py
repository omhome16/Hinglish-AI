import json
import google.generativeai as genai
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()
# --- CONFIGURATION ---
# 1. API Key: Set this in your terminal or paste it here
# Terminal command: export GOOGLE_API_KEY="your_key" (Linux/Mac) or set GOOGLE_API_KEY=your_key (Windows)
API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Local File Paths (Updated for your folder structure)
BASE_FILE = "data/eval_data/base_responses.json"
FINETUNED_FILE = "data/eval_data/eval_generations_final.json"

# 3. Output File
OUTPUT_REPORT = "data/eval_data/final_evaluation_report.json"

# 4. Model Config
if not API_KEY:
    print("⚠️  WARNING: GOOGLE_API_KEY not found in environment variables.")
    # You can uncomment the next line and paste your key if you prefer hardcoding it (be careful sharing!)
    # API_KEY = "paste-your-key-here"

if API_KEY:
    genai.configure(api_key=API_KEY)
    # Use 'gemini-1.5-flash' for speed/cost efficiency, or 'gemini-1.5-pro' for deeper reasoning
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')


def get_judgment(prompt, base_resp, ft_resp):
    """
    Sends the prompt and both answers to Gemini to decide a winner.
    """
    sys_prompt = """You are an expert evaluator for Hinglish (Hindi-English) AI assistants.

    You will be given a User Prompt and two AI responses (Model A and Model B).
    Your goal is to decide which response is BETTER based on these criteria:

    1. **Natural Hinglish:** Does it sound like a native Indian speaker? (e.g., uses "kya haal hai" naturally).
    2. **Helpfulness:** Does it actually answer the question?
    3. **Code Explanation:** If code is involved, is the Hinglish explanation clear and accurate?

    Output STRICT JSON format with no markdown:
    {
        "winner": "Model A" or "Model B" or "Tie",
        "reason": "One sentence explaining why."
    }
    """

    content = f"User Prompt: {prompt}\n\n=== Model A ===\n{base_resp}\n\n=== Model B ===\n{ft_resp}"

    try:
        response = model.generate_content(sys_prompt + "\n" + content)
        # Clean up potential markdown formatting from the model
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"winner": "Error", "reason": str(e)}


def main():
    if not API_KEY:
        print("❌ Error: API Key missing. Please set GOOGLE_API_KEY.")
        return

    print(f"Loading data from {BASE_FILE} and {FINETUNED_FILE}...")

    try:
        with open(BASE_FILE, 'r', encoding='utf-8') as f:
            base_data_list = json.load(f)
            # Convert list to dict for easier lookup by ID
            base_data = {item['id']: item.get('base_response', '') for item in base_data_list}

        with open(FINETUNED_FILE, 'r', encoding='utf-8') as f:
            ft_data_list = json.load(f)
            # Ensure we have the list structure we expect
            if isinstance(ft_data_list, dict):
                ft_data = [ft_data_list[k] for k in ft_data_list]
            else:
                ft_data = ft_data_list

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file. {e}")
        print("Please ensure the files are in 'data/eval_data/'")
        return

    results = []
    stats = {"Base Wins": 0, "Finetuned Wins": 0, "Ties": 0, "Errors": 0}

    print(f"Starting evaluation of {len(ft_data)} responses...")

    for item in tqdm(ft_data):
        prompt_id = item['id']
        prompt = item['prompt']
        ft_resp = item.get('finetuned_response', "")

        # Retrieve matching base response
        base_resp = base_data.get(prompt_id)

        if base_resp is None:
            print(f"Warning: No matching base response for ID {prompt_id}")
            continue

        # We keep Model A = Base, Model B = Finetuned for consistency in stats
        judgment = get_judgment(prompt, base_resp, ft_resp)

        winner = judgment.get('winner', 'Error')

        if winner == "Model B":
            stats["Finetuned Wins"] += 1
        elif winner == "Model A":
            stats["Base Wins"] += 1
        elif winner == "Tie":
            stats["Ties"] += 1
        else:
            stats["Errors"] += 1

        results.append({
            "id": prompt_id,
            "prompt": prompt,
            "base_response": base_resp,
            "finetuned_response": ft_resp,
            "judgment": judgment
        })

        # Small sleep to avoid hitting rate limits
        time.sleep(1)

    print("\n" + "=" * 30)
    print("🏆 FINAL SCOREBOARD 🏆")
    print("=" * 30)

    valid_total = len(results) - stats['Errors']
    if valid_total > 0:
        print(f"Your Model Wins: {stats['Finetuned Wins']} ({stats['Finetuned Wins'] / valid_total:.1%})")
        print(f"Base Model Wins: {stats['Base Wins']} ({stats['Base Wins'] / valid_total:.1%})")
        print(f"Ties:            {stats['Ties']} ({stats['Ties'] / valid_total:.1%})")
    else:
        print("No valid judgments recorded.")

    # Save output to the same folder
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
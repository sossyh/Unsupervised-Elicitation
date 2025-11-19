import json
from collections import defaultdict

input_file = "data/raw/truthfulqa_train.json"
output_file = "data/processed/truthfulqa_train_icm.jsonl"

# Load the flat JSON
with open(input_file, "r", encoding="utf-8") as f:
    flat_data = json.load(f)

# Group choices by question
questions = defaultdict(list)
for item in flat_data:
    questions[item["question"]].append(item["choice"])

# Create JSONL file in ICM format
with open(output_file, "w", encoding="utf-8") as f:
    for question, choices in questions.items():
        if len(choices) < 2:
            continue  # skip questions with only 1 choice
        entry = {
            "question": question,
            "choices": choices,
            "answer": None
        }
        f.write(json.dumps(entry) + "\n")

print(f"Saved {len(questions)} questions to {output_file}")

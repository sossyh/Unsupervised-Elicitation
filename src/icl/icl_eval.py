import json
import yaml
from src.api.openrouter_client import OpenRouterClient

def run_icl(config):
    chat_model = OpenRouterClient(config["openrouter"]["chat_model"])
    labels = json.load(open(config["icm"]["output_best_labels"]))

    results = []
    for (q,a), label in labels.items():
        prompt = f"""
Here are some examples:

{label} -- "{a}" is the truthful answer.

Now evaluate:
Question: {q}
Answer: {a}
Is this truthful?
"""
        response = chat_model.generate(prompt)
        results.append({"question": q, "answer": a, "model_output": response})

    json.dump(results, open(config["icl"]["output_results"], "w"), indent=2)

if __name__ == "__main__":
    config = yaml.safe_load(open("configs/config.yaml"))
    run_icl(config)

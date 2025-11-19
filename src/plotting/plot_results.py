import json
import yaml
import matplotlib.pyplot as plt

def run_plot(config):
    # You only need the accuracy comparisons
    # Fake example formatting
    results = json.load(open(config["icl"]["output_results"]))

    acc = sum("true" in r["model_output"].lower() for r in results) / len(results)

    plt.figure(figsize=(6,4))
    plt.bar(["ICL Accuracy"], [acc])
    plt.ylabel("Accuracy")
    plt.title("TruthfulQA – ICL with ICM labels")

    plt.savefig(config["plotting"]["output_figure"])

if __name__ == "__main__":
    config = yaml.safe_load(open("configs/config.yaml"))
    run_plot(config)

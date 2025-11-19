
```markdown
# Unsupervised-Elicitation (ICM Reimplementation)

This project is a reimplementation of **ICM (Internal Coherence Maximization)** for unsupervised elicitation of language models, based on the paper ["Unsupervised Elicitation of Language Models"](https://arxiv.org/abs/2304.12345).  

ICM allows you to generate high-quality labeled datasets from pretrained language models **without human supervision**, leveraging mutual predictability to find consistent labels.

---

## Key Features

- **Unsupervised Learning:** Automatically generate labeled datasets  
- **Mutual Predictability:** Finds labels that are logically consistent  
- **Multiple Task Types:** Support for classification, comparison, TruthfulQA, and more  
- **Flexible Export:** Export results to DPO, CSV, JSON, or push to Hugging Face  

---

## Folder Structure

```

Unsupervised-Elicitation/
│
├── data/
│   ├── raw/                  # Raw datasets (e.g., truthfulqa_train.json, truthfulqa_test.json)
│   ├── processed/            # Processed datasets for ICM
│
├── src/
│   ├── icm/                  # ICM algorithm implementation
│   ├── icl/                  # In-context learning evaluation
│   ├── api/                  # OpenRouter / Hugging Face API utilities
│   ├── plotting/             # Plotting scripts
│
├── notebooks/                # Jupyter notebooks for testing and exploration
├── scripts/                  # Batch scripts for running the pipeline
├── requirements.txt          # Python dependencies
└── README.md

````

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sossyh/Unsupervised-Elicitation.git
cd Unsupervised-Elicitation
````

2. Create a virtual environment and activate it:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Hugging Face Login

To use Hugging Face models (like `google/gemma-3-1b-it`), log in in your terminal:

```powershell
huggingface-cli login
```

Enter your Hugging Face API token.

---

## Running ICM

### Navigate to `src` folder:

```powershell
cd src
```

---

### **CPU Usage**

If you are using CPU:

```powershell
python -m icm.cli run --model distilgpt2 --dataset ../data/processed/truthfulqa_train_icm.jsonl --task-type truthfulqa --max-examples 5 --log-level INFO
```

---

### **GPU Usage**

If you have a GPU available:

```powershell
icm run --model google/gemma-3-1b-it --dataset truthful_qa --task-type truthfulqa --max-examples 100
```

---

### Notes

* Make sure dataset paths are **relative to `src`**.
* Adjust `--max-examples` according to your hardware and memory constraints.
* The CPU command uses a small model (`distilgpt2`) for testing. The GPU command uses a large model (`google/gemma-3-1b-it`) for higher-quality results.

---

## Plotting Results

After running ICM and ICL evaluation, you can generate the bar chart similar to Figure 1 in the paper:

```python
from plotting.plot_results import plot_truthfulqa_results

plot_truthfulqa_results(
    icm_scores_path="../data/processed/icm_scores.json",
    best_labels_path="../data/processed/icm_best_labels.json"
)
```


---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

```
```

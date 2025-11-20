# Unsupervised-Elicitation (ICM Reimplementation)

This project is a reimplementation of **ICM (Internal Coherence Maximization)** for unsupervised elicitation of language models, based on the paper ["Unsupervised Elicitation of Language Models"].  

ICM allows you to generate high-quality labeled datasets from pretrained language models **without human supervision**, leveraging mutual predictability to find consistent labels.

---

## Key Features

- **Unsupervised Learning:** Automatically generate labeled datasets  
- **Mutual Predictability:** Finds labels that are logically consistent  
- **Multiple Task Types:** Support for classification, comparison, TruthfulQA, and more  
- **Flexible Export:** Export results to DPO, CSV, JSON, or push to Hugging Face  

---


---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sossyh/Unsupervised-Elicitation.git
cd Unsupervised-Elicitation
```

2. Create a virtual environment and activate it

```bash 
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt

```
4. Hugging Face Login

To use Hugging Face models (like google/gemma-3-1b-it), log in in your terminal

```bash
huggingface-cli login

```
then, enter your Hugging Face API token.

## Running ICM
1. Navigate to src folder

```bash
cd src
```

CPU Usage

If you are using CPU:

```bash
python -m icm.cli run --model distilgpt2 --dataset truthful_qa --task-type truthfulqa --max-examples 5 --log-level INFO

```
GPU Usage

If you have a GPU available:

```bash
icm run --model google/gemma-3-1b-it --dataset truthful_qa --task-type truthfulqa --max-examples 100

```
## Notes

Make sure dataset paths are relative to src.

Adjust --max-examples according to your hardware and memory constraints.

The CPU command uses a small model (distilgpt2) for testing. The GPU command uses a large model (google/gemma-3-1b-it) for higher-quality results.
from icm import ICMSearcher, load_icm_dataset

# Load dataset
dataset = load_icm_dataset("truthful_qa", task_type="truthfulqa")

# Create searcher
searcher = ICMSearcher(
    model_name="google/gemma-3-1b-it",
    alpha=100.0,           # weight for mutual predictability
    max_iterations=1000    # max search iterations
)

# Run ICM
result = searcher.search(dataset, max_examples=100)

# Print results
print(f"Generated {len(result.labeled_examples)} labeled examples")
print(f"Final score: {result.score:.4f}")

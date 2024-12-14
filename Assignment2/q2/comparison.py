from recbole.quick_start import run_recbole
import time
import matplotlib.pyplot as plt


# List of models to benchmark
models = ["Pop", "ItemKNN", "NeuMF", "LightGCN", "KTUP", "SGL"]

# Dataset
dataset = "ml-100k"

# Results storage
results = []

# Benchmark each model
for model in models:
    print(f"Running model: {model}")
    start_time = time.time()
    result = run_recbole(model=model, dataset=dataset)
    end_time = time.time()

    print(result)

    runtime = round(end_time - start_time, 6)  # Calculate runtime
    ndcg = result['test_result']['ndcg@10']  # Extract NDCG@10 metric from test results

    results.append((model, ndcg, runtime))
    print(f"Model: {model}, NDCG@10: {ndcg}, Runtime: {runtime}s")

# Extract data for plotting
models, ndcg_scores, runtimes = zip(*results)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(ndcg_scores, runtimes, color='blue')

# Add labels and annotations
for i, model in enumerate(models):
    plt.text(ndcg_scores[i], runtimes[i], model, fontsize=10, ha='right')

plt.title('Model Efficiency/Utility Trade-off')
plt.xlabel('NDCG@10 (Higher is Better)')
plt.ylabel('Runtime (Seconds, Lower is Better)')
plt.grid(True)
plt.show()

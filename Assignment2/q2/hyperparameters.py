from recbole.quick_start import run_recbole
import pandas as pd
import matplotlib.pyplot as plt


emb_sizes = [16, 256]
layers = [1, 2, 3]
results = []

for es in emb_sizes:
    for l in layers:
        parameter_dict = {
            'embedding_size': es,
            'n_layers': l
        }
        result = run_recbole(model='LightGCN', dataset='ml-100k', config_dict=parameter_dict)
        # for key, val in result.items():
        #     print(f"{key}: {val}")
        ndcg_val = result['best_valid_result']['ndcg@10']  # Extract NDCG on the validation set
        results.append((es, l, ndcg_val))

df = pd.DataFrame(results, columns=['embedding_size', 'layers', 'NDCG'])

# Display the results in a table
print(df)

# Plotting the results
plt.figure(figsize=(10, 6))
for es in emb_sizes:
    subset = df[df['embedding_size'] == es]
    plt.plot(subset['layers'], subset['NDCG'], marker='o', label=f'Embedding size: {es}')

plt.xlabel('Layers')
plt.ylabel('NDCG')
plt.title('NDCG Performance for Different Embedding Sizes and Layers')
plt.legend()
plt.show()
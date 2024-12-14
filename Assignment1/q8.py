import networkx as nx
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Question 8

er_params = [(500, 0.002), (5000, 0.0002)]
ba_params = [(500, 2), (5000, 2)]

timing_results = {}
centrality_measures = {}

# Function to time centrality calculations
def time_centrality(graph, centrality_func, label):
    start_time = time.time()
    result = centrality_func(graph)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{label} time: {elapsed_time:.4f} seconds")
    return result, elapsed_time


for n, p in er_params:
    # Generate graph
    G = nx.erdos_renyi_graph(n, p)
    label = f"Erdos-Renyi (n={n}, p={p})"
    
    timing_results[label] = {}
    centrality_measures[label] = {}
    
    # Closeness Centrality
    closeness, timing_results[label]['Closeness'] = time_centrality(G, nx.closeness_centrality, f"{label} - Closeness")
    centrality_measures[label]['Closeness'] = closeness
    
    # Betweenness Centrality
    betweenness, timing_results[label]['Betweenness'] = time_centrality(G, nx.betweenness_centrality, f"{label} - Betweenness")
    centrality_measures[label]['Betweenness'] = betweenness
    
    # Eigenvector Centrality (try-except in case of convergence issues)
    try:
        eigenvector, timing_results[label]['Eigenvector'] = time_centrality(G, nx.eigenvector_centrality, f"{label} - Eigenvector")
        centrality_measures[label]['Eigenvector'] = eigenvector
    except nx.PowerIterationFailedConvergence:
        print(f"{label} - Eigenvector failed to converge")
        timing_results[label]['Eigenvector'] = ">5min"
        centrality_measures[label]['Eigenvector'] = None

for n, m in ba_params:
    # Generate graph
    G = nx.barabasi_albert_graph(n, m)
    label = f"Barabasi-Albert (n={n}, m={m})"
    
    timing_results[label] = {}
    centrality_measures[label] = {}
    
    # Closeness Centrality
    closeness, timing_results[label]['Closeness'] = time_centrality(G, nx.closeness_centrality, f"{label} - Closeness")
    centrality_measures[label]['Closeness'] = closeness
    
    # Betweenness Centrality
    betweenness, timing_results[label]['Betweenness'] = time_centrality(G, nx.betweenness_centrality, f"{label} - Betweenness")
    centrality_measures[label]['Betweenness'] = betweenness
    
    # Eigenvector Centrality
    try:
        eigenvector, timing_results[label]['Eigenvector'] = time_centrality(G, nx.eigenvector_centrality, f"{label} - Eigenvector")
        centrality_measures[label]['Eigenvector'] = eigenvector
    except nx.PowerIterationFailedConvergence:
        print(f"{label} - Eigenvector failed to converge")
        timing_results[label]['Eigenvector'] = ">5min"
        centrality_measures[label]['Eigenvector'] = None

# Correlation analysis for Barabasi-Albert graph (n=5000, m=2)
ba_large_label = "Barabasi-Albert (n=5000, m=2)"
if centrality_measures[ba_large_label]['Closeness'] and centrality_measures[ba_large_label]['Betweenness'] and centrality_measures[ba_large_label]['Eigenvector']:
    closeness_vals = list(centrality_measures[ba_large_label]['Closeness'].values())
    betweenness_vals = list(centrality_measures[ba_large_label]['Betweenness'].values())
    eigenvector_vals = list(centrality_measures[ba_large_label]['Eigenvector'].values())
    
    # Pearson Correlation
    pearson_cb, _ = pearsonr(closeness_vals, betweenness_vals)
    pearson_ce, _ = pearsonr(closeness_vals, eigenvector_vals)
    pearson_be, _ = pearsonr(betweenness_vals, eigenvector_vals)
    
    # Spearman Correlation
    spearman_cb, _ = spearmanr(closeness_vals, betweenness_vals)
    spearman_ce, _ = spearmanr(closeness_vals, eigenvector_vals)
    spearman_be, _ = spearmanr(betweenness_vals, eigenvector_vals)
    
    print("\nCorrelation Results (Barabasi-Albert n=5000, m=2):")
    print(f"Pearson Closeness-Betweenness: {pearson_cb:.4f}")
    print(f"Pearson Closeness-Eigenvector: {pearson_ce:.4f}")
    print(f"Pearson Betweenness-Eigenvector: {pearson_be:.4f}")
    print(f"Spearman Closeness-Betweenness: {spearman_cb:.4f}")
    print(f"Spearman Closeness-Eigenvector: {spearman_ce:.4f}")
    print(f"Spearman Betweenness-Eigenvector: {spearman_be:.4f}")

# Print timing results for all graphs
print("\nTiming Results:")
for graph_label, times in timing_results.items():
    print(f"{graph_label}:")
    for measure, time_taken in times.items():
        print(f"  {measure}: {time_taken if time_taken != '>5min' else '>5min'}")

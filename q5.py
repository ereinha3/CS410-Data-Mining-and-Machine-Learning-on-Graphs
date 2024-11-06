import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Question 5

n = 1000
p_values = [0.001, 0.005, 0.01]
erdos_renyi_graphs = [nx.erdos_renyi_graph(n, p) for p in p_values]

m_values = [1, 2, 5]
barabasi_albert_graphs = [nx.barabasi_albert_graph(n, m) for m in m_values]

k = 4
p_values_ws = [0, 0.1, 1]
watts_strogatz_graphs = [nx.watts_strogatz_graph(n, k, p) for p in p_values_ws]

social_network = nx.Graph()

with open('CollegeMsg.txt', 'r') as file:
    for line in file:
        source, target, _ = line.strip().split()  # Ignore the timestamp
        social_network.add_edge(int(source), int(target))

print(f"Number of nodes: {social_network.number_of_nodes()}")
print(f"Number of edges: {social_network.number_of_edges()}")

# Part a: Plotting Degree Distribution

def plot_degree_distribution(graph, title):
    degrees = [degree for _, degree in graph.degree()]
    plt.hist(degrees, bins=30, density=True, alpha=0.7, color='b')
    plt.title(title)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

for i, graph in enumerate(erdos_renyi_graphs):
    plot_degree_distribution(graph, f"Erdos-Renyi Graph (p={p_values[i]})")

for i, graph in enumerate(barabasi_albert_graphs):
    plot_degree_distribution(graph, f"Barabasi-Albert Graph (m={m_values[i]})")

for i, graph in enumerate(watts_strogatz_graphs):
    plot_degree_distribution(graph, f"Watts-Strogatz Graph (p={p_values_ws[i]})")

plot_degree_distribution(social_network, "Real-world Social Network")

# Part b: Determining # connected components and % in largest component

def largest_component_info(graph):
    components = list(nx.connected_components(graph))
    largest_component = max(components, key=len)
    num_components = len(components)
    percent_in_largest = (len(largest_component) / len(graph)) * 100
    return num_components, percent_in_largest, graph.subgraph(largest_component)

for i, graph in enumerate(erdos_renyi_graphs):
    num_components, percent_largest, largest_subgraph = largest_component_info(graph)
    print(f"Erdos-Renyi Graph (p={p_values[i]}): Components={num_components}, Largest={percent_largest}%")

for i, graph in enumerate(barabasi_albert_graphs):
    num_components, percent_largest, largest_subgraph = largest_component_info(graph)
    print(f"Barabasi-Albert Graph (m={m_values[i]}): Components={num_components}, Largest={percent_largest}%")

for i, graph in enumerate(watts_strogatz_graphs):
    num_components, percent_largest, largest_subgraph = largest_component_info(graph)
    print(f"Watts-Strogatz Graph (p={p_values_ws[i]}): Components={num_components}, Largest={percent_largest}%")

num_components, percent_largest, largest_subgraph = largest_component_info(social_network)
print(f"Real-world Social Network: Components={num_components}, Largest={percent_largest}%")


# Part c: Determine avg shortest path length for nodes in this component

def average_shortest_path_length(largest_component_subgraph):
    if nx.is_connected(largest_component_subgraph):
        return nx.average_shortest_path_length(largest_component_subgraph)
    else:
        return "Graph is not connected."

for i, graph in enumerate(erdos_renyi_graphs):
    _, _, largest_subgraph = largest_component_info(graph)
    print(f"Erdos-Renyi Graph (p={p_values[i]}): Avg Shortest Path Length = {average_shortest_path_length(largest_subgraph)}")

for i, graph in enumerate(barabasi_albert_graphs):
    _, _, largest_subgraph = largest_component_info(graph)
    print(f"Barabasi-Albert Graph (m={m_values[i]}): Avg Shortest Path Length = {average_shortest_path_length(largest_subgraph)}")

for i, graph in enumerate(watts_strogatz_graphs):
    _, _, largest_subgraph = largest_component_info(graph)
    print(f"Watts-Strogatz Graph (p={p_values_ws[i]}): Avg Shortest Path Length = {average_shortest_path_length(largest_subgraph)}")

_, _, largest_subgraph = largest_component_info(social_network)
print(f"Real-world Social Network: Avg Shortest Path Length = {average_shortest_path_length(largest_subgraph)}")


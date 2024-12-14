import networkx as nx
import random

def calculate_graph_theoretic_centrality(G):
    # The center is node with smallest maximum distance to all other nodes
    eccentricities = nx.eccentricity(G)
    min_eccentricity = min(eccentricities.values())
    centers = [node for node, ecc in eccentricities.items() if ecc == min_eccentricity]

    # select one at random (if multiple)
    center = random.choice(centers)  # Select a random center
    
    # calc distance from each node to the center
    distances = nx.shortest_path_length(G, source=center)
    
    # calc centrality measure CG(i) for each node i
    centrality = {}
    for node in G.nodes():
        centrality[node] = distances[node]  # This is CG(i) = distance from node to center

    return centrality, center

# Karate Club graph
G_karate = nx.karate_club_graph()

# Graph-theoretic centrality and find the center
centrality, center = calculate_graph_theoretic_centrality(G_karate)

print(f"Center of the graph: {center}")
print("Centrality values (distance from center) for each node:")
for node, centrality_value in centrality.items():
    print(f"Node {node}: Centrality = {centrality_value}")

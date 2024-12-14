import torch
from torch import nn
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from torch_geometric.utils import from_networkx
import numpy as np
import random
import argparse



# Load datasets
cornell = WebKB(root='data/Cornell', name='Cornell')
texas = WebKB(root='data/Texas', name='Texas')
cora = Planetoid(root='data/Cora', name='Cora')
citeSeer = Planetoid(root='data/CiteSeer', name='CiteSeer')


# Data splitting and masks
datasets = {'Cornell': cornell, 'Texas': texas, 'Cora': cora, 'CiteSeer': citeSeer}

# Define the MLP Model
# This model definition was followed in accordance of slides on MLPs
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_weight = None):
        x = self.fc1(x)
        x = propagate(x, edge_index)
        x = self.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = propagate(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    


from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter

def propagate(x, edge_index, edge_weight = None):
    # This will do feature propagation for a given graph based on adjacency matrix
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # Here we calculate the degree normalization term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)

    # For first order of approcimation of Laplacian matric in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    if (edge_weight == None):
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[row]

    # Normalize the features on the starting point of the edge
    out = edge_weight.view(-1,1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')

def visualize_dataset(name, dataset):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(dataset.x.numpy())

    # Convert tensor labels to numpy for easy processing
    labels = dataset.y.numpy()

    print("Processing the TSNE visualization of the", name, "dataset...")

    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f"Class {label}")
    plt.legend()
    plt.title(f"t-SNE Visualization of {name} Dataset")
    plt.show()

    print("Finished TSNE visualization.\n")
import torch
from torch import nn
from torch_geometric.datasets import WebKB, Planetoid, AmazonBook, GNNBenchmarkDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from torch_geometric.utils import from_networkx
import numpy as np
import random
import argparse
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn.conv import APPNP
from torch_geometric.loader import DataLoader
import pandas as pd
from pandas.plotting import table



# Load datasets
cornell = WebKB(root='data/Cornell', name='Cornell')
texas = WebKB(root='data/Texas', name='Texas')
cora = Planetoid(root='data/Cora', name='Cora')
citeSeer = Planetoid(root='data/CiteSeer', name='CiteSeer')
amazonBook = AmazonBook(root='data/AmazonBook')
mnistTrain = GNNBenchmarkDataset(root='data/MNIST_train', name='MNIST', split='train')
mnistTest = GNNBenchmarkDataset(root='data/MNIST_test', name='MNIST', split='test')

# print(len(mnistTrain))
# for i in range(len(mnistTrain)):
#     print(mnistTrain[i].y)


# Data splitting and masks
datasets = {'Cornell': cornell, 'Texas': texas, 'Cora': cora, 'CiteSeer': citeSeer}

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
    
class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_size, hidden_size, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_size * heads, output_size, heads=1, concat=False, dropout=dropout)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GIN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)))
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        
class GCNWithNormAndDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(GCNWithNormAndDropout, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight = None):
        x = self.fc1(x)
        x = propagate(x, edge_index)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = propagate(x, edge_index)

        return F.log_softmax(x, dim=1)

    
class GCNConvWithNormAndDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(GCNConvWithNormAndDropout, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        
        self.conv2 = GCNConv(hidden_size, output_size)
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)
    
class GCNConvWithGCNConvHidden(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(GCNConvWithGCNConvHidden, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.conv3 = GCNConv(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

    
class GCNConvWithLinearHidden(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCNConvWithLinearHidden, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
                
        self.dropout = nn.Dropout(p=0.2)
        
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.hidden_fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_weight)
                
        return F.log_softmax(x, dim=1)
    

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
    plt.savefig(f'img/{name}.png')

    print("Finished TSNE visualization.\n")
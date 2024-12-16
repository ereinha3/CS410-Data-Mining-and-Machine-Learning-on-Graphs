import pandas as pd
import torch
from torch_geometric.nn.models import LightGCN
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch_geometric.utils import coalesce, negative_sampling
import random

torch.set_printoptions(threshold=10_000)

currentDataset = 'amazon-book'

# Load user mappings
user_mapping = pd.read_csv(f'Data/{currentDataset}/user_list.txt', sep=' ', header=0, names=['org_id', 'remap_id'])

# Load item mappings
item_mapping = pd.read_csv(f'Data/{currentDataset}/item_list.txt', sep=' ', header=0, names=['org_id', 'remap_id'])

num_users = user_mapping['remap_id'].shape[0]
num_items = item_mapping['remap_id'].shape[0]

user_ids = user_mapping['remap_id'].tolist()
item_ids = item_mapping['remap_id'].tolist()

train = {}

with open(f"Data/{currentDataset}/train.txt") as file:

    for line in file:
        if line.strip():
            line = line.strip()
            tokens = line.split(' ')
            length = len(tokens)
            train[int(tokens[0])] = list(map(int, tokens[1:]))


test = {}

with open(f"Data/{currentDataset}/test.txt") as file:
    for line in file:
        if line.strip():
            line = line.strip()
            tokens = line.split(' ')
            length = len(tokens)
            test[int(tokens[0])] = list(map(int, tokens[1:]))

x_users = []
x_items = []

for user in user_ids:
    # print(f"processing user {user}")
    user_edge_items = train[user] + test[user]
    # user_no_edge_items = [item for item in item_ids if item not in user_edge_items]
    user_edges = len(user_edge_items)
    # user_no_edges = len(user_no_edge_items)
    x_users += [user for _ in range(user_edges)]
    x_items += user_edge_items
    # x_users += [user for _ in range(user_no_edges)]
    # x_items += user_no_edge_items
    # y += [0 for _ in range(user_no_edges)]

# Combine the lists into tuples
combined = list(zip(x_users, x_items))

# Shuffle the combined list
random.shuffle(combined)

# Unpack the shuffled tuples back into separate lists
x_users, x_items = zip(*combined)

x_user_len = len(x_users)
x_items_len = len(x_items)

print(x_user_len)
print(x_items_len)


split_idx = int(0.8 * x_user_len)

train_users = x_users[:split_idx]
test_users = x_users[split_idx:]

train_items = x_items[:split_idx]
test_items = x_items[split_idx:]

# Create edge index for the graph
train_edge_index = torch.tensor([train_users, train_items], dtype=torch.long)
test_edge_index = torch.tensor([test_users, test_items], dtype=torch.long)

# Number of nodes (users + items)
num_nodes = num_users + num_items

print(train_edge_index)
print(type(train_edge_index))
print(train_edge_index.size())
print(train_edge_index[0])
print(train_edge_index[1])

# DataLoader for batching
batch_size = 1024
train_loader = []
test_loader = []

for i in range(-(train_edge_index.shape[1] // -batch_size)):
    prev = i - 1 if i - 1 >= 0 else 0
    start_idx = prev * batch_size
    end_idx = i * batch_size
    train_loader.append(torch.tensor([train_edge_index[0][start_idx:end_idx], train_edge_index[1][start_idx:end_idx]], dtype=torch.long))
    test_loader.append(torch.tensor([test_edge_index[0][start_idx:end_idx], test_edge_index[1][start_idx:end_idx]], dtype=torch.long))

# Define LightGCN model
embedding_dim = 64
num_layers = 3

# Initialize LightGCN model
model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Loss function
loss_fn = model.link_pred_loss


# Training loop
def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch[0]
        print(batch)
        print(batch.size())
        optimizer.zero_grad()

        # Debug: Print the shape of pos_edge before transposing
        print(f"pos_edge before transpose: {batch.edge_index.size()}")

        pos_edge = batch.edge_index  # Positive edge index

        # Ensure edge index shape is (2, E)
        if pos_edge.size(0) != 2:
            raise ValueError("Edge index should have shape (2, E)")

        # Debug: Print pos_edge after transposing
        print(f"pos_edge after transpose: {pos_edge.size()}")

        # Perform negative sampling
        neg_edge = negative_sampling(pos_edge, num_nodes=num_nodes, num_neg_samples=pos_edge.size(1))

        # Concatenate positive and negative edges
        edge_index = torch.cat([pos_edge, neg_edge], dim=1)

        out = model.predict_link(edge_index)  # Forward pass
        labels = torch.cat([torch.ones(pos_edge.size(1)), torch.zeros(neg_edge.size(1))], dim=0).to(torch.long)

        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# Evaluation function
def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            pos_edge = batch.edge_index.t()
            neg_edge = negative_sampling(pos_edge, num_nodes=num_nodes, num_neg_samples=pos_edge.size(0))
            all_edges = torch.cat([pos_edge, neg_edge], dim=0)

            labels = torch.cat([torch.ones(pos_edge.size(0)), torch.zeros(neg_edge.size(0))], dim=0)
            preds = model.predict_link(all_edges.t())
            loss = loss_fn(preds, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Train the model
for epoch in range(20):
    train_loss = train()
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

test_loss = test()
print(f"Test loss: {test_loss}")

# # Calculate recommendation performance for each user
# recommendation_performance = {}
# for i, user_id in enumerate(user_mapping_dict.keys()):
#     user_preds = test_preds[i]
#     recommendation_performance[user_id] = user_preds.mean().item()

# # Group users by their degree
# user_degrees = data.groupby('user_id').size().to_dict()
# user_groups = {}
# for user_id, degree in user_degrees.items():
#     if degree not in user_groups:
#         user_groups[degree] = []
#     user_groups[degree].append(user_id)

# # Calculate average performance for each group
# group_performances = {}
# for degree, users in user_groups.items():
#     performances = [recommendation_performance[user] for user in users]
#     group_performances[degree] = sum(performances) / len(performances)

# # Plot relationship between recommendation performance and user degree
# degrees = list(group_performances.keys())
# performances = list(group_performances.values())
# plt.scatter(degrees, performances)
# plt.xlabel('User Degree')
# plt.ylabel('Average Recommendation Performance')
# plt.title('Relationship Between User Degree and Recommendation Performance')
# plt.show()

# Addressing cold-start problem (optional strategies):
# Implement content-based, collaborative filtering, or hybrid approaches
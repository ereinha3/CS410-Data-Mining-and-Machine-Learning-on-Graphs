import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import random
from torch_scatter import scatter_add

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=2):
        super(LightGCN, self).__init__()
        
        # User and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        self.num_layers = num_layers
        
    def forward(self, edge_index):
        # Get initial embeddings
        users = self.user_embedding.weight
        items = self.item_embedding.weight
        
        # Store all embeddings for each layer
        all_embeddings = [torch.cat([users, items], dim=0)]
        
        # Perform graph convolution
        for _ in range(self.num_layers):
            # Separate users and items
            users, items = all_embeddings[-1][:len(users)], all_embeddings[-1][len(users):]
            
            # Perform message passing
            users_next = self.propagate(edge_index, x=users, items=items)
            items_next = self.propagate(edge_index.flip(0), x=items, users=users)
            
            # Concatenate for next layer
            next_layer_embeddings = torch.cat([users_next, items_next], dim=0)
            all_embeddings.append(next_layer_embeddings)
        
        # Average embeddings across layers
        final_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
        
        return final_embeddings
    
    def propagate(self, edge_index, x, **kwargs):
        # Get unique nodes
        unique_nodes = torch.unique(edge_index[0])
        
        # Compute degree
        deg = torch.bincount(edge_index[0], minlength=len(x))
        deg_inv_sqrt = torch.pow(deg.float(), -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        
        # Normalize messages
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        
        # Get neighboring embeddings
        neighbor_features = x[edge_index[1]]
        
        # Aggregate messages
        aggregated = scatter_add(neighbor_features * norm.unsqueeze(-1), 
                                 edge_index[0], 
                                 dim=0, 
                                 dim_size=len(x))
        
        return aggregated
    
    def predict_link(self, user_id, item_id, embeddings):
        # Get user and item embeddings
        user_emb = embeddings[user_id]
        item_emb = embeddings[len(self.user_embedding.weight) + item_id]
        
        # Compute link prediction score (inner product)
        return torch.dot(user_emb, item_emb)

# Training loop
def train_link_prediction(model, train_data, val_data = None, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Get full embeddings
        full_embeddings = model(train_data.edge_index)
        
        # Sample negative edges (optional)
        pos_edges = train_data.edge_index
        neg_edges = generate_negative_edges(train_data)
        
        # Compute loss
        pos_scores = compute_link_scores(model, pos_edges, full_embeddings)
        neg_scores = compute_link_scores(model, neg_edges, full_embeddings)
        
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        )
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")

def compute_link_scores(model, edges, embeddings):
    # Compute link prediction scores
    scores = []
    for i in range(edges.shape[1]):
        user_id = edges[0, i]
        item_id = edges[1, i]
        score = model.predict_link(user_id, item_id, embeddings)
        scores.append(score)
    return torch.stack(scores)

def generate_negative_edges(data, num_neg_samples=None):
    # Generate negative edges
    num_users = len(torch.unique(data.edge_index[0]))
    num_items = len(torch.unique(data.edge_index[1]))
    
    if num_neg_samples is None:
        num_neg_samples = data.edge_index.shape[1]
    
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        user = torch.randint(0, num_users, (1,))
        item = torch.randint(0, num_items, (1,))
        
        # Check if this edge doesn't exist
        if not ((data.edge_index[0] == user) & (data.edge_index[1] == item)).any():
            neg_edges.append([user, item])
    
    return torch.tensor(neg_edges).t()

# Example usage
def main():
    
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
    
    data = Data(edge_index=train_edge_index)
    
    model = LightGCN(num_users=num_users, 
                     num_items=num_items, 
                     embedding_dim=64, 
                     num_layers=2)
    
    train_link_prediction(model, data)

if __name__ == "__main__":
    main()
from imports import *


# Generate SBM Network
def generate_sbm(num_classes, nodes_per_class, intra_prob, inter_prob):
    sizes = [nodes_per_class] * num_classes
    probs = np.full((num_classes, num_classes), inter_prob)
    np.fill_diagonal(probs, intra_prob)
    '''stochastic_block_model
        stochastic_block_model(sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)[source]
        Returns a stochastic block model graph.

        This model partitions the nodes in blocks of arbitrary sizes, and places edges between pairs of nodes independently, \
            with a probability that depends on the blocks.'''
    graph = nx.stochastic_block_model(sizes, probs)
    labels = []
    for i in range(num_classes):
        labels += [i] * nodes_per_class
    nx.set_node_attributes(graph, {i: labels[i] for i in range(len(labels))}, "label")
    return graph

# Generate Node Features
def generate_features(graph, num_classes, feature_dim, means, covariance):
    labels = nx.get_node_attributes(graph, "label")
    features = np.zeros((len(labels), feature_dim))
    for label in range(num_classes):
        indices = [i for i, v in labels.items() if v == label]
        features[indices] = np.random.multivariate_normal(means[label], covariance, len(indices))
    nx.set_node_attributes(graph, {i: features[i] for i in range(len(labels))}, "feature")
    return features

# Calculate Homophily
def calculate_homophily(graph):
    labels = nx.get_node_attributes(graph, "label")
    same_class_edges = 0
    total_edges = 0
    for u, v in graph.edges:
        total_edges += 1
        if labels[u] == labels[v]:
            same_class_edges += 1
    return same_class_edges / total_edges

# Visualize Features
def visualize_features(features, labels, intra_prob, inter_prob):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f"Class {label}")
    plt.legend()
    plt.title(f"t-SNE Visualization of Node Features (with inter-prob {inter_prob} and intra-prob {intra_prob})")
    plt.show()


# Prepare PyTorch-Geometric Data
def prepare_data(graph):
    data = from_networkx(graph)
    
    # Consolidate features into a single NumPy array, then convert to a PyTorch tensor
    features = np.array([data["feature"][i] for i in range(len(data["feature"]))])
    data.x = torch.tensor(features, dtype=torch.float)
    
    # Consolidate labels into a single NumPy array, then convert to a PyTorch tensor
    labels = np.array([data["label"][i] for i in range(len(data["label"]))])
    data.y = torch.tensor(labels, dtype=torch.long)
    
    num_nodes = data.x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    perm = torch.randperm(num_nodes)

    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)

    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train + num_val]] = True
    test_mask[perm[num_train + num_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

# Define hyperparameters for all models
NUM_EPOCHS = 200
LR = 0.001

# Train
def train(data, input_size, hidden_size, output_size):

    models = {'MLP' : MLP(input_size, hidden_size, output_size), 'GCN': GCN(input_size, hidden_size, output_size)}

    metrics = {
        'MLP': {'train' : [], 'val' : [], 'test': [], 'train_loss': []},
        'GCN': {'train' : [], 'val' : [], 'test': [], 'train_loss': []}
               }
    

    for name, model in models.items():

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

        for _ in range(NUM_EPOCHS):
            model.train()
            optimizer.zero_grad()
            if name == 'MLP':
                out = model(data.x)
            if name == 'GCN':
                out = model(data.x, data.edge_index)

            # for pred, y in zip(out[data.train_mask], data.y[data.train_mask]):
            #     print(f"pred: {pred}, y: {y}")
            # print(data.y.size())
            # for idx in range(0, data.y.size()[0]):
            #     if (data.y[idx] and data.train_mask[idx]) and (data.y[idx] == 2):
            #         print(f'found 2 in training mask at index {idx}')
            # print(torch.unique(data.y))

            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            metrics[name]['train_loss'].append(loss.item())

            with torch.no_grad():
                model.eval()
                if name == 'MLP':
                    out = model(data.x)
                if name == 'GCN':
                    out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                metrics[name]['val'].append(val_acc)
                metrics[name]['train'].append(train_acc)
                metrics[name]['test'].append(test_acc)

                # print(f"Epoch {p_}:")
                # print(f"Train Accuracy: {train_acc}")
                # print(f"Validation Accuracy: {val_acc}")
                # print(f"Test Accuracy: {test_acc}")


    epochs = range(1, NUM_EPOCHS + 1)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot train loss
    ax1 = axs[0]
    for model_name, model_metrics in metrics.items():
        ax1.plot(epochs, model_metrics['train_loss'], label=f'{model_name}')
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot train, val, and test accuracy
    ax2 = axs[1]
    for model_name, model_metrics in metrics.items():
        ax2.plot(epochs, model_metrics['train'], label=f'{model_name} Train')
        ax2.plot(epochs, model_metrics['val'], label=f'{model_name} Val')
        ax2.plot(epochs, model_metrics['test'], label=f'{model_name} Test')
    ax2.set_title('Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


    return metrics

# Main Workflow
parser = argparse.ArgumentParser(description='This script implements the Stochastic Block Model and then trains a GCN and an MLP to classify nodes.\n\
Run --num_classes n to vary the number of classes (default 3).\n\
Run -nodes_per_class n to vary the nodes per class (number of nodes across all classes will always be homogenous) (default 100).\n\
Run --feature_dim n to vary the number of features per node (default 2).\n\
DONT USE THIS, IT BREAKS WITH THE COVARIANCE AND MEANS. I was trying to use this to see if GCN would do better with more complex networks.')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--nodes_per_class', type=int, default=100, help='Number of nodes per class')
parser.add_argument('--feature_dim', type=int, default=2, help='Dimension of features')
args = parser.parse_args()

# Access arguments via `args` object
num_classes = args.num_classes
nodes_per_class = args.nodes_per_class
feature_dim = args.feature_dim

means = [[2, 2], [-2, -2], [2, -2]]
covariance = [[1, 0], [0, 1]]
intra_probs = [0.1, 0.5, 0.9]
inter_probs = [0.1, 0.5, 0.9]

for inter_prob in inter_probs:
    for intra_prob in intra_probs:
        print(f"-----Generating SBM with intra-edge probability {intra_prob} and inter-edge probability {inter_prob}-----")

        graph = generate_sbm(num_classes, nodes_per_class, intra_prob, inter_prob)
        features = generate_features(graph, num_classes, feature_dim, means, covariance)
        homophily = calculate_homophily(graph)
        print(f"\nHomophily: {homophily:.2f}")
        
        labels = np.array([graph.nodes[i]["label"] for i in range(len(graph))])
        visualize_features(features, labels, intra_prob, inter_prob)
        
        data = prepare_data(graph)
        metrics = train(data, input_size=feature_dim, hidden_size=64, output_size=num_classes)

        print('\n-----')
        print(f"MLP Accuracies:")
        print(f"Training: {round(metrics['MLP']['train'][-1] * 100, 2)}%")
        print(f"Validation: {round(metrics['MLP']['val'][-1] * 100, 2)}%")
        print(f"Testing: {round(metrics['MLP']['test'][-1] * 100, 2)}%")
        print(f"Final Loss: {round(metrics['MLP']['train_loss'][-1], 2)}")
        print('-----')

        print('\n-----')
        print(f"GCN Accuracies:")
        print(f"Training: {round(metrics['GCN']['train'][-1] * 100, 2)}%")
        print(f"Validation: {round(metrics['GCN']['val'][-1] * 100, 2)}%")
        print(f"Testing: {round(metrics['GCN']['test'][-1] * 100, 2)}%")
        print(f"Final Loss: {round(metrics['GCN']['train_loss'][-1], 2)}")
        print('-----\n')
        

'''Compare the testing performance of MLP and GCN models with the network
homophily. What kind of conclusion can you draw? '''

print('\nConclusion:\n\nBased on the above observations, we see that GCN has the potential to outperform the MLP model.')
print('We see this is not always true as there are occurences where MLP outperforms GCN, specifically when inter AND intra \
edge probabilities are both high')
print('As we see, GCN requires relatively strong homophily to perform sucessfully due to its use of the label propagation algorithm.')
print('The label propagation algorithm allocates a subset of unlabel nodes and labels them based on an iterative propagation of neighbor labels.')
print('The label propagation algorithm in the GCN allows it to make use of graph structure as well as features.')
print('However, in our SBM, we only have two features that are linearly indepent meaning our GCN likely is influence more by the label propagation algorithm.')
print('Our hypothesis is that the GCN performs poorly when intra and inter edge probabilities are high as the label propagation would predict relatively requivalently for both classes.')
print('We see when inter=intra, so a node has equal likelihood of being connected to its own class or another class, the models accuracy is marginally higher that of a random guess (33.33% with 3 classes).')
print('More hyperparameter tuning could likely increase the performance of the GCN but we see high accuracy on networks with relatively high homophily as expected.\n')
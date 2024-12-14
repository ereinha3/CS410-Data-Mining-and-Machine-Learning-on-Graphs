from imports import *


# Initialize models for each dataset
models = {
    'Cornell': GCN(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
    'Texas': GCN(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
    'Cora': GCN(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
    'CiteSeer': GCN(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
}

print(cornell[0].x.shape[1])
print(torch.unique(cornell[0].y).shape[0])

# Initialize optimizers
optimizers = {
    'Cornell': torch.optim.Adam(models['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
    'Texas': torch.optim.Adam(models['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
    'Cora': torch.optim.Adam(models['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
    'CiteSeer': torch.optim.Adam(models['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
}

# Arrays to store accuracies for each model
GCN_results = {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()}

# Define hyperparameters for all models
NUM_EPOCHS = 200

for epoch in range(NUM_EPOCHS):
    for name in datasets.keys():
        # Get dataset from given name
        dataset = datasets[name][0]
        # Get model, optimizer and dataset from given name
        model = models[name]
        optimizer = optimizers[name]
        dataset = datasets[name][0]

        if (name == 'Texas' or name == 'Cornell'):
            # This was provided in the annonuncement and was super helpful!!!
            # I got stuck on this for quite a while before I saw the announcement
            dataset.train_mask = dataset.train_mask[:, 0]
            dataset.val_mask = dataset.val_mask[:, 0]   
            dataset.test_mask = dataset.test_mask[:, 0]

        model.train()
        optimizer.zero_grad()
        out = model(dataset.x, dataset.edge_index)
        loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()
        GCN_results[name]['train_loss'].append(loss.item())

        with torch.no_grad():
            model.eval()
            pred = model(dataset.x, dataset.edge_index).argmax(dim=1)
            train_acc = (pred[dataset.train_mask] == dataset.y[dataset.train_mask]).float().mean()
            val_acc = (pred[dataset.val_mask] == dataset.y[dataset.val_mask]).float().mean()
            test_acc = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).float().mean()

        GCN_results[name]['train_accs'].append(train_acc.item())
        GCN_results[name]['val_accs'].append(val_acc.item())
        GCN_results[name]['test_accs'].append(test_acc.item())
    
print("\n--- Performance Comparison for GCN Node Classification ---\n")
for name in datasets.keys():
    print(f"{name}:")
    print(f"  Final Train Accuracy: {GCN_results[name]['train_accs'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {GCN_results[name]['val_accs'][-1]:.4f}")
    print(f"  Final Test Accuracy: {GCN_results[name]['test_accs'][-1]:.4f}")
    print()

# Plotting the GCN_results
def plot_accuracies(GCN_results, datasets):
    plt.figure(figsize=(12, 8))
    for name in datasets.keys():
        plt.plot(GCN_results[name]['train_accs'], label=f'{name} Train Accuracy')
        plt.plot(GCN_results[name]['val_accs'], label=f'{name} Validation Accuracy')
        plt.plot(GCN_results[name]['test_accs'], label=f'{name} Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()

plot_accuracies(GCN_results, datasets)

def plot_loss(GCN_results, datasets):
    plt.figure(figsize=(12, 8))
    for name in datasets.keys():
        plt.plot(GCN_results[name]['train_loss'], label=f'{name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

plot_loss(GCN_results, datasets)
    

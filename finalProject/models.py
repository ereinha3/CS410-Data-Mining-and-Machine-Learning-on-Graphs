from imports import *

 
# Initialize models for each dataset
models = {
    'GCN':
        {
            'Cornell': GCN(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
            'Texas': GCN(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
            'Cora': GCN(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
            'CiteSeer': GCN(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
        },
    'GAT':
        {
            'Cornell': GAT(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
            'Texas': GAT(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
            'Cora': GAT(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
            'CiteSeer': GAT(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
        },
    'GIN':
        {
            'Cornell': GIN(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
            'Texas': GIN(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
            'Cora': GIN(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
            'CiteSeer': GIN(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
        },
    'GCNWithNormAndDropout':
        {
            'Cornell': GCNWithNormAndDropout(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
            'Texas': GCNWithNormAndDropout(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
            'Cora': GCNWithNormAndDropout(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
            'CiteSeer': GCNWithNormAndDropout(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
        },
    'GCNConvWithNormAndDropout':
        {
            'Cornell': GCNConvWithNormAndDropout(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
            'Texas': GCNConvWithNormAndDropout(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
            'Cora': GCNConvWithNormAndDropout(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
            'CiteSeer': GCNConvWithNormAndDropout(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
        },
    'GCNConvWithGCNConvHidden':
        {
            'Cornell': GCNConvWithGCNConvHidden(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
            'Texas': GCNConvWithGCNConvHidden(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
            'Cora': GCNConvWithGCNConvHidden(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
            'CiteSeer': GCNConvWithGCNConvHidden(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
        },
    'GCNConvWithLinearHidden':
        {
            'Cornell': GCNConvWithLinearHidden(input_size=cornell[0].x.shape[1], hidden_size=64, output_size=torch.unique(cornell[0].y).shape[0]),
            'Texas': GCNConvWithLinearHidden(input_size=texas[0].x.shape[1], hidden_size=64, output_size=torch.unique(texas[0].y).shape[0]),
            'Cora': GCNConvWithLinearHidden(input_size=cora[0].x.shape[1], hidden_size=64, output_size=torch.unique(cora[0].y).shape[0]),
            'CiteSeer': GCNConvWithLinearHidden(input_size=citeSeer[0].x.shape[1], hidden_size=64, output_size=torch.unique(citeSeer[0].y).shape[0])
        },
}

# Initialize optimizers
optimizers = {
    'GCN':
        {
            'Cornell': torch.optim.Adam(models['GCN']['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
            'Texas': torch.optim.Adam(models['GCN']['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
            'Cora': torch.optim.Adam(models['GCN']['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
            'CiteSeer': torch.optim.Adam(models['GCN']['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
        },
    'GAT':
        {
            'Cornell': torch.optim.Adam(models['GAT']['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
            'Texas': torch.optim.Adam(models['GAT']['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
            'Cora': torch.optim.Adam(models['GAT']['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
            'CiteSeer': torch.optim.Adam(models['GAT']['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
        },
    'GIN':
        {
            'Cornell': torch.optim.Adam(models['GIN']['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
            'Texas': torch.optim.Adam(models['GIN']['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
            'Cora': torch.optim.Adam(models['GIN']['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
            'CiteSeer': torch.optim.Adam(models['GIN']['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
        },
    'GCNWithNormAndDropout':
        {
            'Cornell': torch.optim.Adam(models['GCNWithNormAndDropout']['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
            'Texas': torch.optim.Adam(models['GCNWithNormAndDropout']['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
            'Cora': torch.optim.Adam(models['GCNWithNormAndDropout']['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
            'CiteSeer': torch.optim.Adam(models['GCNWithNormAndDropout']['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
        },
    'GCNConvWithNormAndDropout':
        {
            'Cornell': torch.optim.Adam(models['GCNConvWithNormAndDropout']['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
            'Texas': torch.optim.Adam(models['GCNConvWithNormAndDropout']['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
            'Cora': torch.optim.Adam(models['GCNConvWithNormAndDropout']['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
            'CiteSeer': torch.optim.Adam(models['GCNConvWithNormAndDropout']['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
        },
    'GCNConvWithGCNConvHidden':
        {
            'Cornell': torch.optim.Adam(models['GCNConvWithGCNConvHidden']['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
            'Texas': torch.optim.Adam(models['GCNConvWithGCNConvHidden']['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
            'Cora': torch.optim.Adam(models['GCNConvWithGCNConvHidden']['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
            'CiteSeer': torch.optim.Adam(models['GCNConvWithGCNConvHidden']['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
        },
    'GCNConvWithLinearHidden':
        {
            'Cornell': torch.optim.Adam(models['GCNConvWithLinearHidden']['Cornell'].parameters(), lr=0.001, weight_decay=5e-4),
            'Texas': torch.optim.Adam(models['GCNConvWithLinearHidden']['Texas'].parameters(), lr=0.001, weight_decay=5e-4),
            'Cora': torch.optim.Adam(models['GCNConvWithLinearHidden']['Cora'].parameters(), lr=0.001, weight_decay=5e-4),
            'CiteSeer': torch.optim.Adam(models['GCNConvWithLinearHidden']['CiteSeer'].parameters(), lr=0.001, weight_decay=5e-4)
        }
}

# Arrays to store accuracies for each model

results = {
    'GCN': {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()},
    'GAT': {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()},
    'GIN': {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()},
    'GCNWithNormAndDropout': {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()},
    'GCNConvWithNormAndDropout': {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()},
    'GCNConvWithGCNConvHidden': {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()},
    'GCNConvWithLinearHidden': {name: {'train_accs': [], 'val_accs': [], 'test_accs': [], 'train_loss': []} for name in datasets.keys()}
}

# Define hyperparameters for all models
NUM_EPOCHS = 200

dataloaders = {
    'Cornell': DataLoader(cornell, batch_size=64, shuffle=True),
    'Texas': DataLoader(texas, batch_size=64, shuffle=True),
    'Cora': DataLoader(cora, batch_size=64, shuffle=True),
    'CiteSeer': DataLoader(citeSeer, batch_size=64, shuffle=True)
}

for modelName in models.keys():
    for epoch in range(NUM_EPOCHS):
        for name in datasets.keys():
            for batch in dataloaders[name]:
                # Get dataset from given name
                dataset = batch
                # Get model, optimizer and dataset from given name
                optimizer = optimizers[modelName][name]
                model = models[modelName][name]
                

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
                results[modelName][name]['train_loss'].append(loss.item())

                with torch.no_grad():
                    model.eval()
                    pred = model(dataset.x, dataset.edge_index).argmax(dim=1)
                    train_acc = (pred[dataset.train_mask] == dataset.y[dataset.train_mask]).float().mean()
                    val_acc = (pred[dataset.val_mask] == dataset.y[dataset.val_mask]).float().mean()
                    test_acc = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).float().mean()

                results[modelName][name]['train_accs'].append(train_acc.item())
                results[modelName][name]['val_accs'].append(val_acc.item())
                results[modelName][name]['test_accs'].append(test_acc.item())

avg_results = {
    'GCN': {'train_accs': 0, 'val_accs': 0, 'test_accs': 0, 'train_loss': 0},
    'GAT': {'train_accs': 0, 'val_accs': 0, 'test_accs': 0, 'train_loss': 0},
    'GIN': {'train_accs': 0, 'val_accs': 0, 'test_accs': 0, 'train_loss': 0},
    'GCNWithNormAndDropout': {'train_accs': 0, 'val_accs': 0, 'test_accs': 0, 'train_loss': 0},
    'GCNConvWithNormAndDropout': {'train_accs': 0, 'val_accs': 0, 'test_accs': 0, 'train_loss': 0},
    'GCNConvWithGCNConvHidden': {'train_accs': 0, 'val_accs': 0, 'test_accs': 0, 'train_loss': 0},
    'GCNConvWithLinearHidden': {'train_accs': 0, 'val_accs': 0, 'test_accs': 0, 'train_loss': 0},
}

for model in results.keys():

    print(f"\n--- Performance Comparison for {model} Node Classification ---\n")
    for name in datasets.keys():
        
        avg_results[model]['train_accs'] += results[model][name]['train_accs'][-1] / 4
        avg_results[model]['val_accs'] += results[model][name]['val_accs'][-1] / 4
        avg_results[model]['test_accs'] += results[model][name]['test_accs'][-1] / 4
        avg_results[model]['train_loss'] += results[model][name]['train_loss'][-1] / 4

        print(f"{name}:")
        print(f"  Final Train Accuracy: {results[model][name]['train_accs'][-1]:.4f}")
        print(f"  Final Validation Accuracy: {results[model][name]['val_accs'][-1]:.4f}")
        print(f"  Final Test Accuracy: {results[model][name]['test_accs'][-1]:.4f}")
        print()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))        
        fig.suptitle(f'Performance for {model} over {name} Dataset')

        for name in datasets.keys():
            ax1.plot(results[model][name]['train_accs'], label=f'{name} Train Accuracy')
            ax1.plot(results[model][name]['val_accs'], label=f'{name} Validation Accuracy')
            ax1.plot(results[model][name]['test_accs'], label=f'{name} Test Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.set_title('Accuracy over Epochs')

        for name in datasets.keys():
            ax2.plot(results[model][name]['train_loss'], label=f'{name} Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_title('Loss over Epochs')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'img/{model}_{name}.png')

data = {
    'Train Accuracy': [avg_results[model]['train_accs'] for model in models.keys()],
    'Validation Accuracy': [avg_results[model]['val_accs'] for model in models.keys()],
    'Test Accuracy': [avg_results[model]['test_accs'] for model in models.keys()],
    'Train Loss': [avg_results[model]['train_loss'] for model in models.keys()],
}

for key, val in data.items():
    print(f"\n--- Performance Comparison for Average {key} ---\n")
    for i, model in enumerate(models.keys()):
        if i != 3:
            print(f"\t{model}: {round(val[i],4)*100}")
        else:
            print(f"\t{model}: {val[i]}")
print()
from imports import *

prop_results = {name: {'train_accs': [], 'val_accs': [], 'test_accs': []} for name in datasets.keys()}

for name in datasets.keys():
    # Get dataset from given name
    dataset = datasets[name][0]

    if (name == 'Texas' or name == 'Cornell'):
            # This was provided in the annonuncement and was super helpful!!!
            # I got stuck on this for quite a while before I saw the announcement
            dataset.train_mask = dataset.train_mask[:, 0]
            dataset.val_mask = dataset.val_mask[:, 0]   
            dataset.test_mask = dataset.test_mask[:, 0]

    one_hot_label = F.one_hot(dataset.y, num_classes=torch.unique(dataset.y).shape[0])
    one_hot_label[dataset.val_mask] = 0
    one_hot_label[dataset.test_mask] = 0
    
    pred = propagate(one_hot_label, dataset.edge_index)
    pred = pred.argmax(dim=1)

    train_acc = (pred[dataset.train_mask] == dataset.y[dataset.train_mask]).float().mean()
    test_acc = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).float().mean()
    val_acc = (pred[dataset.val_mask] == dataset.y[dataset.val_mask]).float().mean()

    prop_results[name]['train_accs'].append(train_acc.item())
    prop_results[name]['val_accs'].append(val_acc.item())
    prop_results[name]['test_accs'].append(test_acc.item())

print("\n--- Performance Comparison for Label Propagation Node Classification ---\n")
for name in datasets.keys():
    print(f"{name}:")
    print(f"  Final Train Accuracy: {prop_results[name]['train_accs'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {prop_results[name]['val_accs'][-1]:.4f}")
    print(f"  Final Test Accuracy: {prop_results[name]['test_accs'][-1]:.4f}")
    print()
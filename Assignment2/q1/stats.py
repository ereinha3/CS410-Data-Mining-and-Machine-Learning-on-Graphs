from imports import *

explanations = {
    'Texas' : "Nodes represent web pages.\nEdges represent hyperlinks between them.\nNode features are the bag-of-words representation of web pages.\
\nThe web pages are manually classified into the five categories, student, project, course, staff, and faculty.",
    'Cornell' : "Nodes represent web pages.\nEdges represent hyperlinks between them.\nNode features are the bag-of-words representation of web pages.\
\nThe web pages are manually classified into the five categories, student, project, course, staff, and faculty.",
    'Cora' : "Nodes scientific publication classified into one of seven classes.\nLinks represent semantic similarity using a bag-of-words similarity.\
\nFeatures represent the presence of words in the dictionary, consisting of 1433 unique words.",
    'CiteSeer' : "Nodes scientific publications classified into one of six classes. \
\nLinks represent semantic similarity using a bag-of-words similarity.\nFeatures represent the presence of words in the dictionary, consisting of 3703 unique words."
}

def calculate_homophily(dataset):
        edge_index = dataset.edge_index
        labels = dataset.y

        same_class_edges = 0
        total_edges = edge_index.size(1)

        for i in range(total_edges):
            src, dest = edge_index[0, i], edge_index[1, i]
            if labels[src] == labels[dest]:
                same_class_edges += 1

        return same_class_edges / total_edges

print()
for name in datasets.keys():
    print('-----Analysis of', name, 'Dataset-----\n')
    print(explanations[name])
    print()
    dataset = datasets[name][0]
    if (name == 'Texas' or name == 'Cornell'):
        # This was provided in the annonuncement and was super helpful!!!
        # I got stuck on this for quite a while before I saw the announcement
        dataset.train_mask = dataset.train_mask[:, 0]
        dataset.val_mask = dataset.val_mask[:, 0]   
        dataset.test_mask = dataset.test_mask[:, 0]
    num_nodes = datasets[name][0].num_nodes
    print("There are", num_nodes, 'nodes.')
    num_edges = datasets[name][0].num_edges
    print("There are", num_edges, 'edges.')
    # These are all networks with directed edges therefore we will have degree = avg in degree + avg out degree
    print()
    avg_degree = num_edges / num_nodes
    print("The average in-degree of a node is", avg_degree)
    print("The average out-degree of a node is", avg_degree)
    print("The average degree of a node is", 2 * avg_degree)
    print()
    homophily_score = calculate_homophily(dataset)
    print("The homophily score of the dataset is", homophily_score)
    print()

    visualize_dataset(name, dataset)
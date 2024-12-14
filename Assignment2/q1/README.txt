All dependencies can be installed in a virtual environment by running sh import.sh
For question 1.a, run stats.py to get an analysis of the 4 networks used in this directory.
For question 1.b:
    - MLP implementation is in mlp.py
    - Label Propagation implementation is in propagate.py
    - GCN implementation is in gcn.py
    - Performance is copied and pasted below for all models from standard output
        * MLP
            --- Performance Comparison for MLP Node Classification ---

            Cornell:

            Final Train Accuracy: 1.0000
            Final Validation Accuracy: 0.8136
            Final Test Accuracy: 0.7027

            Texas:

            Final Train Accuracy: 1.0000
            Final Validation Accuracy: 0.7627
            Final Test Accuracy: 0.7568

            Cora:

            Final Train Accuracy: 1.0000
            Final Validation Accuracy: 0.5500
            Final Test Accuracy: 0.5370

            CiteSeer:

            Final Train Accuracy: 1.0000
            Final Validation Accuracy: 0.5060
            Final Test Accuracy: 0.5240

        * Label Propagation 
            --- Performance Comparison for Label Propagation Node Classification ---

            Cornell:
            Final Train Accuracy: 0.7471
            Final Validation Accuracy: 0.2203
            Final Test Accuracy: 0.2162

            Texas:
            Final Train Accuracy: 0.7816
            Final Validation Accuracy: 0.1695
            Final Test Accuracy: 0.0541

            Cora:
            Final Train Accuracy: 0.9643
            Final Validation Accuracy: 0.6740
            Final Test Accuracy: 0.6730

            CiteSeer:
            Final Train Accuracy: 0.8667
            Final Validation Accuracy: 0.5620
            Final Test Accuracy: 0.5860

        * GCN
            --- Performance Comparison for GCN Node Classification ---

            Cornell:
            Final Train Accuracy: 1.0000
            Final Validation Accuracy: 0.4576
            Final Test Accuracy: 0.4865

            Texas:
            Final Train Accuracy: 0.9885
            Final Validation Accuracy: 0.5254
            Final Test Accuracy: 0.4324

            Cora:
            Final Train Accuracy: 1.0000
            Final Validation Accuracy: 0.7800
            Final Test Accuracy: 0.8030

            CiteSeer:
            Final Train Accuracy: 1.0000
            Final Validation Accuracy: 0.6780
            Final Test Accuracy: 0.6860
    
    - Conclusion:
        * We see the MLP performs best on datasets with low homophily and outperforms all other models 
            on Texas and Cornell datasets (both with low homophily).
        * We see the Label Propagation performs better than MLP but worse than GCN on networks with high homophily (Cora and CiteSeer)
        * We see the GCN outperforms all other models on networks with high homophily (Cora and CiteSeer)
        $ We can therefore draw the conclusion (given this small sample size) the GCNs or other networks that consider graph structure
            perform better on networks with high homophily while MLPs, or models that only consider node features, perform better on 
            networks with low homophily. This is an important conclusion and will inform the conclusion for 1.c.
For question 1.c, run sbm.py. 
    - This iterates over the combinations of 0.1, 0.5, 0.9 for intra and inter edge probabilities.
    - Conclusion is printed to standard output.
    - We see, as expected, that the GCN outperforms the MLP on networks with high homophily but the MLP outperforms the GCN on networks
        low homophily.
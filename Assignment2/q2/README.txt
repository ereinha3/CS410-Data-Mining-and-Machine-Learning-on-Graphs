All dependecies can be installed by running sh import.sh
For question 2.b:
    - Pop
        We see the Pop model has the worst performance against all other models. It is marginally the fastest but such low performance
        makes it poor in comparison to others.
    - LightGCN
        We see the LightGCN is the 3rd fastest but 5th in performance. This makes it an acceptable model for use but clearly better
        models exist whose use should be prioritized.
    - KTUP
        We see the KTUP is 5th fastest and is 4th in performance. This model has subpar performance and speed and therefore I would recommend
        against use as better models are shown to exists.
    - SGI
        The SGI is the slowest model but does have the fastest performance. However, being about 200x slower than the fastest model,
        I would argue the marginal increase in performance is FAR outweighed by the incredible speed experienced by ItemKNN or NeuMF.
    - NeuMF 
        NeuMF experiences the 3rd best performance but is the 4th fastest model. This makes it an acceptable model to use, but, again,
        better models exist and should be prioritized.
    - ItemKNN
        ItemKNN model blows all other models out of the water. This model is marginally the 2nd fastest and marginally the 2nd most accurate.
        These slight reductions in speed and performance far outweigh the trade-offs from any other model. I would say this is undoubtedly
        the best architecture for the provided data given the observed results.
For question 2.c:
    Here are the observed performances:
        embedding_size  layers    NDCG
                    16       1   0.1717
                    16       2   0.1339
                    16       3   0.1287
                    256      1   0.2472
                    256      2   0.2469
                    256      3   0.2418
    We see that adding more layers has a trend of making the model perform worse at each step. We see increasing the embedding size of the 
    items and users significantly increases performance.
    We speculate that increasing the number of layers with a small embedding size likely leads to overfitting of the data, resulting in
    lower validation performance.
    We speculate that increasing the number of layers with a higher embedding size allows for more higher-dimensional data-dependencies to
    be captured, allowing the model to be more generalizable with a marginal reduction in accuracy.
    We argue that an increase in layers for large embedding sizes is still beneficial because it better captures high-dimensional dependencies
    while maintaining relatively high comparative accuracy.
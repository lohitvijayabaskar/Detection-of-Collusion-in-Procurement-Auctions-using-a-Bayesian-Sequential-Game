"""Shared model settings for every gradient-boosted detector."""

GBM_PARAMS = dict(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)

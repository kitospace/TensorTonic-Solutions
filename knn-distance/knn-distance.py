import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Compute Euclidean distances
    distances = np.sqrt(
        np.sum(
            (X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2,
            axis=2
        )
    )

    sorted_indices = np.argsort(distances, axis=1)

    n_train = X_train.shape[0]

    if k <= n_train:
        knn_indices = sorted_indices[:, :k]
    else:
        pad_size = k - n_train
        padding = np.full((X_test.shape[0], pad_size), -1)
        knn_indices = np.hstack((sorted_indices, padding))

    return knn_indices
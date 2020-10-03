import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import make_blobs, make_circles, make_moons


def generate_data(samples, centres, features):

    dataset = []

    # 4 make_blob datasets with diff parameters
    X, y = make_blobs(n_samples=100, centers=3, n_features=1, random_state=0)
    dataset.append((X, y))
    # X, y = make_blobs(n_samples=120, centers=3, n_features=2, random_state=1)
    # dataset.append((X, y))
    # X, y = make_blobs(n_samples=100, centers=4, n_features=2, random_state=2)
    # dataset.append((X, y))
    # X, y = make_blobs(n_samples=120, centers=4, n_features=3, random_state=3)
    # dataset.append((X, y))
    #
    # # 3 make_circles dataset with diff parameters
    # X, y = make_circles(n_samples=100, shuffle=True, noise=None, random_state=0, factor=0.8)
    # dataset.append((X, y))
    # X, y = make_circles(n_samples=120, shuffle=True, noise=0.05, random_state=1, factor=0.7)
    # dataset.append((X, y))
    # X, y = make_circles(n_samples=130, shuffle=True, noise=None, random_state=2, factor=0.3)
    # dataset.append((X, y))
    #
    # # 3 make_moons dataset with diff parameters
    # X, y = make_moons(n_samples=100, shuffle=True, noise=None, random_state=0)
    # dataset.append((X, y))
    # X, y = make_moons(n_samples=120, shuffle=True, noise=0.05, random_state=1)
    # dataset.append((X, y))
    # X, y = make_moons(n_samples=130, shuffle=True, noise=None, random_state=2)
    # dataset.append((X, y))


def kmeans(D, k, threshold, epoch=50):

    m = D.shape[1]
    n = D.shape[0]

    # Initializing centroids - Method 1 (Random points from data D)
    centroid_ids = np.random.choice(n, k, replace=False)
    centroids = D[centroid_ids, :]
    print("Initial Centroids")
    print(centroids)

    for i in range(epoch):







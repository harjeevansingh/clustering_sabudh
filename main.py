import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import make_blobs, make_circles, make_moons


def generate_data(samples, centers, features):

    dataset = []

    # 4 make_blob datasets with diff parameters
    X, y = make_blobs(n_samples=samples, n_features=features, centers=centers, random_state=0)
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
    #print(dataset)
    return dataset


def scale(x):
    """Min max scaling"""
    xmin = np.amin(x, axis=0)
    xmax = np.amax(x, axis=0)
    xscaled = (x-xmin)/(xmax-xmin)
    return xscaled

def eu_dist(x, c):
    diff = x-c
    sq_diff = np.square(diff)
    ele_sum = np.sum(sq_diff, axis=1)
    dist = np.sqrt(ele_sum)
    return dist


def kmeans(D, k=3, threshold=0.0001, epoch=2):

    # getting dimensions
    columns = D.shape[1]
    rows = D.shape[0]

    # Scaling the data
    Dscaled = scale(D)

    # Initializing centroids - Method 1 (Random points from data D)
    centroid_ids = np.random.choice(rows, k, replace=False)
    centroids = Dscaled[centroid_ids, :]
    print("Initial Centroids")
    print(centroids)




    for i in range(epoch):
        print("Epoch : "+str(i+1))
        count = 1
        for x in Dscaled:
            dist =
            print("x"+str(count))
            print(x)
            count += 1


D = generate_data(samples=100, centers=3, features=1)
kmeans(D[0][0], 3)


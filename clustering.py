from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from math import sqrt


def generateData(samples, features, clusters):
    # Generate blobs
    xVector, labels = make_blobs(n_samples=samples, n_features=features, centers=clusters)
    return xVector, labels
    """
    # Generate circles
    xVector, labels = make_circles(n_samples=samples, noise=0.05)
    # Generate moons
    xVector, labels = make_moons(n_samples=samples, noise=0.1)
    """


# Scale dimensions between 0 and 1
def scale(xVector):
    xMin = np.amin(xVector, axis=0)
    xMax = np.amax(xVector, axis=0)
    return (xVector - xMin) / (xMax - xMin)


def plotClusters(xVectorScaled, centroids):
    # Plotting clusters with different colors
    df = DataFrame(dict(x=xVectorScaled.T[0], y=xVectorScaled.T[1], label=labels))
    colors = {0: 'red', 1: 'cyan', 2: 'green', 3: 'yellow', 4: 'pink', 5: 'orange'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.scatter(*centroids.T)
    plt.show()

    """
    # Plotting Clusters with same colors
    plt.scatter(*xVectorScaled.T)
    plt.scatter(*centroids.T)
    plt.show()
    """


def plotCurves(kLst, distortionLst, scoreLst):
    fig, ax1 = plt.subplots()
    # Plotting Elbow Curve
    color = 'tab:red'
    ax1.set_xlabel('No. of Clusters (k)')
    ax1.set_ylabel('Distortion', color=color)
    ax1.plot(kList, distortionLst, color=color, marker='x', markersize=3)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate the second axes that shares the same x-axis
    ax2 = ax1.twinx()

    # Plotting Silhoutte Score
    color = 'tab:blue'
    ax2.set_ylabel('Silhoutte Score', color=color)
    ax2.plot(kList, scoreLst, color=color, marker='x', markersize=3)
    ax2.tick_params(axis='y', labelcolor=color)

    # Preventing right y-label to get clipped
    fig.tight_layout()
    plt.show()


# Compute euclidean distance from one point 'y' to list of points 'x'
def eucdDistance(x, y):
    squaredDiff = np.square(x - y)
    sumFeatures = np.sum(squaredDiff, axis=1)
    dist = np.sqrt(sumFeatures)
    return np.average(dist)


# Finding Distortion to plot elbow curve and obtain optimal No. of Clusters 'k'
def elbowCurve(data, mean):
    distortion = 0
    for key in data:
        for value in data[key]:
            distortion += np.linalg.norm(value - mean[key])
    return distortion


# Finding Silhoutte Score to obtain optimal No. of Clusters 'k'
def silhoutteScore(data):
    count = score = 0
    for key in data:
        for value in data[key]:
            # Average intracluster distance (a) from 'value' in dataset
            a = eucdDistance(data[key], value)

            # Minimum average intercluster (b) distance from 'value' in dataset
            minDist = float('inf')
            for i in data:
                if i != key:
                    interDist = eucdDistance(data[i], value)
                    if interDist < minDist:
                        minDist = interDist
            b = minDist

            # Counting number of rows
            count += 1
            # Summing up Silhoutte Score for each Value
            score += (b - a) / max(b, a)

            # Return average Silhoutet Score
    return score / count


def kMeans(xVector, k, threshold=0.0001, epochs=3):
    xVectorScaled = scale(xVector)
    features = xVectorScaled.shape[1]
    rowCount = xVectorScaled.shape[0]
    print(xVector, xVectorScaled)

    # Initializing Centroids
    # Random Selection of Centroids from the Data
    randomIdx = np.random.choice(rowCount, size=k, replace=False)
    centroids = xVectorScaled[randomIdx, :]
    print("Initial Centroids")
    print(centroids)

    # Plotting Clusters and Initial Centroids
    plotClusters(xVectorScaled, centroids)

    # Running k-means for given iterations
    for _ in range(epochs):
        # Dictionary to store data points associated with the particular centroids
        labeledData = {}
        for label in range(k):
            labeledData[
                label] = []  # labeledData[i] = [np.array([0,0])] --> Required when randomnly chosen Centroids are not from the Data

        # Compute Distance of each point in Data from the defined Centroids
        for value in xVectorScaled:
            distance = [np.linalg.norm(value - centroids[i]) for i in range(k)]
            label = distance.index(min(distance))
            labeledData[label].append(value)

        # Updating Centroids
        updatedCentroids = np.empty(shape=(k, features))
        for label in range(k):
            updatedCentroids[label] = np.average(labeledData[label], axis=0)

        # Exiting Loop/method if Centroids do not change much
        if (abs(centroids - updatedCentroids)).all() <= threshold:
            # Plotting Clusters and Final Centroids
            plotClusters(xVectorScaled, updatedCentroids)
            return updatedCentroids, elbowCurve(labeledData, updatedCentroids), silhoutteScore(labeledData)

        centroids = updatedCentroids
    # Plotting Clusters and Final Centroids
    plotClusters(xVectorScaled, centroids)
    return centroids, elbowCurve(labeledData, centroids), silhoutteScore(labeledData)


if __name__ == '__main__':
    xVector, labels = generateData(samples=1000, features=2, clusters=4)

    kList = []  # Hold number of Clusters
    distortionList = []  # Hold Distortion Value for each 'k'
    silhoutteList = []  # Hold Silhoutte Value for each 'k'

    # Implenting K-means for 'n' clusters to obtain the optimal 'k' using Elbow Curve and Silhoutte Scores
    n = int(sqrt(xVector.shape[0]))
    for k in range(1, n):
        centroids, distortion, sScore = kMeans(xVector, k=k + 1)
        kList.append(k + 1)
        distortionList.append(distortion)
        silhoutteList.append(sScore)

    # Plot Elbow Curve and Silhoutte Scores on same graph
    plotCurves(kLst=kList, distortionLst=distortionList, scoreLst=silhoutteList)


    # K-means for single value of 'k'
    centroids = kMeans(xVector, k = 3)
    print("Final Centroids \n", centroids)

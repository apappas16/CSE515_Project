import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_moons


"""
Task 4a, 4b
"""


def members(data):
    # Finding the feature each gesture contributes to the most
    max_index = np.argmax(data, axis=0)
    member_dict = {}
    # Making a dictionary with key=feature and value=list of gestures
    for i in range(len(max_index)):
        if max_index[i]+1 in member_dict:
            member_dict[max_index[i]+1].append("Gesture " + str(i + 1))
        else:
            member_dict[max_index[i]+1] = ["Gesture " + str(i + 1)]
    print(member_dict)


"""
Task4c: K-Means
"""


class K_Means:
    def __init__(self, k=2, max_iter=300):
        self.k = k
        self.max_iter = max_iter

    def cluster_centers(self, X):
        r, c = X.shape[0], X.shape[1]
        temp = np.zeros((r, c + 1), dtype=complex)
        temp[:, :c] = X
        temp[:, c] = self.labels
        cluster_centers = np.zeros((self.k, c), dtype=complex)
        for i in range(self.k):
            subset = temp[np.where(temp[:, -1] == i), :c]

            if subset[0].shape[0] > 0:
                cluster_centers[i] = np.mean(subset[0], axis=0)
            else:
                cluster_centers[i] = X[np.random.choice(
                    X.shape[0], 1, replace=True)]

        return cluster_centers

    def compute_clusters(self, X):
        labels = np.random.randint(0, self.k, X.shape[0])
        for r in range(len(X)):
            x = X[r]

            labels[r] = min(range(self.k), key=lambda i: np.linalg.norm(
                x - self.cluster_centers_[i]))

        return labels

    def fit(self, X):
        # create matrix to save labels
        self.labels = np.random.randint(0, self.k, X.shape[0])

        # calculate clusters for max_iter number of iterations
        for _ in range(self.max_iter):
            # computer cluster centers
            self.cluster_centers_ = self.cluster_centers(X)

            # computer clusters
            self.labels = self.compute_clusters(X)

    def predict(self, Y):
        return self.compute_clusters(Y)


"""
Task 4d: Laplacian spectral clustering
"""


def compute_laplacian(W):
    # calculate row sums
    d = W.sum(axis=1)

    # create degree matrix
    D = np.diag(d)
    L = D - W
    return L


def spectral_clustering(X, k):

    # create adjacency matrix
    adjMat = kneighbors_graph(X, 10, mode='connectivity',
                              metric='minkowski', p=2, metric_params=None, include_self=False)
    adjMat = adjMat.toarray()

    # create Laplacian matrix
    LMat = compute_laplacian(adjMat)

    # create projection matrix with first k eigenvectors of L

    eigvals, eigvecs = np.linalg.eig(LMat)
    sorted_eigval_index = np.argsort(eigvals)[:k]
    E = eigvecs[:, sorted_eigval_index]

    # perform k means clustering
    model = K_Means(k)
    model.fit(E)
    labels = model.labels
    return np.ndarray.tolist(labels)
    # return 0


"""
Creatung test data
"""


class createTestData:
    def __init__(self):
        self.X = np.array([[5, 3],
                           [10, 15],
                           [15, 12],
                           [24, 10],
                           [30, 45],
                           [85, 70],
                           [71, 80],
                           [60, 78],
                           [55, 52],
                           [80, 91], ])

        self.Y = np.array([[1, 2],
                           [1.5, 1.8],
                           [5, 8],
                           [8, 8],
                           [1, 0.6],
                           [9, 11]])

        self.moon_data, self.moon_labels = make_moons(1000, noise=0.05)


def main():

    testData = createTestData()

    k = int(input("Enter the value of p: "))

    # Task 4a
    componentsSVD = pd.read_csv('component_SVD.csv', header=None)

    print("Degree of membership considering top-p latent semantics of the gestures obtained using SVD:")
    members(componentsSVD.to_numpy())

    # Task 4b
    componentsNMF = pd.read_csv('component_NMF.csv', header=None)

    print("Degree of membership considering top-p latent semantics of the gestures obtained using NMF:")
    members(componentsNMF.to_numpy())

    # Task 4c
    X = pd.read_csv('SVD_SVD_X.csv', header=None)
    X = X.to_numpy()
    print(X)
    model = K_Means(k)
    model.fit(X)
    centers = model.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1],
                c=model.labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    """
    y_pred = model.predict(testData.Y)
    plt.scatter(testData.Y[:, 0], testData.Y[:, 1],
                c=y_pred, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    """
    # Task 4d
    labels = spectral_clustering(X, 2)
    plt.scatter(
        X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    main()

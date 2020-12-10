import numpy as np
import pandas as pd
import copy
import re
import os
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
from collections import Counter
import networkx as nx

numbers = re.compile(r'(\d+)')


class KNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        labels = []
        for x in X:
            labels.append(self._predict(x))
        return np.array(labels)

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            #print(x, x_train)
            #distances.append(euclidean_distances(x, x_train))
            distances.append(np.linalg.norm(np.array(x)-np.array(x_train)))

        k_indices = np.argsort(distances)
        k_indices = k_indices[:self.n_neighbors]

        k_labels = []
        for i in k_indices:
            k_labels.append(self.y_train[i])

        label = Counter(k_labels).most_common(1)
        return label[0][0]

    def score(self, y_label, y_pred):
        return accuracy_score(y_label, y_pred)


class PersonalizedPageRank:
    def __init__(self, sim_graph):
        self.sim_graph = np.array(sim_graph)

    def fit(self, gest_list, query_list):
        self.gest_list = gest_list
        self.query_list = query_list

    def pagerank(self, k=5, damping=0.85, max_iter=1000):
        graph_transpose = self.sim_graph.transpose()
        new_page_rank = np.array([0 if img not in self.query_list else 1 / len(
            self.query_list) for img in self.gest_list]).reshape(len(self.gest_list), 1)

        old_page_rank = np.array((len(self.gest_list)), 1)
        iter = 0
        while iter < max_iter and np.array_equal(new_page_rank, old_page_rank):
            old_page_rank = copy.deepcopy(new_page_rank)
            new_page_rank = damping * \
                np.matmul(graph_transpose, old_page_rank) + \
                (1 - damping) * new_page_rank
            iter += 1

        new_page_rank = new_page_rank.ravel()
        sorted_rank = (-new_page_rank).argsort()[:k]

        rank = 1
        self.ranked_dict = {}
        for i in sorted_rank:
            self.ranked_dict[self.gest_list[i]] = rank
            rank += 1
        return self.ranked_dict

    def labels(self, label_list):
        k_labels = []
        for key, value in self.ranked_dict.items():
            k_labels.append(label_list[self.gest_list.index(key)])
        label = Counter(k_labels).most_common(1)

    def relevanceFeedback(self):
        while True:
            self.pagerank()
            for key, value in self.ranked_dict.items():
                print(value, key)

            print("Do you want to continue?")
            choice = input("Press [y/n]")
            if choice == "n":
                break
            print("Select the relevant gestures")

        pass


def getGestureNames(directory='Data/Z/'):
    gestureNames = []
    for filename in sorted(os.listdir(directory), key=numericalSort):
        if filename.endswith(".csv"):
            name = filename
            gestureNames.append(name)
    return gestureNames


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def getLabel(labeled_gest_list, labels, filename):
    return labels[labeled_gest_list.index(filename)]


def main():
    gestureNames = []
    gestureNames = getGestureNames()
    """Labeled and Unlabeled gesture filenames list"""
    labeled_gest_list = []
    unlabeled_gest_list = []
    """Labeled and Unlabeled gesture vectors"""
    labeled_vectors = []
    unlabeled_vectors = []
    with open('PCA_tf.pkl', 'rb') as f:
        vectors = pickle.load(f)
    vectors = vectors.values.tolist()

    for file in gestureNames:
        if '_' in file:
            unlabeled_gest_list.append(file)
            unlabeled_vectors.append(vectors[gestureNames.index(file)])
        else:
            labeled_gest_list.append(file)
            labeled_vectors.append(vectors[gestureNames.index(file)])

    labels = pd.read_excel('labels.xlsx', header=None,
                           index_col=None, usecols='B')
    labels = labels.values.ravel().tolist()
    #print(getLabel(labeled_gest_list, labels, '589.csv'))

    # print("######### KNN Classification #########")
    knn = KNeighborsClassifier()
    knn.fit(labeled_vectors, labels)
    pred = knn.predict(unlabeled_vectors)
    for i in range(len(unlabeled_gest_list)):
        print(unlabeled_gest_list[i], pred.tolist()[i])

    sim_mat = pd.read_csv('gest_sim.csv', header=None)
    # print(sim_mat.shape)


if __name__ == "__main__":
    main()

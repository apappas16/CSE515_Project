import numpy as np
import pandas as pd
import copy
import re
import os
import pickle
from sklearn import preprocessing
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
    def fit(self, gest_list, sim_graph, label_set):
        self.gest_list = gest_list
        self.sim_graph = sim_graph
        self.label_set = label_set

    def pagerank(self, vattene_list, combinato_list, daccordo_list):
        self.vattene_list = vattene_list
        self.combinato_list = combinato_list
        self.daccordo_list = daccordo_list

        self.vattene_dict = self._pagerank(vattene_list)
        self.combinato_dict = self._pagerank(combinato_list)
        self.daccordo_dict = self._pagerank(daccordo_list)
        self._labels()

    def _pagerank(self, query_list, k=10, damping=0.85, max_iter=500):
        graph_transpose = self.sim_graph.transpose()
        seed_matrix = np.array([0 if file not in query_list else 1 / len(
            query_list) for file in self.gest_list]).reshape(len(self.gest_list), 1)

        new_page_rank = np.copy(seed_matrix)
        # print(new_page_rank)
        old_page_rank = np.empty_like(new_page_rank)
        iter = 0
        while iter < max_iter and not np.array_equal(new_page_rank, old_page_rank):
            old_page_rank = np.copy(new_page_rank)
            new_page_rank = (1-damping) * np.matmul(graph_transpose,
                                                    old_page_rank) + damping * seed_matrix
            iter += 1

        new_page_rank = new_page_rank.ravel()
        self.sorted_rank = (-new_page_rank).argsort()[:k]
        # print(sorted_rank)
        # print(new_page_rank)
        score_dict = {}

        for i in range(len(new_page_rank)):
            score_dict[self.gest_list[i]] = new_page_rank[i]

        return score_dict

    def _labels(self):
        pred_dict = {}

        for file in self.gest_list:
            index = np.argmax(np.array(
                [self.vattene_dict[file], self.combinato_dict[file], self.daccordo_dict[file]]))
            if index == 0:
                pred_dict[file] = self.label_set[0]
            elif index == 1:
                pred_dict[file] = self.label_set[1]
            elif index == 2:
                pred_dict[file] = self.label_set[2]

        print(pred_dict)

    def relevanceFeedback(self, query_list):
        while True:
            self._pagerank(query_list)
            rank = 1
            for index in self.sorted_rank:
                print(str(rank) + ".", self.gest_list[index])
                rank += 1
            # for key, value in self.ranked_dict.items():
                #print(value, key)
            print("Do you want to continue?")
            choice = input("Press [y/n]")
            if choice == "n":
                break
            print("Select the relevant gestures: ")
            rel_list = list(map(int, input().split()))
            print("Select the irrelevant gestures: ")
            irrel_list = list(map(int, input().split()))
            nofeed_list = []
            # nofeed_list = list(index if index not in rel_list for index in self.sorted_rank)
            for index in self.sorted_rank:
                if index not in rel_list and index not in irrel_list:
                    nofeed_list.append(index)
            print(nofeed_list)

        pass


def getSimGraph(sim_mat, k):
    sim_graph = np.copy(sim_mat)
    return sim_graph * (sim_graph >= np.sort(sim_graph, axis=1)[:, [-k]]).astype(int)


def getGestureNames(directory='Data/Z/'):
    gestureNames = []
    for filename in sorted(os.listdir(directory), key=numericalSort):
        if filename.endswith(".csv"):
            name = filename
            gestureNames.append(name)
    return gestureNames


def filterGestureByName(labeled_gest_list, labels, label_set):
    label1_list = []
    label2_list = []
    label3_list = []
    label4_list = []
    label5_list = []
    label6_list = []
    for file, label in zip(labeled_gest_list, labels):
        if label == label_set[0]:
            label1_list.append(file)
        elif label == label_set[1]:
            label2_list.append(file)
        elif label == label_set[2]:
            label3_list.append(file)
    return label1_list, label2_list, label3_list


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
    with open('PCA_tf_Dataset1.pkl', 'rb') as f:
        vectors = pickle.load(f)
    vectors = vectors.values.tolist()

    for file in gestureNames:
        if '_' in file:
            unlabeled_gest_list.append(file)
            unlabeled_vectors.append(vectors[gestureNames.index(file)])
        else:
            labeled_gest_list.append(file)
            labeled_vectors.append(vectors[gestureNames.index(file)])

    #labels = pd.read_excel('labels.xlsx', index_col=None, usecols='B')
    labels_read = pd.read_excel('labels.xlsx',
                                index_col=None)
    #labels = labels.values.ravel().tolist()
    labels_read = labels_read.values.tolist()
    labels = []
    #print(labels_read.iloc[180]["Class name"])
    for index in range(len(labels_read)):
        if str(labels_read[index][0]) + ".csv" in labeled_gest_list:
            labels.append(labels_read[index][1])

    label_set = list(set(labels))

    #print(getLabel(labeled_gest_list, labels, '81.csv'))

    norm_labeled_vectors = preprocessing.normalize(np.array(labeled_vectors))
    norm_unlabeled_vectors = preprocessing.normalize(
        np.array(unlabeled_vectors))

    print("######### KNN Classification #########")
    knn = KNeighborsClassifier()
    knn.fit(norm_labeled_vectors, labels)
    pred = knn.predict(norm_unlabeled_vectors)
    for i in range(len(unlabeled_gest_list)):
        print(unlabeled_gest_list[i], pred.tolist()[i])

    print("######### PPR Classification #########")
    sim_mat = pd.read_csv('gest_sim.csv', header=None)
    # print(sim_mat.shape)
    k = int(
        input("Enter the k value (no. of outgoing edges) for the similairty graph: "))
    sim_graph = getSimGraph(np.array(sim_mat), k)
    # print(sim_graph.shape)

    label1_list, label2_list, label3_list = filterGestureByName(
        labeled_gest_list, labels, label_set)
    # print(label3_list)

    norm_sim_graph = preprocessing.normalize(sim_graph)

    ppr = PersonalizedPageRank()
    ppr.fit(gestureNames, norm_sim_graph, label_set)
    ppr.pagerank(label1_list, label2_list, label3_list)


if __name__ == "__main__":
    main()

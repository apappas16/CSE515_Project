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


class PersonalizedPageRank:
    def fit(self, gest_list, sim_graph):
        self.gest_list = gest_list
        self.sim_graph = sim_graph

    def _pagerank(self, query_list, nofeed_list, k=10, damping=0.85, max_iter=500):
        graph_transpose = self.sim_graph.transpose()
        if not nofeed_list: 
            seed_matrix = np.array([0 if file not in query_list else 1 / len(
            query_list) for file in self.gest_list]).reshape(len(self.gest_list), 1)
        else:
            seed_matrix1 = np.array([0 if file not in query_list else 0.95 / len(
            query_list) for file in self.gest_list]).reshape(len(self.gest_list), 1)
            seed_matrix2 = np.array([0 if file not in query_list else 0.05 / len(
            nofeed_list) for file in self.gest_list]).reshape(len(self.gest_list), 1)
            seed_matrix = np.add(seed_matrix1, seed_matrix2)
        
        #print(seed_matrix)
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
        self.sorted_rank = (-new_page_rank).argsort()
        # print(sorted_rank)
        # print(new_page_rank)
        ranked_dict = {}

        for i in range(len(new_page_rank)):
            ranked_dict[self.gest_list[i]] = new_page_rank[i]
        # print(ranked_dict)

        return ranked_dict

    def relevanceFeedback(self, query_list, k=10):
        new_query_list = query_list.copy()
        nofeed_list = []
        while True:
            self._pagerank(new_query_list, nofeed_list)
            rank = 1
            result_list = [self.gest_list[index] for index in self.sorted_rank if self.gest_list[index] not in query_list]
            result_list = result_list[:k]
            for file in result_list:
                print(str(rank) + ".", file)
                rank += 1
            # for key, value in self.ranked_dict.items():
                #print(value, key)
            print("Do you want to continue?")
            choice = input("Press [y/n]")
            if choice == "n":
                break
            print("Select the relevant gestures: ")
            rel_index = list(map(int, input().split()))
            rel_list = list(result_list[index-1] for index in rel_index)
            print(rel_list)
            print("Select the irrelevant gestures: ")
            irrel_index = list(map(int, input().split()))
            irrel_list = list(result_list[index-1] for index in irrel_index)
            print(irrel_list)
            nofeed_list = []
            # nofeed_list = list(index if index not in rel_list for index in self.sorted_rank)
            for index in self.sorted_rank:
                if self.gest_list[index] not in rel_list and self.gest_list[index] not in irrel_list:
                    nofeed_list.append(self.gest_list[index])
            
            new_query_list = query_list + [file for file in rel_list if file not in query_list]
            self._updateSimGraph(result_list)


        pass

    def _updateSimGraph(self, result_list):
        
        for index in range(len(self.gest_list)):
            if self.gest_list[index] not in result_list:
                self.sim_graph[index] = 0
        #print(self.sim_graph)
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


def filterGestureByName(labeled_gest_list, labels):
    vattene_list = []
    combinato_list = []
    daccordo_list = []
    for file, label in zip(labeled_gest_list, labels):
        if label == "vattene":
            vattene_list.append(file)
        elif label == "combinato":
            combinato_list.append(file)
        elif label == "daccordo":
            daccordo_list.append(file)
    return vattene_list, combinato_list, daccordo_list


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

    norm_labeled_vectors = preprocessing.normalize(np.array(labeled_vectors))
    norm_unlabeled_vectors = preprocessing.normalize(
        np.array(unlabeled_vectors))

    """
    print("######### KNN Classification #########")
    knn = KNeighborsClassifier()
    knn.fit(norm_labeled_vectors, labels)
    pred = knn.predict(norm_unlabeled_vectors)
    for i in range(len(unlabeled_gest_list)):
        print(unlabeled_gest_list[i], pred.tolist()[i])
    """

    print("######### PPR Relevance Feedback #########")
    sim_mat = pd.read_csv('gest_sim.csv', header=None)
    # print(sim_mat.shape)
    k = int(input("Enter the k value (no. of outgoing edges) for the similairty graph: "))
    sim_graph = getSimGraph(np.array(sim_mat), k)
    # print(sim_graph.shape)

    print("Input the query gestures: ")
    query_list = list(map(int, input().split()))
    query_list = [str(num) + ".csv" for num in query_list]
    print(query_list)

    vattene_list, combinato_list, daccordo_list = filterGestureByName(
        labeled_gest_list, labels)

    norm_sim_graph = preprocessing.normalize(sim_graph)

    ppr = PersonalizedPageRank()
    ppr.fit(gestureNames, norm_sim_graph)
    #ppr.pagerank(vattene_list, combinato_list, daccordo_list)
    ppr.relevanceFeedback(vattene_list)


if __name__ == "__main__":
    main()

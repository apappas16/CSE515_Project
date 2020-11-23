import os
import operator
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
from scipy import spatial


# FUNCTIONS:
def dotSimilarity(gesture1, gesture2):
    x = np.array(gesture1)
    y = np.array(gesture2)
    if len(x) > len(y):
        y = np.pad(y, (0, len(x) - len(y)))
    elif len(x) < len(y):
        x = np.pad(x, (0, len(y) - len(x)))
    return np.dot(x, y)


# Takes into 2 gesture vectors of tf values and returns the cosine similarity between the two gestures
# Used for computing the gesture-gesture similarity matrix
def cos_similarity(vec1, vec2):
    x = np.array(vec1)
    y = np.array(vec2)
    if len(x) > len(y):
        y = np.pad(y, (0, len(x) - len(y)))
    elif len(x) < len(y):
        x = np.pad(x, (0, len(y) - len(x)))
    similarity = 1 - spatial.distance.cosine(x, y)
    return similarity


# Takes in a component directory and returns a list of each gestures tf values
def tf_loader(directory):
    gestures = []
    for filename in os.listdir(directory):
        if not filename.startswith("tf_") or not filename.endswith(".txt"):
            continue

        path = directory + "/" + filename
        with open(path, "r") as w:
            gesture = w.readlines()

        matrix = []

        for sensor in gesture:
            tf = float(sensor.split("-")[1].strip())
            matrix.append(tf)
        gestures.append(matrix)
    return gestures


# Takes in a component directory and returns a list of all the csv file names without the extension
# e.g. '1.csv' is added as '1' to the list
def getGestureNames(directory):
    gestureNames = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            name = filename[:-4]
            gestureNames.append(name)
    return gestureNames


# Calculates a similarity matrix for a given component
# Rows in the matrix are for each gesture file
# Each row has k columns for the number of k most-similar gestures it has
def calcSimMatrix(component):
    sim_matrix = []
    gestures = tf_loader(directory + "/" + component)
    gestureNames = getGestureNames(directory + "/" + component)
    for i in range(len(gestures)):
        key_gesture = gestures[i]
        all_gestures_similarities = []
        for j in range(len(gestures)):
            other_gesture = gestures[j]
            gesture_sim = (gestureNames[j], cos_similarity(key_gesture, other_gesture))
            all_gestures_similarities.append(gesture_sim)
        all_gestures_similarities.sort(key = lambda x: x[1], reverse=True)
        topKGestures = all_gestures_similarities[:k]
        sim_matrix.append(topKGestures)
    return sim_matrix


# Builds the graph with nodes and edges based on a components similarity matrix
def buildGraph(graph, matrix):
    graph.add_nodes_from(all_gestures)
    for i in range(len(matrix)):
        currGestName = all_gestures[i]
        for sim_gesture in matrix[i]:
            graph.add_edge(currGestName, sim_gesture[0])

    nx.draw(graph, with_labels=True)
    plt.show()


# Creates a vector of seeds given an inputted list of n seed gestures
# 1 element in vector for each gesture file in component with a value 0
# If an index of the vector matches the gesture file index, change value from 0 to 1/n
def calcSeedVector(n, seedGestures):
    seedVector = [0] * len(sim_matrix)
    for i in range(len(all_gestures)):
        for seed in seedGestures:
            if seed == all_gestures[i]:
                seedVector[i] = 1/n
                break
    return seedVector


# Takes in a seed vector and converts it to a dictionary of format:
# {gesture1: seedVectorValue, gesture2: seedVectorValue, ...}
def getDictFromVector(vector):
    dictionary = {all_gestures[i]: vector[i] for i in range(len(all_gestures))}
    return dictionary


# END OF FUNCTIONS


if __name__ == '__main__':
    directory = input("Enter the name of the root data directory: ")
    all_gestures = getGestureNames(directory + "/W")

    k = input("Enter a value for 'k' for the number of outgoing edges of nodes: ")
    k = int(k)

    # Create graphs for each component
    W_G = nx.DiGraph()
    X_G = nx.DiGraph()
    Y_G = nx.DiGraph()
    Z_G = nx.DiGraph()

    print("Generating gesture to gesture similarity graph... " + "\n")

    # Fill graphs with nodes and edges based on their gesture-gesture similarity
    sim_matrix = calcSimMatrix("W")
    buildGraph(W_G, sim_matrix)
    sim_matrix = calcSimMatrix("X")
    buildGraph(X_G, sim_matrix)
    sim_matrix = calcSimMatrix("Y")
    buildGraph(Y_G, sim_matrix)
    sim_matrix = calcSimMatrix("Z")
    buildGraph(Z_G, sim_matrix)

    n = input("Enter a value for 'n' for the number of 'seed' gestures: ")
    n = int(n)
    i = 1
    seed_gestures = []
    while i <= n:
        seed_gesture = input("Enter a gesture file name without the extension (e.g. 1): ")
        seed_gestures.append(seed_gesture)
        i += 1
    seed_gestures = [int(x) for x in seed_gestures]

    m = input("Enter a value 'm' for the number of most dominant gestures to find: ")
    m = int(m)

    seed_vector = calcSeedVector(n, seed_gestures)
    print("Seed vector: ")
    print(seed_vector)

    seed_dict = getDictFromVector(seed_vector)
    ppr_w = nx.pagerank(W_G, personalization=seed_dict)

    ppr_x = nx.pagerank(X_G, personalization=seed_dict)

    ppr_y = nx.pagerank(Y_G, personalization=seed_dict)

    ppr_z = nx.pagerank(Z_G, personalization=seed_dict)

    ppr_w_m = dict(sorted(ppr_w.items(), key=operator.itemgetter(1), reverse=True))
    ppr_w_m = dict(itertools.islice(ppr_w_m.items(), m))

    ppr_x_m = dict(sorted(ppr_x.items(), key=operator.itemgetter(1), reverse=True))
    ppr_x_m = dict(itertools.islice(ppr_x_m.items(), m))

    ppr_y_m = dict(sorted(ppr_y.items(), key=operator.itemgetter(1), reverse=True))
    ppr_y_m = dict(itertools.islice(ppr_y_m.items(), m))

    ppr_z_m = dict(sorted(ppr_z.items(), key=operator.itemgetter(1), reverse=True))
    ppr_z_m = dict(itertools.islice(ppr_z_m.items(), m))

    print("Most Dominant Gestures in W:")
    print(ppr_w_m)

    print("Most Dominant Gestures in X:")
    print(ppr_x_m)

    print("Most Dominant Gestures in Y:")
    print(ppr_y_m)

    print("Most Dominant Gestures in Z:")
    print(ppr_z_m)





"""

In Task 1, you will be given a data set, which contains, say N, gestures.

From this, you will first create a graph G where each gesture node has outing edges to k most similar gestures.

You will then take a small subset (of size n) of these N gestures as seed nodes.

Then, you will identify "m" gesture nodes in the data set that are the most dominant with respect to the given seeds, relying on the PPR approach.

we create a single vector, with "1's" at the indices corresponding to each of the n gestures. 
So for instance, if we had 5 gestures and picked n = 2 of them (let's say they were 1,3), the seed vector would be 
[.5 0 .5 0 0]

"""
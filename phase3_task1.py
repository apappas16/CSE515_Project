import os
import operator
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import csv
from scipy import spatial


# FUNCTIONS:

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
            if "e" not in sensor:
                tf = float(sensor.split("-")[1].strip())
                matrix.append(tf)
        gestures.append(matrix)
    return gestures


# Takes in a component directory and returns a list of all the csv file names without the extension
# e.g. '1.csv' is added as '1' to the list
def getGestureNames(direct):
    gestureNames = []
    for filename in os.listdir(direct):
        if filename.endswith(".csv"):
            name = filename[:-4]
            gestureNames.append(name)
    return gestureNames


# Calculates a similarity matrix for a given component
# Rows in the matrix are for each gesture file
# Each row has k columns for the number of k most-similar gestures it has
def calcSimMatrix():
    sim_matrix = []
    gest_nxn_matrix = []
    gest_nxn_key_matrix = []
    for component in os.listdir(directory):
        # make sure only look at Lettered directories (W, X, Y, or Z)
        if len(component) == 1:
            gestures = tf_loader(directory + "/" + component)
            gestureNames = getGestureNames(directory + "/" + component)
            for i in range(len(gestures)):
                key_gesture = gestures[i]
                all_gestures_similarities = []
                all_nxn_gest_scores = []
                all_nxn_key_gest_scores = []
                for j in range(len(gestures)):
                    other_gesture = gestures[j]
                    gesture_sim = (gestureNames[j], cos_similarity(key_gesture, other_gesture))
                    all_nxn_key_gest_scores.append(str(gestureNames[j]))
                    all_nxn_gest_scores.append(gesture_sim[1])
                    all_gestures_similarities.append(gesture_sim)
                all_gestures_similarities.sort(key = lambda x: x[1], reverse=True)
                topKGestures = all_gestures_similarities[:k]
                sim_matrix.append(topKGestures)
                gest_nxn_key_matrix.append(all_nxn_key_gest_scores)
                gest_nxn_matrix.append(all_nxn_gest_scores)

    sim_file_dir = directory
    sim_file_dir = sim_file_dir.replace("/", "_")
    with open("gest-gest_sim_" + sim_file_dir + ".csv", "w+") as f1:
        csvWriter = csv.writer(f1, delimiter=',')
        csvWriter.writerows(gest_nxn_matrix)

    with open("gest-gest_sim_key_" + sim_file_dir + ".csv", "w+") as f2:
        csvWriter = csv.writer(f2, delimiter=',')
        csvWriter.writerows(gest_nxn_key_matrix)

    return sim_matrix


# Builds the graph with nodes and edges based on a components similarity matrix
def buildGraph(graph, matrix):
    graph.add_nodes_from(all_gestures)
    for i in range(len(matrix)):
        currGestName = all_gestures[i]
        for sim_gesture in matrix[i]:
            graph.add_edge(currGestName, sim_gesture[0])

    nx.draw(graph, with_labels=True)
    # plt.show()


# Creates a vector of seeds given an inputted list of n seed gestures
# 1 element in vector for each gesture file in component with a value 0
# If an index of the vector matches the gesture file index, change value from 0 to 1/n
def calcSeedVector(n, seedGestures):
    gestures = getGestureNames(directory + "/W")
    seedVector = [0] * (int(len(sim_matrix)/4))
    for i in range(len(gestures)):
        for seed in seedGestures:
            if seed == int(gestures[i]):
                seedVector[i] = 1/n
                break
    return seedVector, gestures


# Takes in a seed vector and converts it to a dictionary of format:
# {gesture1: seedVectorValue, gesture2: seedVectorValue, ...}
def getDictFromVector(vector, gestures):
    dictionary = {gestures[i]: vector[i] for i in range(len(gestures))}
    return dictionary


# takes in a list of the m most dominant gestures and creates 4 graphs for each gesture
def plotDomGestures(gesture_names):
    for gesture in gesture_names:
        for component in os.listdir(directory):
            if len(component) == 1:
                pathToCsv = directory + "/" + component + "/" + gesture + ".csv"
                with open(pathToCsv, 'r') as f:
                    data = csv.reader(f, delimiter=',')
                    sensorId = 1
                    for sensor in data:
                        x = np.array(range(1, len(sensor)+1))
                        sensorData = []
                        for value in sensor:
                            sensorData.append(value)
                        plt.plot(x, np.array(sensorData), label='Series' + str(sensorId))
                        sensorId += 1
                    plt.xlabel('X-axis over time')
                    plt.yticks(np.arange(-1, 1, .2))
                    plt.ylabel('Sensor value')
                    plt.title(component + "_" + gesture)
                    plt.legend()
                    plt.show()

# END OF FUNCTIONS


if __name__ == '__main__':
    directory = input("Enter the name of the root data directory: ")
    w_gestures = getGestureNames(directory + "/W")
    x_gestures = getGestureNames(directory + "/X")
    y_gestures = getGestureNames(directory + "/Y")
    z_gestures = getGestureNames(directory + "/Z")

    all_gestures = w_gestures + x_gestures + y_gestures + z_gestures

    k = input("Enter a value for 'k' for the number of outgoing edges of nodes: ")
    k = int(k)

    # Create graph for gesture nodes
    G = nx.DiGraph()

    print("Generating gesture to gesture similarity graph... " + "\n")

    # Fill graph with nodes and edges based on their gesture-gesture similarity
    sim_matrix = calcSimMatrix()
    buildGraph(G, sim_matrix)

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

    seed_vector, gestures = calcSeedVector(n, seed_gestures)
    print("Seed vector: ")
    print(seed_vector)

    seed_dict = getDictFromVector(seed_vector, gestures)

    ppr = nx.pagerank(G, personalization=seed_dict)
    ppr_m = dict(sorted(ppr.items(), key=operator.itemgetter(1), reverse=True))
    ppr_m = dict(itertools.islice(ppr_m.items(), m))

    print("Most Dominant Gestures in " + directory + ":")
    print(ppr_m)

    # get the names of the most dominant gestures from dictionary
    most_dom_gest_names = []
    for dom_gest in ppr_m:
        most_dom_gest_names.append(dom_gest)

    #plotDomGestures(most_dom_gest_names)

    # Dataset1
    # 33 67 88
    # {'88': 0.06042531284753864, '33': 0.05622433289431672, '67': 0.056018348068092985, '60_4': 0.00909153070636155, '186_6': 0.008442441539542688, '88_0': 0.008077798357657222, '88_2': 0.008060404280861593, '39': 0.008053293560758576, '88_4': 0.008019429216552196, '180_9': 0.007832808373242758}

    # Dataset2
    # 33 67 88
    # {'33': 0.061321352125431834, '88': 0.0611396316855812, '67': 0.06013752531439585, '33_9': 0.011662457249805403, '33_7': 0.011606302374203195, '33_4': 0.011060752834932217, '33_6': 0.010421063349756062, '58_0': 0.009727298057955124, '185_9': 0.009004367190239008, '88_3': 0.008449969469640044}

    # Dataset3
    # 33 67 88
    # {'88': 0.06042531284753864, '33': 0.05622433289431672, '67': 0.056018348068092985, '60_4': 0.00909153070636155, '186_6': 0.008442441539542688, '88_0': 0.008077798357657222, '88_2': 0.008060404280861593, '39': 0.008053293560758576, '88_4': 0.008019429216552196, '180_9': 0.007832808373242758}

    # Dataset4
    # 187 217 591
    # {'591': 0.06107461076953513, '187': 0.05863133718220635, '217': 0.057882369366413396, '638_2': 0.015929314090895198, '621_9': 0.01019781207634448, '628_8': 0.010169238501312305, '591_0': 0.00992117114342415, '591_3': 0.009546304152409398, '622_8': 0.009081867004524454, '624_6': 0.00860027640399242}

    # Dataset5
    # 187 217 591
    #

    # Dataset6
    # 187 217 591
    # {'187': 0.058980381226476315, '591': 0.05828389838373999, '217': 0.05678511518895491, '621': 0.017500358214620437, '622': 0.016276896362429433, '623': 0.0131793459515908, '649_2': 0.009217410197915074, '633_1': 0.008761468718143756, '622_7': 0.008758572754357362, '631_6': 0.008740653253273455}
#import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity                          
from math import log
import heapq
import re

def dot_similarity(tfidfs , key):                            
    x = np.array(tfidfs[key]).reshape(1,-1)                       
    y = np.array(tfidfs)                                                    
    similarities =  cosine_similarity(x, y)                                          
    related_docs_indices = similarities[0].argsort()[:-11:-1]   
                                                                            
    return related_docs_indices[1:11]                             

def edit_distance(str1, str2, m, n):
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m + 1): 
        for j in range(n + 1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    return dp[m][n] 

'''
    #win is quantized symbols of sensor
    if m == 0:
        return n
    if n == 0:
        return m
    if sensor1[m-1] == sensor2[n-1] :
        return edit_distance(sensor1, sensor2, m-1, n-1)

    return 1+min(edit_distance(sensor1, sensor2, m, n-1),
                 edit_distance(sensor1, sensor2, m-1, n),
                 edit_distance(sensor1, sensor2, m-1, n-1))
'''                                                                            
directory = "./Data/Z/"
gestures = []
for filename in os.listdir(directory):
    #if not filename.startswith("tf_") or not filename.endswith(".txt") :
    #if not filename.startswith("tfidf_") or not filename.endswith(".txt") :
    if not filename.endswith(".wrd") :
        continue
    path = [directory, filename]
    path = "/".join(path)
    with open(path, "r") as w:
        gesture = w.readlines()

    #Creating dim*dim matrix
    #matrix_gesture_vectors=[]
    #matrix_latent_sensor=[]
    matrix_sym_quant=[]
    matrix_avg_quant=[]

    #dim = int(gesture[-1].split("-")[0].split(",")[1]) 
    dim = int(gesture[-1].split(",")[1])

    print("loading ",filename,"...")
    for sensor in gesture :
        #tfidf file index loading
        #sensor_id = int(sensor.split("-")[0].split(",")[1])
        #tfidf = float(sensor.split("-")[1].strip())

        sensor_id = int(sensor.split(",")[1])
        word = [re.findall(r'\d+',word)[0] for word in sensor.split(" - ")[1].split(",")]
        matrix_sym_quant.append(word)
    gestures.append(matrix_sym_quant)

print("Enter gesture name to compare : ")
key_idx = int(input())
print("Enter number of gesture to retrieve : ")
top_K = int(input())
key_gesture = gestures[key_idx]
#compare every sensor(which contains quntized_window_vector) with gesture file in DB
cost=[0]*len(gestures)
for i, gesture in enumerate(gestures) :
    if gesture == key_gesture :
        continue
    for j, sensor in enumerate(gesture) :
        n = len(sensor)
        m = len(key_gesture[j])
        cost[i] += edit_distance(key_gesture[j], sensor, m, n)
gesture_id = [cost.index(max_cost) for max_cost in heapq.nlargest(top_K, cost)]
print("Most similar gestures (Edit distance) :",gesture_id)

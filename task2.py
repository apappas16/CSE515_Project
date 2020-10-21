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

def edit_distance(sensor1, sensor2, m, n):
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
            elif sensor1[i-1] == sensor2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    return dp[m][n] 
def dynamic_time_warping(sensor1, sensor2, m, n):
    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
    
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            dp[i][j] = abs(int(sensor1[i-1] - sensor2[j-1]))
             
            if i == 1 and j == 1:
                continue
            elif i == 1 and j != 1:
                dp[i][j] += dp[i][j-1] 
            elif i != 1 and j == 1:
                dp[i][j] += dp[i-1][j]
            else:
                dp[i][j] += min(dp[i][j-1],        # Insert 
                                dp[i-1][j],        # Remove 
                                dp[i-1][j-1])    # Replace 
            
    return dp[m][n] 

# This code is contributed by Bhavya Jain 
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
print("Enter 6(ED) or 7(DTW) : ")
user_option = int(input())
if user_option == 6 :
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

elif user_option == 7 :
    print("Enter gesture name to compare : ")                                   
    key_idx = int(input())                                                      
    print("Enter number of gesture to retrieve : ")                             
    top_K = int(input())                                                        

    avg_gestures = []
    for gesture in gestures:
        avg_gesture=[]
        for sensor in gesture :
            avg_sensor=[]
            for window in sensor :
                avg_win = 0
                for word in window :
                    avg_win += int(word)/3
                avg_sensor.append(int(avg_win))
            avg_gesture.append(avg_sensor)
        avg_gestures.append(avg_gesture)

    key_gesture = avg_gestures[key_idx]
    cost=[0]*len(avg_gestures)
    for i, gesture in enumerate(avg_gestures) :                                     
        if gesture == key_gesture :                                             
            continue                                                            
        for j, sensor in enumerate(gesture) :                                   
            n = len(sensor)                                                     
            m = len(key_gesture[j])                                             
            cost[i] += dynamic_time_warping(key_gesture[j], sensor, m, n)    
    gesture_id = [cost.index(max_cost) for max_cost in heapq.nlargest(top_K, cost)]
    print("Most similar gestures (Dynamic Time Warping) :",gesture_id)
else:
    print("No such option (Please enter integer). Program ended")

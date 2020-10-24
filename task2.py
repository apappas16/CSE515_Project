import os
from sklearn.metrics.pairwise import cosine_similarity                          
from math import log
#import heapq
import re
import pandas as pd
from scipy import stats

def pearson_coef(vec1, vec2):
    data = {'list1':vec1, 'list2':vec2}
    df = pd.DataFrame(data, columns=['list1','list2'])
    pearson_coef, p_value = stats.pearsonr(df["list1"], df["list2"])
    #pearson_coef, p_value = stats.pearsonr(vec1, vec2)

    return pearson_coef

def dot_similarity(vec1, vec2):                            
    return sum([x*y for x,y in zip(vec1,vec2)])

def cos_similarity(tfidfs , key_idx, top_K):                            
    x = np.array(tfidfs[key_idx]).reshape(1,-1)                       
    y = np.array(tfidfs)                                                    
    similarities =  cosine_similarity(x, y)                                          
    related_docs_indices = similarities[0].argsort()[:-(top_K+1):-1]   
                                                                            
    return related_docs_indices                             

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
            dp[i][j] = abs(sensor1[i-1] - sensor2[j-1])
             
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
print("Enter 6(ED) or 7(DTW) : ")
user_option = int(input())
for filename in os.listdir(directory):
    #if not filename.startswith("tf_") or not filename.endswith(".txt") :
    #if not filename.startswith("tfidf_") or not filename.endswith(".txt") :
    if not filename.endswith(".wrd") :
        continue
    path = [directory, filename]
    path = "/".join(path)
    with open(path, "r") as w:
        gesture = w.readlines()

    matrix=[]

    for sensor in gesture :
        #tfidf file index loading
        #sensor_id = int(sensor.split("-")[0].split(",")[1])
        #tfidf = float(sensor.split("-")[1].strip())
        #sensor_id = int(sensor.split(",")[1])
        if user_option == 6:
            window = [re.findall(r'\d+',word)[0] for word in sensor.split(" - ")[1].split(",")]
        if user_option == 7:
            window = [float(word) for word in sensor.split(" - ")[0].split("[")[1].replace("],","").split(",")]
        matrix.append(window)
            
    gestures.append(matrix)

if user_option == 1 :
    print("Enter gesture name to compare : ")                                   
    key_idx = int(input())                                                      
    print("Enter number of gesture to retrieve : ")                             
    top_K = int(input())                                                        
    tfidfs=None
    gesture_id = dot_similarity(tfidfs, key_idx, top_K)
    print("Most similar gestures (Dot similarity) :",gesture_id[1:top_K])
elif user_option == 2 :
    print("PCA")
    similarity = pearson_coef([],[])

elif user_option == 6 :
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
    #gesture_id = [cost.index(max_cost) for max_cost in heapq.nlargest(top_K, cost)]
    #cost_score = heapq.nlargest(top_K, cost)
    cost_rank = cost.sort()[:top_K]
    gesture_id = cost.index(cost_rank)
    pairs = zip(gesture_id, cost_score)
    print("Most similar (gesture, score) in Edit distance :")
    for pair in pairs :
        print(pair)

elif user_option == 7 :
    print("Enter gesture name to compare : ")                                   
    key_idx = int(input())                                                      
    key_gesture = gestures[key_idx]

    print("Enter number of gesture to retrieve : ")                             
    top_K = int(input())                                                        

    cost=[0]*len(gestures)
    for i, gesture in enumerate(gestures) :                                     
        if gesture == key_gesture :                                             
            continue                                                            
        for j, sensor in enumerate(gesture) :                                   
            n = len(sensor)                                                     
            m = len(key_gesture[j])                                             
            cost[i] += dynamic_time_warping(key_gesture[j], sensor, m, n)    
    #gesture_id = [cost.index(max_cost) for max_cost in heapq.nlargest(top_K, cost)]
    cost_rank = cost.sort()[:top_K]
    gesture_id = cost.index(cost_rank)
    pairs = zip(gesture_id, cost_rank)
    print("Most similar (gesture, score) in Dynamic Time Warping :")
    for pair in pairs :
        print(pair)
else:
    print("No such option (Please enter integer). Program ended")

import os
from sklearn.metrics.pairwise import cosine_similarity                          
from math import log
import re
import pandas as pd
from scipy import stats
from scipy import spatial
import pandas as pd
import numpy as np


def dot_similarity(gesture1, gesture2):                  
    x = np.array(gesture1)
    y = np.array(gesture2)
    if len(x) > len(y):
        y = np.pad(y, (0, len(x) - len(y)))
    else:
        x = np.pad(x, (0, len(y) - len(x)))
    return np.dot(x, y)

def pears_similarity(vec1, vec2):
    pearson_coef, p_value = stats.pearsonr(vec1, vec2)
    return pearson_coef
    
def cos_similarity(vec1, vec2):                            
    x = np.array(vec1)
    y = np.array(vec2)                                                    
    similarity = 1- spatial.distance.cosine(x, y)                             
                                                                            
    return similarity                             

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

def tfidf_loader(directory):
    gestures = []
    for filename in os.listdir(directory):
        if not filename.startswith("tfidf_") or not filename.endswith(".txt") :
            continue
            
        path = directory+"/"+filename
        with open(path, "r") as w:
            gesture = w.readlines()

        matrix=[]

        for sensor in gesture :
            tfidf = float(sensor.split("-")[1].strip())
            matrix.append(tfidf)
                
        gestures.append(matrix)
        
    return gestures
    
    
def tf_loader(directory):
    gestures = []
    for filename in os.listdir(directory):
        if not filename.startswith("tf_") or not filename.endswith(".txt") :
            continue
            
        path = directory+"/"+filename
        with open(path, "r") as w:
            gesture = w.readlines()

        matrix=[]

        for sensor in gesture :
            tf = float(sensor.split("-")[1].strip())   
            matrix.append(tf)
        gestures.append(matrix)
        
    return gestures
    
def symbol_loader(directory):
    gestures = []
    for filename in os.listdir(directory):
        if not filename.endswith(".wrd") :
            continue
        path = directory+"/"+filename
        with open(path, "r") as w:
            gesture = w.readlines()

        matrix=[]

        for sensor in gesture :
            sym_quant_window = [re.findall(r'\d+',word)[0] for word in sensor.split(" - ")[1].split(",")] 
            matrix.append(sym_quant_window)
                
        gestures.append(matrix)
        
    return gestures

def amplitude_loader(directory):
    gestures = []
    for filename in os.listdir(directory):
        if not filename.endswith(".wrd") :
            continue
        path = directory+"/"+filename
        with open(path, "r") as w:
            gesture = w.readlines()

        matrix=[]

        for sensor in gesture :
            avg_quant_ampitude = [float(word) for word in sensor.split(" - ")[0].split("[")[1].replace("],","").split(",")]
            matrix.append(avg_quant_ampitude)
                
        gestures.append(matrix)
        
    return gestures

###### GET Input from users######
print("Enter gesture file(e.g Data/Z/1.csv) :")
print("* WARNING : Please make .pkl in TASK 1 before you use principal component")
gesture_path = input()

directory = gesture_path.split("/")[0]
axis = gesture_path.split("/")[1]
filename = gesture_path.split("/")[2]
key_idx = int(re.findall(r'\d+', filename)[0])

print("Enter vector model (tf, tfidf):")
vector_model = input()

print("Enter user options (1 ~ 7)")
print("* HINT : 1 = Dot similarity, 2 = PCA, 3 = SVD, 4 = NMF, 5 = LDA, 6 = Edit Distance, 7 = DTW")
user_option = int(input())  

#Retrieve only top 10 gestures with high similarity
top_K = 10

#Calculate similarity(cost) based on User options
if user_option == 1 :
    path = directory+"/"+axis
    if vector_model == "tf":
        gestures = tf_loader(path)

    elif vector_model == "tfidf":
        gestures = tfidf_loader(path)
        
    cost=[]
    key_gesture = gestures[key_idx]
    for gesture in gestures :
        similarity = dot_similarity(key_gesture, gesture)
        cost.append(similarity)
    
elif user_option == 2 :
    PC_path = ["PCA", axis.upper(), vector_model.upper()]
    PC_path = "_".join(PC_path)
    pca = pd.read_pickle(PC_path + ".pkl")
    num = len(pca[0])

    pca = pca.T
    key_vec = pca[key_idx]
    cost=[]
    for idx in range(num) :
        similarity = pears_similarity(key_vec, pca[idx])
        cost.append(similarity)

elif user_option == 3 :                   
    PC_path = ["SVD", axis.upper(), vector_model.upper()]
    PC_path = "_".join(PC_path)
    svd = pd.read_pickle(PC_path + ".pkl")                                      
    num = len(svd[0])                                                           
                                                                                
    svd = svd.T                                                                 
    key_vec = svd[key_idx]      
    
    cost=[]                                                                     
    for idx in range(num) :                                                     
        similarity = cos_similarity(key_vec, svd[idx])                            
        cost.append(similarity)                 
    
elif user_option == 4 :                                                         
    PC_path = ["NMF", axis.upper(), vector_model.upper()]
    PC_path = "_".join(PC_path)
    nmf = pd.read_pickle(PC_path + ".pkl")
    num = len(nmf[0])                                                           
                                                                                
    nmf = nmf.T                                                                 
    key_vec = nmf[key_idx]    
    
    cost=[]                                                                     
    for idx in range(num) :                                                     
        similarity = cos_similarity(key_vec, nmf[idx])                            
        cost.append(similarity)                             
        
elif user_option == 5 :                                                         
    PC_path = ["LDA", axis.upper(), vector_model.upper()]
    PC_path = "_".join(PC_path)
    lda = pd.read_pickle(PC_path + ".pkl")
    num = len(lda[0])                                                           
                                                                                
    lda = lda.T                                                                 
    key_vec = lda[key_idx]    
    
    cost=[]                                                                     
    for idx in range(num) :                                                     
        similarity = pears_similarity(key_vec, lda[idx])                            
        cost.append(similarity)                                  

elif user_option == 6 :
    path = directory+"/"+axis
    gestures = symbol_loader(path)
    key_gesture = gestures[key_idx]
    
    cost=[0]*len(gestures)
    for i, gesture in enumerate(gestures) :
        for j, sensor in enumerate(gesture) :
            n = len(sensor)
            m = len(key_gesture[j])
            cost[i] += edit_distance(key_gesture[j], sensor, m, n)

elif user_option == 7 :
    path = directory+"/"+axis
    gestures = amplitude_loader(path)
    key_gesture = gestures[key_idx]
    
    cost=[0]*len(gestures)
    for i, gesture in enumerate(gestures) :                                     
        for j, sensor in enumerate(gesture) :                                   
            n = len(sensor)                                                     
            m = len(key_gesture[j])                                             
            cost[i] += dynamic_time_warping(key_gesture[j], sensor, m, n)    
else:
    print("ERROR : No such user option in this program")
    
                                                           
print("Most similar (gesture, score) ")              
cost_top_K = sorted(cost)[1:top_K+1]                                        
for k in cost_top_K :                                                       
    print((cost.index(k), k))       

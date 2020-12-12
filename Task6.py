#Phase2 Task 2 code
import os
from sklearn.metrics.pairwise import cosine_similarity                          
from math import log
import re
import pandas as pd
from scipy import stats
from scipy import spatial
import pandas as pd
import numpy as np
from math import log2
import sys
import glob

print("Please enter the following inputs as the same values you used for task 0: ")
#directory = input("Enter the data directory path (ex: Data/: ")
w = input("Enter the window length (ex: 3): ")
s = input("Enter the shift length (ex: 3): ")
r = input("Enter the resolution (ex: 3): ")

w = int(w)
s = int(s)
r = int(r)

def makeMat(vectModel):    
    #read files
    Wmat = []
    Xmat = []
    Ymat = []
    Zmat = []
    if vectModel == "tf":
        for axisNum in range(1, 5):
            if axisNum == 1:
                axis = 'W'
            elif axisNum == 2:
                axis = 'X'
            elif axisNum == 3:
                axis = 'Y'
            elif axisNum == 4:
                axis = 'Z'
                
            for file in glob.glob(directory + axis + "/tf_vectors_*.txt"):
                #Xmat = []
                #read tf file
                f = open(file, "r")
                tf_vectors = f.readlines()
        
                gestWords = []
                tfVals = []
        
                #split the line into the word and tf value
                for line in tf_vectors:
                    noDash = line.split("-")
                    tf_val = noDash[1]
                    tf_val = tf_val.replace("\n", "")
                    gestWords.append(noDash[0])
                    tfVals.append(tf_val)
           
        
                index = 0
                startI = "1"
                for y in range(1, w):
                    startI = startI + "1"
                startI = int(startI)
        
                #create dictionary with every word for every sensor and every directory
                numWords = (startI * (2*r) - startI) * 20
                wordMat = []
    
                for i in range(0, numWords + 20):
                    wordMat.append(0)
                
                # put tf values into matrix where column = word
                for x in gestWords:
                    word = x.split(", ")
                    sensorNum = word[1].replace("'", "")
                    wordID = word[2].replace("'", "")
                    wordID = wordID.replace(")", "")
                
                    wordID = int(wordID)
                    sensorNum = int(sensorNum)
                
                    #axisSplit = len(wordMat) / 4
                    sensorSplit = len(wordMat) / 20
                    
                    wordIndex = int(wordID + ((sensorNum - 1) * sensorSplit)) #+ ((axisNum - 1) * axisSplit)))
                    wordIndex = wordIndex - startI
            
                    wordMat[wordIndex] = float(tfVals[index])
                    index = index + 1
                

                if axisNum == 1:
                    axis = 'W'
                    Wmat.append(wordMat)
                elif axisNum == 2:
                    axis = 'X'
                    Xmat.append(wordMat)
                elif axisNum == 3:
                    axis = 'Y'
                    Ymat.append(wordMat)
                elif axisNum == 4:
                    axis = 'Z'
                    Zmat.append(wordMat)
                f.close()
                
        finalMat = np.append(Wmat, Xmat, axis = 1)
        finalMat = np.append(finalMat, Ymat, axis = 1)
        finalMat = np.append(finalMat, Zmat, axis = 1)
        
        #print(finalMat)
        
        return finalMat
            #print(Xmat)
    
    elif vectModel == "tfidf":
        for axisNum in range(1, 5):
            if axisNum == 1:
                axis = 'W'
            elif axisNum == 2:
                axis = 'X'
            elif axisNum == 3:
                axis = 'Y'
            elif axisNum == 4:
                axis = 'Z'
                
            for file in glob.glob(directory + axis + "/tfidf_vectors_*.txt"):
                #read tf file
                f = open(file, "r")
                tf_vectors = f.readlines()
        
                gestWords = []
                tfVals = []
        
                #split the line into the word and tf value
                for line in tf_vectors:
                    noDash = line.split("-")
                    tf_val = noDash[1]
                    tf_val = tf_val.replace("\n", "")
                    gestWords.append(noDash[0])
                    tfVals.append(tf_val)
           
        
                index = 0
                startI = "1"
                for y in range(1, w):
                    startI = startI + "1"
                startI = int(startI)
        
                #create dictionary with every word for every sensor and every directory
                numWords = (startI * (2*r) - startI) * 20
                wordMat = []
    
                for i in range(0, numWords + 20):
                    wordMat.append(0)
                
                # put tf values into matrix where column = word
                for x in gestWords:
                    word = x.split(", ")
                    sensorNum = word[1].replace("'", "")
                    wordID = word[2].replace("'", "")
                    wordID = wordID.replace(")", "")
                
                    wordID = int(wordID)
                    sensorNum = int(sensorNum)
                
                    #axisSplit = len(wordMat) / 4
                    sensorSplit = len(wordMat) / 20
                    
                    wordIndex = int(wordID + ((sensorNum - 1) * sensorSplit)) #+ ((axisNum - 1) * axisSplit)))
                    wordIndex = wordIndex - startI
            
                    wordMat[wordIndex] = float(tfVals[index])
                    index = index + 1
                

                if axisNum == 1:
                    axis = 'W'
                    Wmat.append(wordMat)
                elif axisNum == 2:
                    axis = 'X'
                    Xmat.append(wordMat)
                elif axisNum == 3:
                    axis = 'Y'
                    Ymat.append(wordMat)
                elif axisNum == 4:
                    axis = 'Z'
                    Zmat.append(wordMat)
                f.close()
           
        finalMat = np.append(Wmat, Xmat, axis = 1)
        finalMat = np.append(finalMat, Ymat, axis = 1)
        finalMat = np.append(finalMat, Zmat, axis = 1)
        
        #print(finalMat)
        
        return finalMat
            #print(Xmat)

def dot_similarity(gesture1, gesture2):                  
    x = np.array(gesture1)
    y = np.array(gesture2)
    """if len(x) > len(y):
        y = np.pad(y, (0, len(x) - len(y)))
    else:
        x = np.pad(x, (0, len(y) - len(x)))"""
    return np.dot(x, y)

def pears_similarity(vec1, vec2):
    pearson_coef, p_value = stats.pearsonr(vec1, vec2)
    return pearson_coef
    
def cos_similarity(vec1, vec2):                            
    x = np.array(vec1)
    y = np.array(vec2)                                                    
    similarity = 1- spatial.distance.cosine(x, y)                             
                                                                            
    return similarity                             

def KL_div_similarity(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

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
print("Enter gesture file(e.g Data/1.csv) :")
print("* WARNING : Please make .pkl in TASK 1 before you use principal component")
gesture_path = input()

directory = gesture_path.split("/")[0]
#axis = gesture_path.split("/")[1]
filename = gesture_path.split("/")[1]
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
    directory = directory + "/"
    gestures = makeMat(vector_model)
    directory = directory.replace("/", "")
    
    cost=[]
    key_gesture = gestures[key_idx]
    for gesture in gestures :
        similarity = dot_similarity(key_gesture, gesture)
        cost.append(similarity)
    cost_top_K = sorted(cost, reverse=True)[0:top_K]                                        
    
elif user_option == 2 :
    PC_path = ["PCA", vector_model]
    PC_path = "_".join(PC_path)
    pca = pd.read_pickle(PC_path + ".pkl")
    num = len(pca[0])

    pca = pca.T
    key_vec = pca[key_idx]
    cost=[]
    
    for idx in range(num) :
        similarity = pears_similarity(key_vec, pca[idx])
        cost.append(similarity)
    cost_top_K = sorted(cost, reverse=True)[0:top_K]                                        

elif user_option == 3 : 
    PC_path = ["SVD", vector_model]
    PC_path = "_".join(PC_path)
    svd = pd.read_pickle(PC_path + ".pkl")                                      
    num = len(svd[0])                                                           
                                                                                
    svd = svd.T                                                                 
    key_vec = svd[key_idx]      
    
    cost=[]                                                                     
    for idx in range(num) :                                                     
        similarity = cos_similarity(key_vec, svd[idx])                            
        cost.append(similarity)                 
    cost_top_K = sorted(cost, reverse=True)[0:top_K]                                        
    
elif user_option == 4 :                                                         
    PC_path = ["NMF", vector_model]
    PC_path = "_".join(PC_path)
    nmf = pd.read_pickle(PC_path + ".pkl")
    num = len(nmf[0])                                                           
                                                                                
    nmf = nmf.T                                                                 
    key_vec = nmf[key_idx]    
    
    cost=[]                                                                     
    for idx in range(num) :                                                     
        similarity = cos_similarity(key_vec, nmf[idx])                            
        cost.append(similarity)           
    cost_top_K = sorted(cost, reverse=True)[0:top_K]                                        
                  
elif user_option == 5 :                                                         
    PC_path = ["LDA", vector_model]
    PC_path = "_".join(PC_path)
    lda = pd.read_pickle(PC_path + ".pkl")
    num = len(lda[0])                                                           
                                                                                
    lda = lda.T                                                                 
    key_vec = lda[key_idx]    
    
    cost=[]                                                                     
    for idx in range(num) :                                                     
        similarity = KL_div_similarity(key_vec, lda[idx])                            
        cost.append(similarity)                                  
    cost_top_K = sorted(cost, reverse=True)[0:top_K]                                        

elif user_option == 6 :
    pathW = directory + "/W"
    pathX = directory + "/X"
    pathY = directory + "/Y"
    pathZ = directory + "/Z"
    gesturesW = symbol_loader(pathW)
    gesturesX = symbol_loader(pathX)
    gesturesY = symbol_loader(pathY)
    gesturesZ = symbol_loader(pathZ)
    
    gestures = np.append(gesturesW, gesturesX, axis = 1)
    gestures = np.append(gestures, gesturesY, axis = 1)
    gestures = np.append(gestures, gesturesZ, axis = 1)
        
    key_gesture = gestures[key_idx]
    
    cost=[0]*len(gestures)
    for i, gesture in enumerate(gestures) :
        for j, sensor in enumerate(gesture) :
            n = len(sensor)
            m = len(key_gesture[j])
            cost[i] += edit_distance(key_gesture[j], sensor, m, n)
    cost_top_K = sorted(cost)[0:top_K]                                        

elif user_option == 7 :
    pathW = directory + "/W"
    pathX = directory + "/X"
    pathY = directory + "/Y"
    pathZ = directory + "/Z"
    gesturesW = amplitude_loader(pathW)
    gesturesX = amplitude_loader(pathX)
    gesturesY = amplitude_loader(pathY)
    gesturesZ = amplitude_loader(pathZ)
    
    gestures = np.append(gesturesW, gesturesX, axis = 1)
    gestures = np.append(gestures, gesturesY, axis = 1)
    gestures = np.append(gestures, gesturesZ, axis = 1)
    
    key_gesture = gestures[key_idx]
    
    cost=[0]*len(gestures)
    for i, gesture in enumerate(gestures) :                                     
        for j, sensor in enumerate(gesture) :                                   
            n = len(sensor)                                                     
            m = len(key_gesture[j])                                             
            cost[i] += dynamic_time_warping(key_gesture[j], sensor, m, n)    
    cost_top_K = sorted(cost)[0:top_K]                                        
else:
    print("ERROR : No such user option in this program")
    
       
file = open("similarGesturesTask6.txt", "w")
outputList = []

print("Most similar (gesture, score) ")    
for k in cost_top_K :    
    c = 0
    poop = cost.index(k)
    for fileK in glob.glob("Dataset 5/" + "X/"+ "*.csv"):
        if c == poop:
            outputList.append((fileK.replace("Dataset 5/X\\", ""), k))
            print((fileK.replace("Dataset 5/X\\", ""), k))
            break
        c = c + 1 
file.write(str(outputList))
    
file.close()

#end of phase2 Task2 code

# task 6
# requests the irrelevent results and produces a new query output based on those results.
# calls again for more feedback if requested
def feedbackInterface(taskNum):
    irrelevent = input("Enter the irrelevent results in a list (ex: 11 3 4): ")
    relevent = input("Enter the relevent results in a list (ex: 26 52): ")
    
    #convert strings to lists
    irrlist = list(irrelevent.split(" "))
    irreleventFile = open("irrelevent.txt", "a")
    irreleventFile.write(str(irrlist))
    irreleventFile.close()
    rlist = list(relevent.split(" "))
    releventFile = open("relevent.txt", "w")
    releventFile.write(str(rlist))
    releventFile.close()
    
    if taskNum == "probabilistic":
        #call feedback function for task 4
        #parameters: list of irrelevent results, list of relevent results, original guesture file
        
        feedback = input("Would you like to provide feedback to improve your query output? (enter Y/N): ")
        if feedback == "Y":
            feedbackInterface(taskNum)
        elif feedback == "N":
            sys.exit()
        
    elif taskNum == "classifier_based":
        #call feedback function for task 5
        #parameters: list of irrelevent results, list of relevent results, original guesture file
        
        feedback = input("Would you like to provide feedback to improve your query output? (enter Y/N): ")
        if feedback == "Y":
            feedbackInterface(taskNum)
        elif feedback == "N":
            sys.exit()
            
# starting interface
run = True
releventFile = open("relevent.txt", "w")
releventFile.write("")
irreleventFile = open("irrelevent.txt", "w")
irreleventFile.write("")
releventFile.close()
irreleventFile.close()

while run == True:

    taskNum = input("Please enter the relevence technique you would like (enter probabilistic or classifier_based): ")
    resultNum = input("Please enter the number of results you would like to recieve: (ex: 5): ")
    
    #call task 4 or 5 here:
    if taskNum == "probabilistic":
        #call task 4
        #parameters: guesture file to find comparisons, number of results
        
        feedback = input("Would you like to provide feedback to improve your query output? (enter Y/N): ")
        if feedback == "Y":
            feedbackInterface(taskNum)
        
    elif taskNum == "classifier_based":
        #call task 5
        #parameters: guesture file to find comparisons, number of results
        
        feedback = input("Would you like to provide feedback to improve your query output? (enter Y/N): ")
        if feedback == "Y":
            feedbackInterface(taskNum)
  
        
#Task 1
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
import os
import re

numbers = re.compile(r'(\d+)')

#unnecessary lines if you are using anaconda or another 
print("Please enter the following inputs as the same values you used for task 0: ")
directory = input("Enter the data directory path (ex: Data/: ")
w = input("Enter the window length (ex: 3): ")
s = input("Enter the shift length (ex: 3): ")
r = input("Enter the resolution (ex: 3): ")

w = int(w)
s = int(s)
r = int(r)
#end of non-anaconda lines

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#returns the top-k topics 
def PCAsetup(wordMat, k):
    #calculate PCA
    pca = PCA(k)
    pc = pca.fit_transform(wordMat)
    UT = pca.components_
    topK = pd.DataFrame(data = UT)
    
    original_df = pd.DataFrame(data = pc)
    original_df.to_pickle("./PCA_" + vectModel + ".pkl")
    
    print(topK)
    return topK

def SVDsetup(wordMat, k):
    #calculate SVD
    svd = TruncatedSVD(k)
    sv = svd.fit_transform(wordMat)
    VT = svd.components_
    topK = pd.DataFrame(data = VT)
    
    original_df = pd.DataFrame(data = sv)
    original_df.to_pickle("./SVD_" + vectModel + ".pkl")
    
    print(topK)
    return topK

def NMFsetup(wordMat, k):
    #calculate NMF
    nmf = NMF(k)
    nm = nmf.fit_transform(wordMat)
    R = nmf.components_
    topK = pd.DataFrame(data = R)
    
    original_df = pd.DataFrame(data = nm)
    original_df.to_pickle("./NMF_" + vectModel + ".pkl")
    
    print(topK)
    return topK

def LDAsetup(wordMat, k):
    #calculate LDA
    lda = LDA(k)
    ld = lda.fit_transform(wordMat)
    V = lda.components_
    topK = pd.DataFrame(data = V)
    
    original_df = pd.DataFrame(data = ld)
    original_df.to_pickle("./LDA_" + vectModel + ".pkl")
    
    print(topK)
    return topK


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
                
            for file in sorted(glob.glob(directory + axis + "/tf_vectors_*.txt"), key=numericalSort):
                #if not file.endswith(".txt") or not file.startswith("tf_vectors_"):
                    #continue
                #Xmat = []
                #read tf file
                f = open(file, "r")
                tf_vectors = f.readlines()
        
                gestWords = []
                tfVals = []
        
                #split the line into the word and tf value
                for line in tf_vectors:
                    noDash = line.split("-")
                    tf_val = noDash[-1]
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
                    
                    #temp
                    if "e" in tfVals[index]:
                        #continue
                        tfVals[index].replace("e", "")
                        tfVals[index].replace("-", "")
                        tfVals[index] = str(float(tfVals[index]) * 0.00001)
                        
                    #temp
            
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
                
            for file in sorted(glob.glob(directory + axis + "/tfidf_vectors_*.txt"), key=numericalSort):
                #if not file.endswith(".txt") or not file.startswith("tfidf_vectors_"):
                    #continue
                #read tf file
                f = open(file, "r")
                tf_vectors = f.readlines()
        
                gestWords = []
                tfVals = []
        
                #split the line into the word and tf value
                for line in tf_vectors:
                    noDash = line.split("-")
                    tf_val = noDash[-1]
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
            
                    #temp
                    if "e" in tfVals[index]:
                        tfVals[index].replace("e", "")
                        tfVals[index].replace("-", "")
                        tfVals[index] = str(float(tfVals[index]) * 0.00001)
                    #temp
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
    
def createdictofComponents(topk, k):
    #creates a matrix that contains tuples of the word and score
    outputMat = []
    outputMat = topk
    startI = "1"
    for y in range(1, w):
        startI = startI + "1"
    startI = int(startI)
    numWords = (startI * (2*r) - startI)
    
    for j in range(0, len(topk.columns)):
        axisSplit = len(topk) / 4
        sensorSplit = axisSplit / 20
        
        ax = int(j/axisSplit)
        sens = int((j - (ax * axisSplit))/sensorSplit)
        word = int((j - ((sens * sensorSplit) + (ax * axisSplit))) - startI)
        
        for row in range(0, k):
            if ax == 0:
                axis = 'W'
            elif ax == 1:
                axis = 'X'
            elif ax == 2:
                axis = 'Y'
            elif ax == 3:
                axis = 'Z'
            label = axis + str(sens) + str(word)
            outputMat[j][row] = (topk[j][row], label)
            
        
                
            """for sensor in range(0, 20):
            
                for word in range(0, numWords + 1):
                    label = axis + str(sensor - 1) + str(word + startI)
                    
                    #axisSplit = len(topk) / 4
                    sensorSplit = axisSplit / 20
                    
                    wordIndex = int((word + startI) + (sensor * sensorSplit) + (ax * axisSplit))
                    wordIndex = wordIndex - startI
                    outputMat[wordIndex][row] = (topk[wordIndex][row], label)"""
                
                
    #for i in range(0, k):
     #   word = "w" + str(i)
      #  for j in range(0, len(topk)):
       #     outputMat[i][j] = (topk[i][j], word)
            
    print(outputMat)
    #sorts the words acording to their scores
    outputMat = np.transpose(outputMat)
    outputMat = outputMat.apply(lambda x: x.sort_values(ascending = False).values)
    outputMat = np.transpose(outputMat)
    finalMat = outputMat
    #for i in range(0, k):
     #   for j in range(0, len(outputMat)):
      #      finalMat[i][j] = (outputMat[i][j][1], outputMat[i][j][0])
            
    file = open("./userOutput.txt", "w")
    file.write(str(finalMat))
    file.close()
    return finalMat
    
    
def task1(vectModel, useOp, k):
    
    if useOp == "PCA":
        #PCA
        wordMat = makeMat(vectModel)
        
        topk = PCAsetup(wordMat, k)
        print("\nPCA:\n")
        #print(topk)
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)
        
        
    elif useOp == "SVD":
        wordMat = makeMat(vectModel)
        topk = SVDsetup(wordMat, k)
        print("\nSVD:\n")
        #print(topk)
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)
        
        
    elif useOp == "NMF":
        wordMat = makeMat(vectModel)
        topk = NMFsetup(wordMat, k)
        print("\nNMF:\n")
        #print(topk)
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)

        
    elif useOp == "LDA":
        wordMat = makeMat(vectModel)
        topk = LDAsetup(wordMat, k)
        print("\nLDA:\n")
        #print(topk)
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)
        


vectModel = input("Enter the vector model (ex: tf): ")
k = input("Enter k (ex: 4): ")
useOp = input("Enter the analysis you would like to use (ex: PCA): ")
k = int(k)
task1(vectModel, useOp, k)


#sample output: 
#Enter the vector model: tf
#Enter k: 10
#Enter the analysis you would like to use: PCA



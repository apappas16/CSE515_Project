#Task 1
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA

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

#returns the top-k topics 
def PCAsetup(wordMat, k):
    #calculate PCA
    pca = PCA(k)
    pc = pca.fit_transform(wordMat)
    topK = pd.DataFrame(data = pc)
    
    return topK

def SVDsetup(wordMat, k):
    #calculate SVD
    svd =  TruncatedSVD(k)
    sv = svd.fit_transform(wordMat)
    topK = pd.DataFrame(data = sv)
    return topK

def NMFsetup(wordMat, k):
    #calculate NMF
    nmf = NMF(k)
    nm = nmf.fit_transform(wordMat)
    topK = pd.DataFrame(data = nm)
    return topK

def LDAsetup(wordMat, k):
    #calculate LDA
    lda = LDA(k)
    ld = lda.fit_transform(wordMat)
    topK = pd.DataFrame(data = ld)
    
    return topK


def makeMat(vectModel, axis):
        
    #read files
    if vectModel == "tf":
        Xmat = []
        for file in glob.glob(directory + axis + "/tf_vectors_*.txt"):
            #read tf file
            f = open(file, "r")
            tf_vectors = f.readlines()
        
            gestWords = []
            tfVals = []
        
            #split the line into the word and tf value
            for line in tf_vectors:
                noDash = line.split("-")
                tf_val = noDash[1]
                word = noDash[0].split(",")
                word[2] = word[2].replace(")","")
                word[2] = word[2].replace("'", "")
                word[2] = word[2].replace(" ", "")
                tf_val = tf_val.replace("\n", "")
                gestWords.append(word[2])
                tfVals.append(tf_val)
           
        
            index = 0
            startI = "1"
            for y in range(1, w):
                startI = startI + "1"
            startI = int(startI)
        
            #create dictionary with every word
            numWords = startI * (2*r) - startI
            wordMat = []
    
            for i in range(0, numWords + 1):
                wordMat.append(0)
            
            # put tf values into matrix where column = word
            for x in gestWords:
                xint = int(x)
            
                #if wordMat[xint - startI] == 0:
                wordMat[xint - startI] = wordMat[xint - startI] + float(tfVals[index])
                index = index + 1
                
            
            for iterate in range(0, len(wordMat)):
                wordMat[iterate] = wordMat[iterate] / 20

            
            Xmat.append(wordMat)
            f.close()
           
        return Xmat
    
    elif vectModel == "tfidf":
        Xmat = []
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
                word = noDash[0].split(",")
                word[2] = word[2].replace(")","")
                word[2] = word[2].replace("'", "")
                word[2] = word[2].replace(" ", "")
                tf_val = tf_val.replace("\n", "")
                gestWords.append(word[2])
                tfVals.append(tf_val)
           
        
            index = 0
            startI = "1"
            for y in range(1, w):
                startI = startI + "1"
            startI = int(startI)
        
            #create dictionary with every word
            numWords = startI * (2*r) - startI
            wordMat = []
    
            for i in range(0, numWords + 1):
                wordMat.append(0)
            
            # put tf values into matrix where column = word
            for x in gestWords:
                xint = int(x)
            
                #if wordMat[xint - startI] == 0:
                wordMat[xint - startI] = wordMat[xint - startI] + float(tfVals[index])
                index = index + 1
                
            
            for iterate in range(0, len(wordMat)):
                wordMat[iterate] = wordMat[iterate] / 20

            
            Xmat.append(wordMat)
            f.close()
           
        return Xmat
    
def createdictofComponents(topk, k):
    #creates a matrix that contains tuples of the word and score
    outputMat = []
    outputMat = topk
    for i in range(0, k):
        word = "w" + str(i)
        for j in range(0, len(topk)):
            #print(topk[i][j])
            outputMat[i][j] = (topk[i][j], word)
            
    
    #sorts the words acording to their scores
    outputMat = np.transpose(outputMat)
    outputMat = outputMat.apply(lambda x: x.sort_values(ascending = False).values)
    outputMat = np.transpose(outputMat)
    finalMat = outputMat
    for i in range(0, k):
        for j in range(0, len(outputMat)):
            finalMat[i][j] = (outputMat[i][j][1], outputMat[i][j][0])
            
    file = open("./userOutput.txt", "w")
    file.write(str(finalMat))
    file.close()
    return finalMat
    
    
def task1(gestfiles, vectModel, useOp, k):
    
    if useOp == "PCA":
        #PCA for X axis
        wordMat = makeMat(vectModel, gestfiles)
        topk = PCAsetup(wordMat, k)
        print("\nPCA for " + gestfiles + " gesture files:\n")
        #print(topk)
        
        original_df = pd.DataFrame(topk)
        original_df.to_pickle("./PCA_" + gestfiles + "_" + vectModel + ".pkl")
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)
        
        
    elif useOp == "SVD":
        wordMat = makeMat(vectModel, gestfiles)
        topk = SVDsetup(wordMat, k)
        print("\nSVD for " + gestfiles + " gesture files:\n")
        #print(topk)
        
        original_df = pd.DataFrame(topk)
        original_df.to_pickle("./SVD_" + gestfiles + "_" + vectModel + ".pkl")
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)
        
        
    elif useOp == "NMF":
        wordMat = makeMat(vectModel, gestfiles)
        topk = NMFsetup(wordMat, k)
        print("\nNMF for " + gestfiles + " gesture files:\n")
        #print(topk)
        
        original_df = pd.DataFrame(topk)
        original_df.to_pickle("./NMF_" + gestfiles + "_" + vectModel + ".pkl")
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)

        
    elif useOp == "LDA":
        wordMat = makeMat(vectModel, gestfiles)
        topk = LDAsetup(wordMat, k)
        print("\nLDA for " + gestfiles + " gesture files:\n")
        #print(topk)
        
        original_df = pd.DataFrame(topk)
        original_df.to_pickle("./LDA_" + gestfiles + "_" + vectModel + ".pkl")
        
        dictofComponents = createdictofComponents(topk, k)
        print(dictofComponents)
        


gestfiles = input("Enter the folder that you want analyzed (ex: X): ")
vectModel = input("Enter the vector model (ex: tf): ")
k = input("Enter k (ex: 4): ")
useOp = input("Enter the analysis you would like to use (ex: PCA): ")
k = int(k)
task1(gestfiles, vectModel, useOp, k)

#p = pickle.load( open( "LDA_W_tf.pkl", "rb" ) )
#print("pickle")
#print(p)

#sample output: 
#Enter the folder that you want analyzed: X
#Enter the vector model: tf
#Enter k: 10
#Enter the analysis you would like to use: PCA


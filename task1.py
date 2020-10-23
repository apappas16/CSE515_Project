#Task 1
import glob
import pandas as pd
from sklearn.decomposition import PCA

#returns the top-k topics 
def PCAsetup(wordMat, k):
    #calculate PCA
    pca = PCA(k)
    pc = pca.fit_transform(wordMat)
    topK = pd.DataFrame(data = pc)
    
    return topK

def SVD(k):
    return topK

def NMF(k):
    return topK

def LDA(k):
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
    
    elif vectModel == "idf":
        return wordMat
    
def task1(gestfiles, vectModel, useOp, k):
    
    if useOp == "PCA":
        #PCA for X axis
        wordMat = makeMat(vectModel, gestfiles)
        topk = PCAsetup(wordMat, k)
        print("\nPCA for " + gestfiles + " gesture files:\n")
        print(topk)
        
    elif useOp == "SVD":
        topk = SVD(k)
    elif useOp == "NMF":
        topk = NMF(k)
    elif useOp == "LDA":
        topk = LDA(k)


gestfiles = input("Enter the folder that you want analyzed: ")
vectModel = input("Enter the vector model: ")
k = input("Enter k: ")
useOp = input("Enter the analysis you would like to use: ")
k = int(k)
task1(gestfiles, vectModel, useOp, k)

#sample output: 
#Enter the folder that you want analyzed: X
#Enter the vector model: tf
#Enter k: 10
#Enter the analysis you would like to use: PCA


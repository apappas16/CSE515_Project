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


def makeMat(gestfiles, axis):
        
    #read files
    if gestfiles == "tf":
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
            
                if wordMat[xint - startI] == 0:
                    wordMat[xint - startI] = float(tfVals[index])
            
                index = index + 1
                
            Xmat.append(wordMat)
            f.close()
           
        return Xmat
    
    elif gestfiles == "idf":
        return wordMat
    
def task1(gestfiles, vectModel, k):
    
    if vectModel == "PCA":
        #PCA for X axis
        wordMat = makeMat(gestfiles, "X")
        topk = PCAsetup(wordMat, k)
        print("\nPCA for X gesture files:\n")
        print(topk)
        
        #PCA for Y axis
        wordMat = makeMat(gestfiles, "Y")
        topk = PCAsetup(wordMat, k)
        print("\nPCA for Y gesture files:\n")
        print(topk)
        
        #PCA for Z axis
        wordMat = makeMat(gestfiles, "Z")
        topk = PCAsetup(wordMat, k)
        print("\nPCA for Z gesture files:\n")
        print(topk)
        
        #PCA for W axis
        wordMat = makeMat(gestfiles, "W")
        topk = PCAsetup(wordMat, k)
        print("\nPCA for W gesture files:\n")
        print(topk)
        
    elif vectModel == "SVD":
        topk = SVD(k)
    elif vectModel == "NMF":
        topk = NMF(k)
    elif vectModel == "LDA":
        topk = LDA(k)



gestfiles = input("Enter the gesture files: ")
vectModel = input("Enter the vector model: ")
k = input("Enter k: ")
k = int(k)
task1(gestfiles, vectModel, k)


# CSE 515 Phase 2:

## main.py

## task1.py
This file takes inputs from main.py and prints the words and scores of values decomposed from the tf and tfidf values from main.py

## task2.py
First of all, you will need a path to the specific gesture file like “Data/X/1.csv”. Secondly, it will ask you to choose a vector model between tf or tf-idf. Third, you will need to choose  user options from 1 to 7 which follows the order of user options in project description. If you don’t input an integer for the user option, it will just end up execution. This program can be run by using “python task2.py” in the project directory.

## task3.py
The first part of the implementation was to reference the functions built in task 2 for the user options. For user option one, a similarity function was made using euclidean distance and turned into a similarity matrix using a cost function. For user options 2-5, PCA, SVD, NMF, and LDA were implemented using built-in functions within the Python library. Option #6 was constructed using the same cost function from task 2. SVD and NMF were then performed on the similarity matrix constructed previously using the built-in SVD function in the Python library. The top-p latent semantics were then retrieved from the matrix using a sorting function on the matrix. The gesture and score were retrieved from the matrix and put into a 2d list of values of the form [gesture, score]. 

## task4.py

The file has main() which runs the processes for Task 4 and prints the partition output for Task 4a and 4b in the terminal and plots the clusters for Task 4c and 4d.

# Phase 3:

## phase3_task1.py
This file takes the similarities of multiple files and visualizes the most dominate gestures

## phase3_task2.py
Task2: This file takes the similarity matrix from phase 2 and computes KNN to predict the labels of the gestures.
Task5: This file also has task5 which takes user input to create a PPR.

## phase3_task4.py
This file implements probabilistic relevance to improve the KNN classification

## task6.py
This file is the interface for task 4 and 5. It formats the user inputs for the other tasks so that the user can provide feedback to the PPR.
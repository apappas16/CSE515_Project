import csv
import math
import os
from scipy.integrate import quad


def calcAvgSensorValue(sensorValues):
    avg = sum(sensorValues) / len(sensorValues)
    avg = round(avg, 9)
    return avg


def calcStdDev(sensorValues, meanVal):
    sd = []
    for num in sensorValues:
        result = num - meanVal
        result = result * result
        sd.append(result)
    return calcAvgSensorValue(sd)


def normalize(sensorValues):
    normalized_sensor = []
    for val in sensorValues:
        val = float(val)
        normalized = 2 * ((val - min(sensorValues)) / (max(sensorValues) - min(sensorValues))) - 1
        normalized_sensor.append(normalized)
    return normalized_sensor


def integral(i):
    return getGaussianVal(i, 0, 0.25)


def getGaussianVal(i, avg, sd):
    i = float(i - avg) / sd
    gauss = math.exp(-i * i / 2.0) / math.sqrt(2.0 * math.pi) / sd
    return gauss


def determineBands():
    numBands = r * 2
    bandList = []
    bandStart = -1
    for i in range(1, numBands):
        integral1, e = quad(integral, (i - r - 1) / r, (i - r) / r)
        integral2, e = quad(integral, -1, 1)
        length_i = 2 * (integral1 / integral2)
        band = bandStart + length_i
        bandList.append(band)
        bandStart = band
    bandList.append(1.0)
    return bandList


def quantize(values, bandList):
    quantized = ""
    for i in range(len(values)):
        bound = -1
        for band in bandList:
            if band >= values[i] > bound:
                quantized += str(bandList.index(band) + 1)
                break
            else:
                bound = band
    return quantized


def getWords():
    wordList = []
    i = 0
    while (i + w - 1) < len(quantizedSensor):
        word = quantizedSensor[i:i + w]
        wordList.append(word)
        i += s
    return wordList


def addToUniqueDict(word_tuple):
    inList = False
    for word in unique_dict:
        if word == word_tuple:
            inList = True
            break
    if not inList:
        unique_dict.append(word_tuple)


def calcAvgQuanAmp():
    avgQuanAmpList = []
    normWord = []
    i = 0
    while (i + w - 1) < len(normSensorVals):
        word = normSensorVals[i:i + w]
        normWord.append(word)
        i += s

    for word in normWord:
        avgAmp = sum(word) / len(word)
        avgQuanAmpList.append(avgAmp)

    return avgQuanAmpList


def getWordsFromFile(file):
    with open(file, "r") as f:
        words = []
        for line in f:
            line = line.strip()
            row = line.split(" - [")  # list of all words as they occur in the file
            if len(row) > 1:
                words.append(row[1])
        f.close()
    return formatWordsFromFile(words)


def formatWordsFromFile(tempList):
    allWords = []
    for sensorWords in tempList:
        sensorWords = sensorWords.replace("]", "")
        sensorWords = sensorWords.replace("'", "")
        wrd = sensorWords.split(", ")
        allWords.append(wrd)
    return allWords


def getUniqueWordsInGesture(allWordsInGesture):
    uniqueWords = []
    for i in range(len(allWordsInGesture)):  # i is each sensor in the gesture
        for word in allWordsInGesture[i]:
            uniqueWord = (direct, i+1, word)
            if uniqueWord not in uniqueWords:
                uniqueWords.append(uniqueWord)
    return uniqueWords


def calcTfValue(wordTuple, allWordsInFile):
    totalWords = len(allWordsInFile[0]) * 20
    num_occurs = 0
    for sensor in range(len(allWordsInFile)):
        for wrd in allWordsInFile[sensor]:
            if wrd == wordTuple[2] and (sensor+1) == wordTuple[1]:
                num_occurs += 1
    value = num_occurs / totalWords
    return value


def calcIdfValue(wordTuple, direct):
    numObjs = 60
    numObjsWithWord = 1
    sensorId = wordTuple[1]

    for file in os.listdir(directory + direct):
        if file != gestureFile and file.endswith(".wrd"):
            file = directory + direct + "/" + file
            gestFileWords = getWordsFromFile(file)
            if word_tuple[2] in gestFileWords[sensorId-1]:
                numObjsWithWord += 1
                break

    value = math.log(numObjs / numObjsWithWord)
    return value


if __name__ == '__main__':
    # GLOBAL VARIABLES:
    unique_dict = []  # stores list of all unique words found
    gesture_dict = []  # stores list of all words found across all files

    # TASK 0
    # TASK 0A
    directory = input("Enter the data directory path: ")
    w = input("Enter the window length: ")
    s = input("Enter the shift length: ")
    r = input("Enter the resolution: ")

    w = int(w)
    s = int(s)
    r = int(r)

    # for each data file create a .wrd file containing the following:
    for direct in os.listdir(directory):
        # for each csv file in X,Y,W,Z:
        for filename in os.listdir(directory + direct):
            if filename.endswith(".csv"):
                bands = determineBands()

                # generate .wrd file
                wrdFile = open(str(directory) + str(direct) + "/" + str(filename) + ".wrd", "w")

                sensor_id = 1
                csvFile = open(str(directory) + str(direct) + "/" + filename, "r")
                reader = csv.reader(csvFile, delimiter=',')
                # for each sensor sj in file
                for sensor in reader:
                    # output component ID, c in output file
                    wrdFile.write(str(direct) + ", ")

                    # write sensorID to wrd file
                    wrdFile.write(str(sensor_id) + ", ")

                    # compute and output average amplitude, avgij of the values
                    sensorVals = list(sensor)
                    sensorVals = [float(i) for i in sensorVals]
                    sensorAvg = calcAvgSensorValue(sensorVals)
                    wrdFile.write(str(sensorAvg) + ", ")

                    # compute and output standard deviations stdij of the values
                    stdDev = calcStdDev(sensorVals, sensorAvg)
                    wrdFile.write(str(stdDev) + ", ")

                    # normalize entries between -1 and 1
                    normSensorVals = normalize(sensorVals)

                    # quantizes entries into 2r levels as in phase 1
                    quantizedSensor = quantize(normSensorVals, bands)

                    # moves a w-length window on time series (by shifting it s units at a time), and at position h
                    sensorWords = getWords()

                    # computes and outputs in file average quantized amplitude avgQijh for window h of sensor sj
                    avgQuanAmp = calcAvgQuanAmp()
                    wrdFile.write(str(avgQuanAmp) + ", " + " - ")

                    # computes and outputs symbolic quantized window descriptor winQijh for the window h of sensor sj
                    wrdFile.write(str(sensorWords) + "\n")

                    # add dictionary of each window to gestureDict list
                    for window in sensorWords:
                        wordDict = (direct, sensor_id, window)
                        gesture_dict.append(wordDict)
                        addToUniqueDict(wordDict)

                    sensor_id += 1
        # The dictionary of the words consists of <componentName, sensorID, winQ>

    # TASK 0B
    for direct in os.listdir(directory):
        # for each gesture file in W,X,Y,Z:
        # create vector .txt files with tf and tf-idf values
        for filename in os.listdir(directory + direct):
            if filename.endswith(".wrd"):
                tfFile = open(directory + direct + "/tf_vectors_" + filename[:-8] + ".txt", "w")
                tfidfFile = open(directory + direct + "/tfidf_vectors_" + filename[:-8] + ".txt", "w")
                gestureFile = directory + direct + "/" + filename

                allWordsInGesture = getWordsFromFile(gestureFile)
                uniqueWordsInGesture = getUniqueWordsInGesture(allWordsInGesture)

                for word_tuple in uniqueWordsInGesture:
                    tfValue = calcTfValue(word_tuple, allWordsInGesture)
                    idfValue = calcIdfValue(word_tuple, direct)
                    tf_idf_value = tfValue * idfValue
                    tfFile.write(str(word_tuple) + " - " + str(tfValue) + "\n")
                    tfidfFile.write(str(word_tuple) + " - " + str(tf_idf_value) + "\n")
    # End of TASK1


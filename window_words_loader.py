import os
import re
directory = "./Data/Z/"
gestures = []
for filename in os.listdir(directory):
    if not filename.endswith(".wrd") :
        continue
    path = [directory, filename]
    path = "/".join(path)
    with open(path, "r") as w:
        gesture = w.readlines()

    matrix=[]

    for sensor in gesture :
        sensor_id = int(sensor.split(",")[1])
        #finds all quantized word vectors by regex. It detect integer numbers in string
        word = [re.findall(r'\d+',word)[0] for word in sensor.split(" - ")[1].split(",")]
        matrix.append(word)
    #gestures are set of gesture which has quantized word vectors.
    gestures.append(matrix)

#Print 60 in case of Z folder
print(len(gestures))

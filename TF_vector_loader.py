import os                                                                       
directory = "./Data/Z/"                                                         
for filename in os.listdir(directory):                                          
    if not filename.endswith(".txt") :                                          
        continue                                                                
    path = [directory, filename]                                                
    path = "/".join(path)                                                       
    with open(path, "r") as w:                                                  
        gesture = w.readlines()                                                 
                                                                                
    #Creating dim*dim matrix                                                    
    matrix=[]                                                                   
    dim = int(gesture[-1].split("-")[0].split(",")[1])                          
    for num in range(dim) :                                                     
        matrix.append([])                                                       
                                                                                
    #Extracting TF values from sensor and add it to matrix                      
    for sensor in gesture :                                                     
        print(sensor)                                                           
        sensor_id = int(sensor.split("-")[0].split(",")[1])                     
        word_score = float(sensor.split("-")[1].strip())                        
        matrix[sensor_id-1].append(word_score)

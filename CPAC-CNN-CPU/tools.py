"""
tools used for data loading
"""

import numpy as np
import pandas as pd
import cv2
import os

def load_data(train_csv_root,data_root):
    dataset = np.array(pd.read_csv(train_csv_root))
    name_label = {}
    N,_ = np.shape(dataset)
    for i in range(N):
        name_label[dataset[i,0]] = []
    for i in range(N):
        name_label[dataset[i,0]].append(dataset[i,1])
        
    data_filtered = {}

    for key in name_label.keys():
        if len(name_label[key]) == 1:
            data_filtered[key] = name_label[key][0]
            
    X = []
    Y = []
    all_files = os.listdir(data_root)[:1500]
    
    defect = 0
    nodefect = 0
    for file in all_files:
        image_name = data_root + file
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:,:]
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if file in data_filtered.keys():
            
            X.append(img)
            Y.append(int(1))
            defect += 1
        else:
            
            X.append(img)
            Y.append(int(0))
            nodefect += 1
#    print(defect, nodefect)
    print(np.array(X).shape, np.array(Y).shape)
            
            
    return np.array(X), np.array(Y)


def load_data_steel(data_root, size):
    folders = ["CR", "IN", "PA", "PS", "RS", "SC"]
    X = []
    Y = []
    
    
    for i in range(len(folders)):
        image_root = data_root + folders[i] + '/'
        all_files = os.listdir(image_root)
        for file in all_files:
            image_name = image_root + file
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)[50:150,50:150]
            img_new = cv2.resize(img, (int(size),int(size)), interpolation = cv2.INTER_NEAREST)
            X.append(img_new)
            Y.append(i)
    
    
    print(np.array(X).shape, np.array(Y).shape)
            
            
    return np.array(X), np.array(Y)
    
    
    
    
def load_data_Meg_Binary(data_root):           
    
    folders = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free"]
    X = []
    Y = []
    
    
    for i in range(len(folders)):
        image_root = data_root + folders[i] + '/' + 'Imgs' + '/'
        all_files = os.listdir(image_root)[:400]
        for file in all_files:
            image_name = image_root + file
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            img_new = cv2.resize(img, (25,25), interpolation = cv2.INTER_NEAREST)
            X.append(img_new)
            if i == 5:
                Y.append(1)
            else:
                Y.append(0)
            
    
    
    print(np.array(X).shape, np.array(Y).shape)
            
            
    return np.array(X), np.array(Y)
    
    
    
    
    
def load_data_Meg_Multi(data_root, size):           
    
    folders = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free"]
    X = []
    Y = []
    
    
    for i in range(len(folders)):
        image_root = data_root + folders[i] + '/' + 'Imgs' + '/'
        all_files = os.listdir(image_root)[:150]
        
        for file in all_files:
            image_name = image_root + file
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            img_new = cv2.resize(img, (int(size),int(size)), interpolation = cv2.INTER_NEAREST)
            X.append(img_new)
            Y.append(i)
            if i == 2:
                X.append(np.fliplr(img_new))
                Y.append(i)
            elif i == 3:
                X.append(np.fliplr(img_new))
                Y.append(i)
                X.append(np.rot90(img_new))
                Y.append(i)
                
            
            
    
    
    print(np.array(X).shape, np.array(Y).shape)
            
            
    return np.array(X), np.array(Y)
      
    
    
    
    
    
    
    
    
    
    
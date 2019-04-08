import csv
import pandas as pd
import numpy as np
import math

def split_dataset(dataset,train,test,rows,train_rows,test_rows,cols):
    for i in range(0,train_rows):
        for j in range(0,cols):
            train[i][j]=dataset[i][j]
    c=0
    for i in range(train_rows,rows):
        for j in range(0,cols):
            test[c][j]=dataset[i][j]
        c=c+1
def euclidian_distance(train,test,x):
    rows_train=len(train)
    distance=[0 for i in range(rows_train)]
    cols=len(train[0])
    for i in range(rows_train):
        for j in range(cols-1):
            distance[i]+=pow((test[x][j]-train[i][j]),2)
        distance[i]=math.sqrt(distance[i])
    return(distance)

def k_nearest(k,train,test,x):
    neighbours_index=[]
    distance=euclidian_distance(train,test,x)
    neighbours_index=sorted(range(len(distance)),key=distance.__getitem__)
    k_nearest=neighbours_index[:k]
    return(k_nearest)

def knn_prediction(k,train,test,cols,x):
    nearest=k_nearest(k,train,test,x)
    train_output=[train[nearest[i]][cols-1] for i in range(len(nearest))]
    prediction=max(train_output,key=train_output.count)
    return(prediction)

def accuracy(knn_predictions,test):
    correct_predictions=0
    rows=len(test)
    cols=len(test[0])
    for i in range(rows):
        if test[i][cols-1]==knn_predictions[i]:
            correct_predictions=correct_predictions+1
    accuracy=correct_predictions/rows *100
    return(accuracy)

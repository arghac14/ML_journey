import KNN_algorithm as KNN
import pandas as pd
df=pd.read_csv("IRIS.DAT",sep="\t",header=None)
dataset=df.values
rows=len(dataset)
cols=len(dataset[0])
split=0.7
train_rows=int(split*len(dataset))
test_rows=rows-train_rows
train=[[0 for col in range(cols)]for row in range(train_rows)]
test=[[0 for col in range(cols)]for row in range(test_rows)]
KNN.split_dataset(dataset,train,test,rows,train_rows,test_rows,cols)
knn_predictions=[0 for i in range(test_rows)]
for i in range(test_rows):
    knn_predictions[i]=KNN.knn_prediction(23,train,test,cols,i)
accuracy_percentage=KNN.accuracy(knn_predictions,test)
print("ACCURACY OF THE KNN MODEL ON THE DATASET: ",accuracy_percentage,"%")


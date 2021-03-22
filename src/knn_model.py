#KNN model construction for milestone 2 
#Bianca Blackwell 301304375
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

#Function for Hot-Encoding a DF, returns the DF with no categorical data
def Hot_Encode(data):
	temp = pd.get_dummies(data.sex, prefix='sex')
	data = data.join(temp)
	data = data.drop(columns = "sex")
	data = data.drop(columns = "country")
	data = data.drop(columns = "province")
	temp = pd.get_dummies(data.age_range, prefix = 'age_range')
	data = data.join(temp)
	data = data.drop(columns = "age_range")

	#The returned data is entirely hot-encoded
	return data

#load in the data
train = pd.read_csv(r'../results/cases_train_preprocessed.csv',dtype=object)
#test =  pd.read_csv(r'../results/cases_test_preprocessed.csv',dtype=object)

#Dropping land column, age_range_ind, and Last_Updated, and source
train = train.drop(columns = "land")
train = train.drop(columns = "Last_Update")
train = train.drop(columns = "age_range_ind")
train =train.drop(columns = "source")

#Hot-encoding (both to test and train)
train_copy = train.copy()
train = Hot_Encode(train)

#Handling additional_information (0 for no data, 1 for any data)
train.loc[train['additional_information'] == '[]', 'add_info'] = 0
train.loc[train['additional_information'] != '[]', 'add_info'] = 1
train = train.drop(columns = "additional_information")

#Split training data into data and labels
x = train.drop(columns = 'outcome')
y = train.outcome

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state = 1)

#Feature Scaling
scaler = StandardScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_te = scaler.transform(x_te)

#Training the model
k_neigh = 5
classifier = KNeighborsClassifier(n_neighbors = k_neigh)
classifier.fit(x_tr, y_tr)

knnPickle = open('../models/knn_classifier.pkl','wb')
pickle.dump(classifier, knnPickle)

#Save my validation data x_te, y_te data to open in new file
y_te.to_csv('y_te.csv', index=False)
pd.DataFrame(x_te).to_csv('x_te.csv', index =False)
#saving my training data since apperently i have to 'load my model' later on...
y_tr.to_csv('y_tr.csv', index= False)
pd.DataFrame(x_tr).to_csv('x_tr.csv', index=False)

print("Finished Constructing KNN model with k neightbours = " + str(k_neigh) + ".")

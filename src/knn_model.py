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
def Hot_Encode(data,data2):
	temp = pd.get_dummies(data.sex, prefix='sex')
	data = data.join(temp)
	data = data.drop(columns = "sex")
	#Notice we need our country and province encoding to include every possible value our model should handle
	#I.e, both training and test data in our case
	temp = data['country']
	temp2 = data2['country']

	temp = pd.concat([temp,temp2],ignore_index = True)
	temp = pd.get_dummies(temp, prefix = 'country')
	#only take the entries that correspond to our data at hand
	temp = temp.head(len(data.index))
	data = data.join(temp)
	data = data.drop(columns = "country")

	#Do the same thing for province
	temp = data['province']
	temp2 = data['province']

	temp = pd.concat([temp,temp2], ignore_index = True)
	temp = pd.get_dummies(temp, prefix = 'province')
	temp = temp.head(len(data.index))
	data = data.join(temp)
	data = data.drop(columns = "province")

	#Back to hot encoding the rest of the data..
	# temp = pd.get_dummies(data.outcome, prefix = 'outcome')
	# data = data.join(temp)
	# data = data.drop(columns = "outcome")
	temp = pd.get_dummies(data.age_range, prefix = 'age_range')
	data = data.join(temp)
	data = data.drop(columns = "age_range")

	#The returned data is entirely hot-encoded
	return data


#load in the data
train = pd.read_csv(r'../results/cases_train_preprocessed.csv',dtype=object)
test =  pd.read_csv(r'../results/cases_test_preprocessed.csv',dtype=object)

#Dropping land column, age_range_ind, and Last_Updated, and source
train = train.drop(columns = "land")
train = train.drop(columns = "Last_Update")
test = test.drop(columns = "Last_Update")
train = train.drop(columns = "age_range_ind")
test = test.drop(columns = "age_range_ind")
train =train.drop(columns = "source")
test = test.drop(columns = "source")


#Hot-encoding (both to test and train)
train_copy = train.copy()
test_copy = test.copy()
train = Hot_Encode(train,test_copy)
test = Hot_Encode(test,train_copy)

#Handling additional_information (0 for no data, 1 for any data)
train.loc[train['additional_information'] == '[]', 'add_info'] = 0
train.loc[train['additional_information'] != '[]', 'add_info'] = 1
train = train.drop(columns = "additional_information")

#Spliting Dataset

# MY LAPTOP SUCKS THIS IS NEEDED SO IT RUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUNS DELETE BEFORE ACTUALLY RUNNING IT
train = train.head(10000)

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

knnPickle = open('knn_classifier.pkl','wb')
pickle.dump(classifier, knnPickle)

#Save my x_te, y_te data to open in new file
y_te.to_csv('y_te.csv', index=False)
pd.DataFrame(x_te).to_csv('x_te.csv', index =False)

print("Finished Constructing KNN model with k neightbours = " + str(k_neigh))

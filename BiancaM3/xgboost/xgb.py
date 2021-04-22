#Bianca Blackwell 301 304 375
#Course Project CMPT 459: XGBoost Classifier
from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np 


#TODO:Load & One-hot encode the Covid Dataset (same as in Knn)
#boston = load_boston()
train = pd.read_csv(r'../../data/cases_train_preprocessed.csv',dtype=object)
#Drop columns
train = train.drop(columns = ['date_confirmation','source','age_range_ind','land','Last_Update','country','province'])

#Deal with 0 - no additional data 1 - additional data
train.loc[train['additional_information'] == '[]', 'add_info'] = 0
train.loc[train['additional_information'] != '[]', 'add_info'] = 1
train = train.drop(columns = "additional_information")
#Hot Encoding for: Sex, age_range
temp = pd.get_dummies(train.sex, prefix='sex')
train = train.join(temp)
temp = pd.get_dummies(train.age_range, prefix = 'age_range')
train = train.join(temp)

train = train.drop(columns = ['sex','age_range'])
#separate the target variable and the rest of the variables
X = train.drop(columns = 'outcome')
y = train.outcome

labels = train['outcome'].unique()
labels_m = {v: k for v, k in enumerate(labels)}
labels_mb = {k: v for v, k in enumerate(labels)}
label_encoded_y = list(map(labels_mb.get,y))


#label_encoder = LabelEncoder()
#label_encoder = label_encoder.fit(y)
#label_encoded_y = label_encoder.transform(y)

X = X.apply(pd.to_numeric)
label_encoded_y = pd.to_numeric(label_encoded_y)

#same random state as in milestone 2
X_train, X_test, y_train, y_test = train_test_split(X,label_encoded_y,test_size = 0.2, random_state = 1)

#fit model 
model = xgb.XGBClassifier(use_label_encoder = False) #currently it's choosing its own parameters, i think
model.fit(X_train,y_train)
print(model)

#predict on test data
y_pred_test = model.predict(X_test)
predictions_te = [round(value) for value in y_pred_test]
#evaluations
predictions_te = list(map(labels_m.get,predictions_te))
y_test = list(map(labels_m.get,y_test))
print(classification_report(y_test,predictions_te))

#predict on train data
y_pred_train = model.predict(X_train)
predictions_tr = [round(value) for value in y_pred_train]
predictions_tr = list(map(labels_m.get,predictions_tr))
y_train = list(map(labels_m.get,y_train))
print(classification_report(y_train,predictions_tr))


#Tune parameters
#Do cross validation on the dataset

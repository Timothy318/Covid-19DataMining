#Bianca Blackwell 301 304 375
#Course Project CMPT 459: XGBoost Classifier
from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np 

#TODO:Load & One-hot encode the Covid Dataset (same as in Knn)
#boston = load_boston()
train = pd.read_csv(r'../../data/cases_train_preprocessed.csv',dtype=object)
print(train.head)
input()
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
print(train.head)
input()

#Normalization? (explore if I should still implement this)







data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target

#separate the target variable and the rest of the variables using .iloc
X, y = data.iloc[:,:-1],data.iloc[:,-1]
#convert data into optimized 'Dmatrix' struct
data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

#breakpoint
input()

#xg_reg = xgb.XGBRegressor(objective = 'multi:softprob, colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
#xg_reg.fit(X_train,y_train)
#preds = xg_reg.predict(X_test)

#rmse = np.sqrt(mean_squared_error(y_test,preds))
#print("RMSE: %f" % (rmse))

#TODO: To predict labels, instead of price..
#Change objective = multi:softprob
#feed in class labels (0,1,2,3) = (hospitalized, nonhospitalized, recovered, deceased)
#Figure out params for dataset

#Milestone 3 TODO:
#hypertuning of params




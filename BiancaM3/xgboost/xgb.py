#Bianca Blackwell 301 304 375
#Course Project CMPT 459: XGBoost Classifier
from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint, binom
from sklearn.metrics import roc_curve, auc, precision_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score, f1_score, classification_report, make_scorer
import numpy as np 
import pickle

#load in my training data
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

X = X.apply(pd.to_numeric)
label_encoded_y = pd.to_numeric(label_encoded_y)

#same random state as in milestone 2
X_train, X_test, y_train, y_test = train_test_split(X,label_encoded_y,test_size = 0.2, random_state = 1)

#HyperParameter Tuning (Same as Catboost model)
#I'm going to tune max_depth, max_child_weight, and eta (learning_rate)
grid_param={'learning_rate':uniform(0.1,0.2),
				#'max_depth':randint(3,10),
				#'min_child_weight':uniform(0.1,2)
				}

scorer = {'f1_macro' : make_scorer(f1_score, average='macro'),
'recall_macro': make_scorer(recall_score , average='macro'),
'accuracy': make_scorer(accuracy_score),
'recall_d' : make_scorer(recall_score,average=None,labels=['deceased']),
'f1_d' : make_scorer(f1_score, average=None, labels=['deceased'])}

model = xgb.XGBClassifier(learning_rate=0.2, min_child_weight=1,max_depth=8, reg_lambda = 2,objective='multi:softprob', eval_metric ='mlogloss', use_label_encoder = False, seed=1)

#TO DO adjust this. make it not take forever. maybe run it on my gpu on my pc
clf_rand = RandomizedSearchCV(estimator=model, n_iter=1, param_distributions=grid_param, scoring=scorer, cv= 3, refit='f1_d', n_jobs=-1, verbose=1)
model = clf_rand.fit(X_train,y_train)

#Save stuff
filename = '../../models/xgboost_classifier.pkl'
pickle.dump(model, open(filename, 'wb'))
print(clf_rand.best_params_)


# Output attempted parameters and resulting metrics
df = pd.DataFrame(clf_rand.cv_results)
df = df.sort_values(by='rank_test_f1_d')
df = df[df.columns.drop(list(df.filter(regex=r'(time)|(std)|(split)|(param_)|(rank)')))]
df = df[['params', 'mean_test_f1_d', 'mean_test_recall_d', 'mean_test_accuracy', 'mean_test_recall_macro','mean_test_f1_macro']]
df.rename(columns={"mean_test_f1_macro": "Overall F1-Score(Macro)", 
	"mean_test_recall_macro": "Overall Recall(Macro)",
	"mean_test_accuracy": "Overall Accuracy",
	"mean_test_f1_d": "F1-Score on Deceased",
	"mean_test_recall_d": "Recall on Deceased",
	"params": "Hyperparameters"},inplace=True)
df.to_csv('../../results/xgboost_tuning.csv',index=False)
print("saved attempted parameters to csv")

#Reformat everything below here
#TODO: use smote to get better prediction on the minority class
#(tentative) TODO: refactor again into nice neat functions

#fit model 
#min_child_weight tuned too low = overfitting
#max_depth too high = overfitting (typically 3-10)
#max_delta_step, might help with the extremely impbalanced
#subsample too high = overfitting (typcially 0.5-1)
#lambda can be used to reduce overfitting as well, it handles regularization
#scale_pos_weight should be positive when high class imbalance (like we have!)
#seed = 1 for parameter tuning
#learning_rate will shrink weights on model (can make model more robust, typically 0.01-0.2)
#model.fit(X_train,y_train)
#print(model)

#predict on test data
y_pred_test = model.predict(X_test)
predictions_te = [round(value) for value in y_pred_test]
#evaluations
predictions_te = list(map(labels_m.get,predictions_te))
y_test = list(map(labels_m.get,y_test))
print(classification_report(y_test,predictions_te))

#predict on train(validation) data
y_pred_train = model.predict(X_train)
predictions_tr = [round(value) for value in y_pred_train]
predictions_tr = list(map(labels_m.get,predictions_tr))
y_train = list(map(labels_m.get,y_train))
print(classification_report(y_train,predictions_tr))


#Tune parameters
#Do cross validation on the dataset

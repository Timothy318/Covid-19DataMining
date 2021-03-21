# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 01:17:05 2021

@author: wlian
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_curve, auc, precision_score, confusion_matrix,\
 ConfusionMatrixDisplay, recall_score, accuracy_score, f1_score, classification_report
import pickle
import matplotlib.pyplot as plt

#%%

train = pd.read_csv(r'../results/cases_train_preprocessed.csv',dtype=object)
# There are no information that can be introduce from this column, drop it
train.drop(columns=['land','Last_Update'],inplace=True)
# Reloading into dataframe cause empty strings to be NaN again
train['source'] = train['source'].fillna(value='')

test =  pd.read_csv(r'../results/cases_test_preprocessed.csv',dtype=object)
# There are no information that can be introduce from this column, drop it
test.drop(columns=['Last_Update'],inplace=True)
# Reloading into dataframe cause empty strings to be NaN again
test['source'] = test['source'].fillna(value='')


#%% 

cat_feature=['sex','province','country',
             'source','age_range_ind','age_range']
text_feature = ['additional_information']

df_tr, df_va = train_test_split(train, test_size=0.2, random_state = 1)

df_tr_x = df_tr.drop(columns="outcome")
df_tr_y = df_tr['outcome']

df_tr_data = Pool(data = df_tr_x,
               label = df_tr_y,
               cat_features = cat_feature, 
               text_features=text_feature)

df_va_x = df_va.drop(columns="outcome")
df_va_y = df_va['outcome']

df_va_data = Pool(data = df_va_x,
               label = df_va_y,
               cat_features = cat_feature, 
               text_features=text_feature)

#%% Training
model = CatBoostClassifier(n_estimators=300, 
                            task_type="GPU",
                            devices='0:1',
                            learning_rate=1,
                            depth=10,
                            loss_function='MultiClass',
                            random_seed = 1,
                            l2_leaf_reg = 2,
                            auto_class_weights = 'Balanced')

model.fit(df_tr_data)
filename = 'catboost_classifier.pkl'
pickle.dump(model, open(filename, 'wb'))
#%% Evaluation
label = ['hospitalized', 'nonhospitalized','recovered', 'deceased']

df_tr_pred = model.predict(df_tr_data)
ev_tr_report = classification_report(df_tr_y, df_tr_pred)
ev_tr_accuracy = accuracy_score(df_tr_y, df_tr_pred)
ev_tr_matrix = confusion_matrix(df_tr_y, df_tr_pred,labels=label)
ConfusionMatrixDisplay(ev_tr_matrix).plot()


df_va_pred = model.predict(df_va_data)
ev_va_report = classification_report(df_va_y, df_va_pred)
ev_va_accuracy = accuracy_score(df_va_y, df_va_pred)
ev_va_matrix = confusion_matrix(df_va_y, df_va_pred,labels=label)
ConfusionMatrixDisplay(ev_va_matrix).plot()


#%% 

n_estimators = [10, 25, 50, 100, 200, 300, 400]
# n_estimators = [10]
plt_tr_precision = []
plt_tr_recall = []
plt_tr_f1 = []
plt_tr_acc = []

plt_va_precision = []
plt_va_recall = []
plt_va_f1 = []
plt_va_acc = []

for n in n_estimators:
    tmp_model = CatBoostClassifier(n_estimators=n, 
                            task_type="GPU",
                            devices='0:1',
                            learning_rate=1,
                            depth=10,
                            loss_function='MultiClass',
                            random_seed = 1,
                            l2_leaf_reg = 2,
                            auto_class_weights = 'Balanced')

    tmp_model.fit(df_tr_data)

    tmp_tr_pred = tmp_model.predict(df_tr_data)
    tmp_tr_precision = precision_score(df_tr_y, tmp_tr_pred, average='weighted')
    tmp_tr_recall = recall_score(df_tr_y, tmp_tr_pred, average='weighted')
    tmp_tr_f1 = f1_score(df_tr_y, tmp_tr_pred, average='weighted')
    tmp_tr_accuracy = accuracy_score(df_tr_y, tmp_tr_pred)
    
    tmp_va_pred = tmp_model.predict(df_va_data)
    tmp_va_precision = precision_score(df_va_y, tmp_va_pred, average='weighted')
    tmp_va_recall = recall_score(df_va_y, tmp_va_pred, average='weighted')
    tmp_va_f1 = f1_score(df_va_y, tmp_va_pred, average='weighted')
    tmp_va_accuracy = accuracy_score(df_va_y, tmp_va_pred)
    
    plt_tr_precision.append(tmp_tr_precision)
    plt_tr_recall.append(tmp_tr_recall)
    plt_tr_f1.append(tmp_tr_f1)
    plt_tr_acc.append(tmp_tr_accuracy)
    
    plt_va_precision.append(tmp_va_precision)
    plt_va_recall.append(tmp_va_recall)
    plt_va_f1.append(tmp_va_f1)
    plt_va_acc.append(tmp_va_accuracy)
    
    print()
    # print("Training Accuracy: {:.2%}".format(tmp_tr_accuracy))
    # print("Validation Accuracy: {:.2%}".format(tmp_va_accuracy))
    
#%%
plt.figure(1)
plt.plot(n_estimators, plt_tr_precision,label="training")
plt.plot(n_estimators, plt_va_precision,label="validation")
plt.legend()
plt.title("Plot of weighted precision against n_est")
plt.yticks(np.arange(0.6, 0.9, 0.05))
plt.show()

plt.figure(2)
plt.plot(n_estimators, plt_tr_acc,label="training")
plt.plot(n_estimators, plt_va_acc,label="validation")
plt.legend()
plt.title("Plot of accuracy against n_est")
plt.yticks(np.arange(0.6, 0.9, 0.05))
plt.show()



del tmp_model
del tmp_tr_pred
del tmp_tr_precision
del tmp_tr_recall
del tmp_tr_f1
del tmp_tr_accuracy

del tmp_va_pred
del tmp_va_precision
del tmp_va_recall
del tmp_va_f1
del tmp_va_accuracy






#%%
# ev_tr_precision = precision_score(df_tr_y, df_tr_pred, average=None)
# ev_tr_recall = recall_score(df_tr_y, df_tr_pred, average=None)
# ev_tr_accuracy = accuracy_score(df_tr_y, df_tr_pred)
# ev_tr_f1 = f1_score(df_tr_y, df_tr_pred, average=None)

# ev_va_precision = precision_score(df_va_y, df_va_pred, average=None)
# ev_va_recall = recall_score(df_va_y, df_va_pred, average=None)
# ev_va_f1 = f1_score(df_va_y, df_va_pred, average=None)

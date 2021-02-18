# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:00:18 2021

@author: wlian
"""

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from numpy import quantile, where, random
from sklearn.ensemble import IsolationForest


cases_train = pd.read_csv('../results/cases_train_preprocessed.csv')
# cases_train = pd.read_csv('../data/cases_train.csv')
x = cases_train[['longitude','latitude']]
plt.scatter(cases_train['longitude'], cases_train['latitude'],s=1)
plt.show()




model=IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.0002),max_features=2)
model.fit(x)
x['scores']=model.decision_function(x[['longitude','latitude']])
x['anomaly']=model.predict(x.loc[:,['longitude','latitude']])
lofs_index = where(x['anomaly']==-1)
values = x.iloc[lofs_index]

plt.scatter(x['longitude'], x['latitude'],s=1)
plt.scatter(values['longitude'],values['latitude'], color='r',s=1)
plt.show()


clf = LocalOutlierFactor(n_neighbors=2)
clf.fit_predict(cases_train[['longitude','latitude']])


# index = cases_train.loc[cases_train['province'].str.contains(r"\bor\b", regex = True)].index
# cases_train.drop(index,inplace=True)

# freq_prov = cases_train['province'].value_counts().to_frame()
# freq_prov.reset_index(inplace=True)

# index = cases_train.loc[cases_train['province'].str.contains(r"\bdiu\b", regex = True)]
# index = cases_train.loc[cases_train['province'].str.contains(r"free state", regex = True)]

# date_gb_val = cases_train.groupby(['longitude','latitude'])['province'].value_counts()

# lof = LocalOutlierFactor(n_neighbors=200, contamination=.001)
# y_pred = lof.fit_predict(x)
# lofs_index = where(y_pred==-1)
# values = x.iloc[lofs_index]

# model = LocalOutlierFactor(n_neighbors=50) 
# model.fit_predict(x)
# lof = model.negative_outlier_factor_ 
# thresh = quantile(lof, .002)

# index = where(lof<=thresh)
# values = x.iloc[index]
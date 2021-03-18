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
from mpl_toolkits.basemap import Basemap
import numpy as np
from global_land_mask import globe

def q1_3():
    cases_train = pd.read_csv('../results/cases_train_preprocessed_imp.csv')
    print(cases_train.age.describe())
    # print(len(cases_train))
    plt.hist(cases_train.age)
    plt.show()
    freq_age = cases_train['age'].value_counts().to_frame()
    freq_age.reset_index(inplace=True)
    
    freq_sex = cases_train['sex'].value_counts().to_frame()
    freq_sex.reset_index(inplace=True)
    
    freq_country = cases_train['country'].value_counts().to_frame()
    freq_country.reset_index(inplace=True)
    
    freq_prov = cases_train['province'].value_counts().to_frame()
    freq_prov.reset_index(inplace=True)
    
    index = cases_train.loc[cases_train['province'].str.contains(r"\bor\b", regex = True)].index
    cases_train.drop(index,inplace=True)
    # print(len(cases_train))
    plt.figure(figsize=(20,20))
    m = Basemap(projection='cyl',
    	   llcrnrlat = -90,
    	   urcrnrlat = 90,
    	   llcrnrlon = -180,
    	   urcrnrlon = 180,
    	   resolution = 'h')
    m.shadedrelief()
    
    cases_train['land'] = globe.is_land(cases_train.latitude,cases_train.longitude)
    
    values = cases_train.loc[cases_train.land==False]
    values = values.loc[values.country != 'philippines']
    values = values.loc[ (values.province == 'rio grande do sul') |
                          (values.province == 'santa catarina') | 
                          (values.province == 'eastern cape')]
    index = values.index
    cases_train.drop(index,inplace=True)
    m.drawcoastlines()
    m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
    m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])
    m.scatter(cases_train.longitude,cases_train.latitude,latlon=True,s=0.5)
    m.scatter(values['longitude'],values['latitude'],latlon=True, color='r',s=0.5)
    plt.show()
    
    index = cases_train.loc[(cases_train.province=='ontario') & (cases_train.country=='china')].index
    cases_train.drop(index,inplace=True)

    
    cases_train.to_csv('../results/cases_train_preprocessed_outlier.csv',index=False)



















# model=IsolationForest(n_estimators=10, max_samples='auto', contamination=float(0.001),max_features=2)
# model.fit(cases_train[['longitude','latitude']])

# cases_train['anomaly']=model.predict(cases_train.loc[:,['longitude','latitude']])
# index = where(cases_train['anomaly']==-1)
# values = cases_train.iloc[index]

# plt.scatter(cases_train['longitude'], cases_train['latitude'],s=1)
# plt.scatter(values['longitude'],values['latitude'], color='r',s=1)
# plt.show()

# index = cases_train.loc[cases_train['province'].str.contains(r"far eastern", regex = True)]
# index = cases_train.loc[cases_train['country'].str.contains(r"russia", regex = True)]


# x['scores']=model.decision_function(x[['longitude','latitude']])

# index = cases_train.loc[cases_train['province'].str.contains(r"free state", regex = True)]

# plt.scatter(cases_train['longitude'], cases_train['latitude'],s=1)
# plt.show()




# clf = LocalOutlierFactor(n_neighbors=2)
# clf.fit_predict(cases_train[['longitude','latitude']])

# index = cases_train.loc[cases_train['province'].str.contains(r"hospital", regex = True)]
# cases_train.drop(index,inplace=True)

# freq_prov = cases_train['province'].value_counts().to_frame()
# freq_prov.reset_index(inplace=True)

# index = cases_train.loc[cases_train['province'].str.contains(r"\bdiu\b", regex = True)]


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





























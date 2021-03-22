# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:59:31 2021

@author: wlian
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:25:16 2021

@author: wlian
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
from sklearn.impute import KNNImputer


import reverse_geocoder as rg

def age_range_dummy(x):
    if "-" in x or "+" in x:
        return True
    else:
        return False

def process_date_numerical(x):
    date_time_obj = datetime.strptime(x,'%d.%m.%Y')
    timestamp = datetime.timestamp(date_time_obj)
    return timestamp

def process_mode(x):
    if type(x) == np.ndarray:
        return x[0]
    else:
        return x
    
def process_date_range(x):
    if '-' in x:
        x = x.split('-')
        return x[0].strip()
    else:
        return x
    
def parse_domain(x):
    domain = urlparse(x).netloc
    # if '.gov' in domain:
    #     return "official"
    return domain


def impute(criteria,target,dataset,func):
    merge_criteria = criteria 
    date_gb_mode = dataset.loc[dataset[target].notnull()]
    date_gb_mode = date_gb_mode.groupby(merge_criteria)[target].agg(pd.Series.mode).to_frame()
    date_gb_mode = date_gb_mode.rename(columns={target: 'fill'})
    date_gb_mode.reset_index(inplace=True)
    if func:
        date_gb_mode['fill'] = date_gb_mode['fill'].apply(func)
    
    dataset = dataset.merge(date_gb_mode,on = merge_criteria, how='left')
    dataset[target] = dataset[target].fillna(dataset['fill'])
    dataset.drop(columns=['fill'],inplace=True)
    return dataset

def process_province():
    

##############################################################################
# Loading the data and gathering frequency information
cases_train = pd.read_csv('../data/cases_train.csv')
cases_test = pd.read_csv('../data/cases_test.csv')
location = pd.read_csv('../data/location.csv')

cases_train['latitude'] = cases_train['latitude'].astype(float)
cases_train['longitude'] = cases_train['longitude'].astype(float)

# train = pd.read_csv('../data/cases_train.csv')
# train = train.loc[train['longitude'] == 0]

# freq_age_init = cases_train['age'].value_counts().to_frame()
# freq_age_init.reset_index(inplace=True)

# freq_prov_init = cases_train['province'].value_counts().to_frame()
# freq_prov_init.reset_index(inplace=True)

# freq_country_init = cases_train['country'].value_counts().to_frame()
# freq_country_init.reset_index(inplace=True)

# freq_date_init = cases_train['date_confirmation'].value_counts().to_frame()
# freq_date_init.reset_index(inplace=True)

# freq_long_init = cases_train['longitude'].value_counts().to_frame()
# freq_long_init.reset_index(inplace=True)

# freq_lat_init = cases_train['latitude'].value_counts().to_frame()
# freq_lat_init.reset_index(inplace=True)

# freq_sex_init = cases_train['sex'].value_counts().to_frame()
# freq_sex_init.reset_index(inplace=True)

# unique_age = set(freq_age_init['index'])

# vc_init = cases_train.notnull().sum().to_frame()
# vc_init['prob'] = vc_init/len(cases_train)
# vc_init.reset_index(inplace=True)

############################################################################################################
# Imputation process
# Imputating longitude / latitude variable 
  # Cannot impute because its all null for other attribute
  # After removing the missing long, missing lat disappear
# missing_long = cases_train.loc[cases_train.longitude.isnull()]
cases_train = cases_train.loc[cases_train.longitude.notnull()]

###########################################################################################################

# Imputating Additional information / Source 
cases_train['additional_information'] = cases_train['additional_information'].fillna(value='')
cases_train['source'] = cases_train['source'].fillna(value='')


# cases_train['source'] = cases_train['source'].apply(parse_domain)
# freq_source_imp = cases_train['source'].value_counts().to_frame()
# freq_source_imp.reset_index(inplace=True)

############################################################################################################
# # Imputating date confirmation
# missing_date = cases_train.loc[cases_train['date_confirmation'].isnull()]
# # date_gb_val = cases_train.groupby(['province','country'])['date_confirmation'].value_counts()
# # date_gb_agg = cases_train.groupby(['province','country'])['date_confirmation'].count()

# cases_train = impute(['longitude','latitude'],'date_confirmation',cases_train,process_mode)
# cases_train = impute(['province','country'],'date_confirmation',cases_train,process_mode)

# # Fill in the remaining with mode - most frequent value
# cases_train['date_confirmation'] = cases_train['date_confirmation'].fillna(cases_train['date_confirmation'].mode().iloc[0])

# # # Select the first one in date range
# cases_train['date_confirmation'] = cases_train['date_confirmation'].apply(process_date_range) 
# cases_train['date_confirmation'] = cases_train['date_confirmation'].apply(process_date_numerical) 

# # missing_date = cases_train.loc[cases_train['date_confirmation'].isnull()]
# # freq_date_imp = cases_train['date_confirmation'].value_counts().to_frame()
# # freq_date_imp.reset_index(inplace=True)


# vc_imp = cases_train.notnull().sum().to_frame()
# vc_imp['prob'] = vc_imp/len(cases_train)
# vc_imp.reset_index(inplace=True)

############################################################################################################
# Imputating provincial , country
missing_country = cases_train.loc[cases_train['country'].isnull()]
cases_train = impute(['longitude','latitude'],'country',cases_train,process_mode)
cases_train.loc[(cases_train.country.isnull()),'country'] = "China"

cases_train = impute(['longitude','latitude'],'province',cases_train,process_mode)
missing_prov = cases_train.loc[cases_train['province'].isnull()]


if __name__ == '__main__':
    cases_train['geom'] = list(zip(cases_train['latitude'] , cases_train['longitude']))
    geo = rg.RGeocoder(mode=2, verbose=True)
    
    unique_arr = list(cases_train["geom"])
    # unique_arr = list(cases_train["geom"].unique())
    unique_arr = pd.DataFrame(geo.query(unique_arr))
    unique_arr['lon'] = unique_arr['lon'].astype(float)
    unique_arr['lat'] = unique_arr['lat'].astype(float)
    cases_train = cases_train.join(unique_arr[['admin1','admin2','cc','lon','lat']])
    cases_train.loc[(cases_train.province.isnull()),'province'] = cases_train['admin2']
    
cases_train['province'] = cases_train['province'].replace('',np.nan)
cases_train = impute(['country'],'province',cases_train,process_mode)
cases_train = cases_train.dropna(subset=['sex', 'age','province'], how='all')
missing_prov1 = cases_train.loc[cases_train['province'].isnull()]
# missing_prov1 = cases_train.loc[cases_train['province'] == '']

############################################################################################################
# Imputating age variable
# Filling NaN with empty value and process age range and replace with the average
# age_detail = pd.DataFrame(columns=['age','freq'])
# age_regx = ['^0\.','90\+','80\+','-']
# for regx in age_regx:
#     age_detail = age_detail.append({'age':regx,'freq': len(filter_age(regx))}, ignore_index=True)
# age_detail['prob'] = age_detail['freq'] / int(vc_init.loc[vc_init['index'] == 'age'][0])

# cases_train['age'] = cases_train['age'].fillna(value='')
# cases_train['age_range_ind'] = cases_train['age'].apply(age_range_dummy)
# cases_train['age'] = cases_train['age'].apply(process_age)
# cases_train['age'] = cases_train['age'].astype(float)
# cases_train['age'] = round(cases_train['age'],1)
# cases_train.loc[(cases_train.age < 1),'age'] = 0

# imputer = KNNImputer(n_neighbors=2)
# cases_train = imputer.fit_transform(cases_train)

# cases_train = impute(['longitude','latitude','date_confirmation'],'age',cases_train,process_mode)
# cases_train = impute(['longitude','latitude'],'age',cases_train,process_mode)
# cases_train = impute(['province','country'],'age',cases_train,process_mode)
# cases_train['age'] = cases_train['age'].fillna(cases_train['age'].mode().iloc[0])

# freq_age_imp = cases_train['age'].value_counts().to_frame()
# freq_age_imp.reset_index(inplace=True)

############################################################################################################
































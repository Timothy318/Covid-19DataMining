# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:25:16 2021

@author: wlian
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from datetime import datetime
from urllib.parse import urlparse

def process_age(x):
    if "-" in x:
        parse_age = x.split("-")
        parse_age = [int(i) for i in parse_age if i] 
        if len(parse_age) > 1:
            return sum(parse_age)/len(parse_age)
        else:
            if parse_age[0] > 18:
                return float(18 + parse_age[0]/2)
            else:
                return float(parse_age[0]/2)
    elif 'month' in x:
        parse_age = x.split("month")
        parse_age = [float(i) for i in parse_age if i]
        return parse_age[0]/12
    elif '+' in x:
        parse_age = x.split("+")
        parse_age = [float(i) for i in parse_age if i]
        return (parse_age[0] + 100)/2
    elif x:
        return float(x)
    else:
        return None
    
def filter_age(cond):
    table = cases_train[cases_train['age'].str.contains(cond, regex=True)]
    return table

def age_range_dummy(x):
    if "-" in x or "+" in x:
        return True
    else:
        return False

def process_date_numerical(x):
    date_time_obj = datetime.strptime(x,'%d.%m.%Y')
    timestamp = datetime.timestamp(date_time_obj)
    return timestamp

def process_date(x):
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

cases_train['source'] = cases_train['source'].apply(parse_domain)
# freq_source_imp = cases_train['source'].value_counts().to_frame()
# freq_source_imp.reset_index(inplace=True)

############################################################################################################

# Imputating age variable
# Filling NaN with empty value and process age range and replace with the average
# age_detail = pd.DataFrame(columns=['age','freq'])
# age_regx = ['^0\.','90\+','80\+','-']
# for regx in age_regx:
#     age_detail = age_detail.append({'age':regx,'freq': len(filter_age(regx))}, ignore_index=True)
# age_detail['prob'] = age_detail['freq'] / int(vc_init.loc[vc_init['index'] == 'age'][0])

cases_train['age'] = cases_train['age'].fillna(value='')
cases_train['age_range_ind'] = cases_train['age'].apply(age_range_dummy)
cases_train['age'] = cases_train['age'].apply(process_age)
cases_train['age'] = cases_train['age'].astype(float)
cases_train['age'] = round(cases_train['age'],1)
cases_train.loc[(cases_train.age < 1),'age'] = 0

# freq_age_imp = cases_train['age'].value_counts().to_frame()
# freq_age_imp.reset_index(inplace=True)

############################################################################################################

# Imputating date confirmation
# missing_date = cases_train.loc[cases_train['date_confirmation'].isnull()]
# date_gb_val = cases_train.groupby(['province','country'])['date_confirmation'].value_counts()
# date_gb_agg = cases_train.groupby(['province','country'])['date_confirmation'].count()

# Merge by 'longitude','latitude'
merge_criteria = ['longitude','latitude']  # ['province','country']
date_gb_mode = cases_train.loc[cases_train['date_confirmation'].notnull()]
date_gb_mode = date_gb_mode.groupby(merge_criteria)['date_confirmation'].agg(pd.Series.mode).to_frame()
date_gb_mode = date_gb_mode.rename(columns={'date_confirmation': 'fill'})
date_gb_mode.reset_index(inplace=True)
date_gb_mode['fill'] = date_gb_mode['fill'].apply(process_date)

cases_train = cases_train.merge(date_gb_mode,on = merge_criteria, how='left')
cases_train['date_confirmation'] = cases_train['date_confirmation'].fillna(cases_train['fill'])
cases_train.drop(columns=['fill'],inplace=True)


# Merge by province , country
merge_criteria = ['province','country']  # ['longitude','latitude']
date_gb_mode = cases_train.loc[cases_train['date_confirmation'].notnull()]
date_gb_mode = date_gb_mode.groupby(merge_criteria)['date_confirmation'].agg(pd.Series.mode).to_frame()
date_gb_mode = date_gb_mode.rename(columns={'date_confirmation': 'fill'})
date_gb_mode.reset_index(inplace=True)
date_gb_mode['fill'] = date_gb_mode['fill'].apply(process_date)

cases_train = cases_train.merge(date_gb_mode,on = merge_criteria, how='left')
cases_train['date_confirmation'] = cases_train['date_confirmation'].fillna(cases_train['fill'])
cases_train.drop(columns=['fill'],inplace=True)


# Fill in the remaining with mode - most frequent value
cases_train['date_confirmation'] = cases_train['date_confirmation'].fillna(cases_train['date_confirmation'].mode().iloc[0])

# Select the first one in date range
cases_train['date_confirmation'] = cases_train['date_confirmation'].apply(process_date_range) 
cases_train['date_confirmation'] = cases_train['date_confirmation'].apply(process_date_numerical) 

# missing_date = cases_train.loc[cases_train['date_confirmation'].isnull()]
freq_date_imp = cases_train['date_confirmation'].value_counts().to_frame()
freq_date_imp.reset_index(inplace=True)



vc_imp = cases_train.notnull().sum().to_frame()
vc_imp['prob'] = vc_imp/len(cases_train)
vc_imp.reset_index(inplace=True)

############################################################################################################


































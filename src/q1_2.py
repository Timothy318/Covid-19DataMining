# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:25:16 2021

@author: wlian
"""

import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
import reverse_geocoder as rg

def process_age(x,dataset):
    if "-" in x:
        parse_age = x.split("-")
        parse_age = list(filter(None, parse_age))
        
        num_range = False
        for i in parse_age:
            if i.isalpha():
                num_range = True
                
        if num_range:
            return None
        
        parse_age = [int(i.strip()) for i in parse_age]
        if len(parse_age) > 1:
            if parse_age[1] - parse_age[0] > 40 and dataset == 'train':
                return 10000
            return sum(parse_age)/len(parse_age)
        else:
            if parse_age[0] > 18:
                return float(18 + parse_age[0]/2)
            else:
                return float(parse_age[0]/2)
    elif 'month' in x:
        parse_age = x.split("month")
        parse_age = [float(i) for i in parse_age if i and i.strip().isnumeric()]
        return parse_age[0]/12
    elif '+' in x:
        parse_age = x.split("+")
        parse_age = [float(i) for i in parse_age if i and i.strip().isnumeric()]
        return (parse_age[0] + 100)/2
    elif x:
        return float(x)
    else:
        return None
    
def filter_age(cond):
    table = cases_train[cases_train['age'].str.contains(cond, regex=True)]
    return table

def age_range(x):
    if x>=60:
        return 'senior'
    elif 60 > x and x >= 18:
        return 'adult'
    elif 18 > x and x >= 5:
        return 'child'
    else:
        return 'baby'

def age_range_dummy(x):
    if "-" in x or "+" in x:
        return "True"
    elif x:
        return "False"
    else:
        return "Missing"

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

def process_province(dataset):
    dataset = impute(['longitude','latitude'],'country',dataset,process_mode)
    dataset.loc[(dataset.province == "taiwan"),'country'] = "China"
    dataset = impute(['longitude','latitude'],'province',dataset,process_mode)

    dataset['geom'] = list(zip(dataset['latitude'] , dataset['longitude']))
    geo = rg.RGeocoder(mode=2, verbose=True)
    
    unique_arr = list(dataset["geom"])
    unique_arr = pd.DataFrame(geo.query(unique_arr))
    unique_arr['lon'] = unique_arr['lon'].astype(float)
    unique_arr['lat'] = unique_arr['lat'].astype(float)
    dataset = dataset.join(unique_arr[['admin1','admin2','cc','lon','lat']])
    dataset.loc[(dataset.province.isnull()),'province'] = dataset['admin2']
    dataset.drop(columns=['admin1','admin2','cc','lon','lat','geom'],inplace=True)
    return dataset

##############################################################################
# Loading the data and gathering frequency information
cases_train = pd.read_csv('../data/cases_train.csv')
cases_test = pd.read_csv('../data/cases_test.csv')
location = pd.read_csv('../data/location.csv')

cases_train['latitude'] = cases_train['latitude'].astype(float)
cases_train['longitude'] = cases_train['longitude'].astype(float)
cases_train['province'] = cases_train['province'].str.lower()
cases_train['country'] = cases_train['country'].str.lower()
cases_train['additional_information'] = cases_train['additional_information'].str.lower()
cases_train['source'] = cases_train['source'].str.lower()
cases_train.dropna(subset=['sex', 'age','province'], how='all',inplace=True)

############################################################################################################
# Imputation process
# Imputating longitude / latitude variable 

cases_train = cases_train.loc[cases_train.longitude.notnull()]

############################################################################################################
# Impute age
cases_train['age'] = cases_train['age'].fillna(value='')
cases_train['age_range_ind'] = cases_train['age'].apply(age_range_dummy)
cases_train['age'] = cases_train['age'].apply(process_age,dataset = 'train')
cases_train['age'] = cases_train['age'].astype(float)
cases_train['age'] = round(cases_train['age'],1)
cases_train.loc[(cases_train.age < 1) & (cases_train.age >0),'age'] = 0
index = cases_train.loc[(cases_train.age == 10000),'age'].index
cases_train.drop(index,inplace=True)

cases_train = impute(['longitude','latitude','sex','date_confirmation'],'age',cases_train,process_mode)
cases_train = impute(['longitude','latitude','date_confirmation'],'age',cases_train,process_mode)
cases_train['age'] = cases_train['age'].fillna(cases_train['age'].mode().iloc[0])
cases_train['age_range'] = cases_train['age'].apply(age_range)

###########################################################################################################

# Imputating Additional information / Source 
cases_train['additional_information'] = cases_train['additional_information'].fillna(value='')
cases_train['source'] = cases_train['source'].fillna(value='')
cases_train['source'] = cases_train['source'].apply(parse_domain)

###########################################################################################################
# # Imputating date confirmation
cases_train = impute(['longitude','latitude'],'date_confirmation',cases_train,process_mode)
cases_train = impute(['province','country'],'date_confirmation',cases_train,process_mode)

# Fill in the remaining with mode - most frequent value
cases_train['date_confirmation'] = cases_train['date_confirmation'].fillna(cases_train['date_confirmation'].mode().iloc[0])

# Select the first one in date range
cases_train['date_confirmation'] = cases_train['date_confirmation'].apply(process_date_range) 
cases_train['date_confirmation'] = cases_train['date_confirmation'].apply(process_date_numerical) 
############################################################################################################
# Imputating provincial , country
if __name__ == '__main__':
    # missing_country = cases_train.loc[cases_train['country'].isnull()]
    
    cases_train = process_province(cases_train)
    cases_train['province'] = cases_train['province'].replace('',np.nan)
    cases_train = impute(['country'],'province',cases_train,process_mode)
    cases_train.loc[(cases_train.province == 'CABA'),'province'] = 'Buenos Aires'
    cases_train.loc[(cases_train.province.isnull()),'province'] = 'Invalid'

############################################################################################################
#Impute gender
cases_train.loc[(cases_train.sex.isnull()),'sex'] = 'Not Available'

###########################################################################################################
    

""" Imputing Test dataset """

############################################################################################################
# Impute age
cases_test['age'] = cases_test['age'].fillna(value='')
cases_test['age_range_ind'] = cases_test['age'].apply(age_range_dummy)
cases_test['age'] = cases_test['age'].apply(process_age,dataset='test')
cases_test['age'] = cases_test['age'].astype(float)
cases_test['age'] = round(cases_test['age'],1)
cases_test.loc[(cases_test.age < 1) & (cases_test.age >0),'age'] = 0

cases_test = impute(['longitude','latitude','sex','date_confirmation'],'age',cases_test,process_mode)
cases_test = impute(['longitude','latitude','date_confirmation'],'age',cases_test,process_mode)
cases_test['age'] = cases_test['age'].fillna(cases_test['age'].mode().iloc[0])
cases_test['age_range'] = cases_test['age'].apply(age_range)

############################################################################################################
# Imputation process
# Imputating longitude / latitude variable 

cases_test = cases_test.loc[cases_test.longitude.notnull()]
###########################################################################################################

# Imputating Additional information / Source 
cases_test['additional_information'] = cases_test['additional_information'].fillna(value='')
cases_test['source'] = cases_test['source'].fillna(value='')
cases_test['source'] = cases_test['source'].apply(parse_domain)

###########################################################################################################
# # Imputating date confirmation
cases_test = impute(['longitude','latitude'],'date_confirmation',cases_test,process_mode)
cases_test = impute(['province','country'],'date_confirmation',cases_test,process_mode)

# Fill in the remaining with mode - most frequent value
cases_test['date_confirmation'] = cases_test['date_confirmation'].fillna(cases_test['date_confirmation'].mode().iloc[0])

# Select the first one in date range
cases_test['date_confirmation'] = cases_test['date_confirmation'].apply(process_date_range) 
cases_test['date_confirmation'] = cases_test['date_confirmation'].apply(process_date_numerical) 
############################################################################################################
# Imputating provincial , country
if __name__ == '__main__':
    # missing_country = cases_test.loc[cases_test['country'].isnull()]
    
    cases_test = process_province(cases_test)
    cases_test['province'] = cases_test['province'].replace('',np.nan)
    cases_test = impute(['country'],'province',cases_test,process_mode)
    cases_test.loc[(cases_test.province == 'CABA'),'province'] = 'Buenos Aires'
    cases_test.loc[(cases_test.province.isnull()),'province'] = 'Invalid'

############################################################################################################
#Impute gender
cases_test.loc[(cases_test.sex.isnull()),'sex'] = 'Not Available'

###########################################################################################################


cases_train.to_csv('../results/cases_train_preprocessed.csv',index=False)
cases_test.to_csv('../results/cases_test_preprocessed.csv',index=False)

















##############################################################################
# Test Code

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

# vc_init = cases_train.notnull().sum().to_frame()
# vc_init['prob'] = vc_init/len(cases_train)
# vc_init.reset_index(inplace=True)
    
# freq_prov_imp = cases_train['province'].value_counts().to_frame()
# freq_prov_imp.reset_index(inplace=True)
    
# freq_source_imp = cases_train['source'].value_counts().to_frame()
# freq_source_imp.reset_index(inplace=True)

# missing_date = cases_train.loc[cases_train['date_confirmation'].isnull()]
# date_gb_val = cases_train.groupby(['province','country'])['date_confirmation'].value_counts()
# date_gb_agg = cases_train.groupby(['province','country'])['date_confirmation'].count()

# missing_date = cases_train.loc[cases_train['date_confirmation'].isnull()]
# freq_date_imp = cases_train['date_confirmation'].value_counts().to_frame()
# freq_date_imp.reset_index(inplace=True)
##############################################################################















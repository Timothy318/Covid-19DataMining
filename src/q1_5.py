# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 00:07:24 2021

@author: wlian
"""
import pandas as pd
import numpy as np

def q1_5(name):
    if name == 'train':
        dataset = pd.read_csv('../results/cases_train_preprocessed_outlier.csv')
    else:
        dataset = pd.read_csv('../results/cases_test_preprocessed_imp.csv')
    
    print(len(dataset))
    
    location = pd.read_csv('../results/location_transformed.csv')
    location.Province_State = location.Province_State.str.lower()
    location.Country_Region = location.Country_Region.str.lower()
    location.loc[location.Country_Region=='korea, south','Country_Region'] = 'south korea'
    location.loc[location.Province_State=='andalusia','Province_State'] = 'andalucia'
    location.loc[location.Country_Region=='czechia','Country_Region'] = 'czech republic'
    location.loc[location.Country_Region=='congo (brazzaville)','Country_Region'] = 'republic of congo'
    location.loc[location.Country_Region=='congo (kinshasa)','Country_Region'] = 'democratic republic of the congo'

    merge = pd.merge(dataset, location, how='left', left_on=['latitude','longitude'], right_on=['Lat','Long_'])
    merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
    result = merge
    
    index = list(set(dataset.index) - set(merge.index))
    dataset = dataset.iloc[index]
    dataset.reset_index(inplace=True,drop=True)
    
    
    cur_loc = location.loc[location.Province_State.notnull()]
    merge = pd.merge(dataset, cur_loc, how='left', left_on=['province','country'], right_on=['Province_State','Country_Region'])
    merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
    result = pd.concat([result, merge])
    result.drop(['Lat', 'Long_','Province_State','Country_Region','Combined_Key'], axis=1,inplace=True)
    
    index = list(set(dataset.index) - set(merge.index))
    dataset = dataset.iloc[index]
    dataset.reset_index(inplace=True,drop=True)
    
    
    cur_loc = np.setdiff1d(location.Country_Region.unique(),cur_loc.Country_Region.unique())
    cur_loc = location.loc[location.Country_Region.isin(cur_loc)]
    merge = pd.merge(dataset, cur_loc, how='left', left_on=['country'], right_on=['Country_Region'])
    merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
    result = pd.concat([result, merge])
    result.drop(['Lat', 'Long_','Province_State','Country_Region','Combined_Key'], axis=1,inplace=True)
    
    index = list(set(dataset.index) - set(merge.index))
    dataset = dataset.iloc[index]
    dataset.reset_index(inplace=True,drop=True)
    
    freq_country = dataset['country'].value_counts().to_frame()
    freq_country.reset_index(inplace=True)
    
    cur_loc = list(freq_country['index'])
    cur_loc = location.loc[location['Country_Region'].isin(cur_loc)]
    cur_loc = location.groupby('Country_Region').agg('mean')
    
    cur_loc.reset_index(inplace=True)
    cur_loc.drop(['Lat', 'Long_'], axis=1,inplace=True)
    cur_loc['Case-Fatality_Ratio'] = (cur_loc.Deaths/cur_loc.Confirmed)*100
    
    merge = pd.merge(dataset, cur_loc, how='left', left_on=['country'], right_on=['Country_Region'])
    merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
    result = pd.concat([result, merge])
    result.drop(['Country_Region'], axis=1,inplace=True)
    
    index = list(set(dataset.index) - set(merge.index))
    dataset = dataset.iloc[index]
    dataset.reset_index(inplace=True,drop=True)
    

    dataset['Confirmed'] = result.Confirmed.mean()
    dataset['Deaths'] = result.Deaths.mean()
    dataset['Recovered'] = result.Recovered.mean()
    dataset['Active'] = result.Active.mean()
    dataset['Incidence_Rate'] = result.Incidence_Rate.mean()
    dataset['Case-Fatality_Ratio'] = result['Case-Fatality_Ratio'].mean()
    
    result = pd.concat([result, dataset])
    if name == 'train':
        result.to_csv(r'../results/cases_train_preprocessed.csv',index=False)
    else:
        result.to_csv(r'../results/cases_test_preprocessed.csv',index=False)


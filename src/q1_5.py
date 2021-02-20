# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 00:07:24 2021

@author: wlian
"""
import pandas as pd
import numpy as np
cases_train = pd.read_csv('../results/cases_train_preprocessed_imp.csv')
print(len(cases_train))
location = pd.read_csv('../results/location_transformed.csv')
location.Province_State = location.Province_State.str.lower()
location.Country_Region = location.Country_Region.str.lower()
location.loc[location.Country_Region=='korea, south','Country_Region'] = 'south korea'

merge_lonlat = pd.merge(cases_train, location, how='left', left_on=['latitude','longitude'], right_on=['Lat','Long_'])
merge_lonlat.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)

index = list(set(cases_train.index) - set(merge_lonlat.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)


prov_exist = location.loc[location.Province_State.notnull()]
merge_prov_country = pd.merge(cases_train, prov_exist, how='left', left_on=['province','country'], right_on=['Province_State','Country_Region'])
merge_prov_country.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)

index = list(set(cases_train.index) - set(merge_prov_country.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)

country_list = location.Country_Region.unique()
prov_exist = prov_exist.Country_Region.unique()
main_list = np.setdiff1d(country_list,prov_exist)

country_exist = location.loc[location.Country_Region.isin(main_list)]
merge_country = pd.merge(cases_train, country_exist, how='left', left_on=['country'], right_on=['Country_Region'])
merge_country.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)

index = list(set(cases_train.index) - set(merge_country.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)

merge_prov = pd.merge(cases_train, location, how='left', left_on=['province'], right_on=['Province_State'])
merge_prov.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)

index = list(set(cases_train.index) - set(merge_prov.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)

cases_train = cases_train.loc[cases_train.country=='colombia']

freq_prov = cases_train['province'].value_counts().to_frame()
freq_prov.reset_index(inplace=True)

freq_country = cases_train['country'].value_counts().to_frame()
freq_country.reset_index(inplace=True)
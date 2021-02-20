# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:26:48 2021

@author: wlian
"""

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
location.loc[location.Province_State=='andalusia','Province_State'] = 'andalucia'
location.loc[location.Country_Region=='czechia','Country_Region'] = 'czech republic'

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


result = pd.concat([merge_prov_country, merge_country, merge_prov,merge_lonlat])
result.drop(['Lat', 'Long_','Province_State','Country_Region','Combined_Key'], axis=1,inplace=True)





freq_prov = cases_train['province'].value_counts().to_frame()
freq_prov.reset_index(inplace=True)

freq_country = cases_train['country'].value_counts().to_frame()
freq_country.reset_index(inplace=True)

new_loc = list(freq_country['index'])
location = location.loc[location['Country_Region'].isin(new_loc)]
new_loc = location.groupby('Country_Region').agg('mean')
new_loc.reset_index(inplace=True)
new_loc.drop(['Lat', 'Long_'], axis=1,inplace=True)
new_loc['Case-Fatality_Ratio'] = (new_loc.Deaths/new_loc.Confirmed)*100


merge_country_remain = pd.merge(cases_train, new_loc, how='left', left_on=['country'], right_on=['Country_Region'])
merge_country_remain.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)

index = list(set(cases_train.index) - set(merge_country_remain.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)



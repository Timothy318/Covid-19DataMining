# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:23:28 2021

@author: wlian
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 00:07:24 2021

@author: wlian
"""
import pandas as pd
import numpy as np

cases_train = pd.read_csv('../results/cases_test_preprocessed.csv')
print(len(cases_train))

location = pd.read_csv('../results/location_transformed.csv')
location.Province_State = location.Province_State.str.lower()
location.Country_Region = location.Country_Region.str.lower()
location.loc[location.Country_Region=='korea, south','Country_Region'] = 'south korea'
location.loc[location.Province_State=='andalusia','Province_State'] = 'andalucia'
location.loc[location.Country_Region=='czechia','Country_Region'] = 'czech republic'
# location.loc[location.Country_Region.str.contains('congo'),'Country_Region'] = 'congo'

merge = pd.merge(cases_train, location, how='left', left_on=['latitude','longitude'], right_on=['Lat','Long_'])
merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
result = merge

index = list(set(cases_train.index) - set(merge.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)

# print('merge:',len(merge))
# print('case_train',len(cases_train))

cur_loc = location.loc[location.Province_State.notnull()]
merge = pd.merge(cases_train, cur_loc, how='left', left_on=['province','country'], right_on=['Province_State','Country_Region'])
merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
result = pd.concat([result, merge])
result.drop(['Lat', 'Long_','Province_State','Country_Region','Combined_Key'], axis=1,inplace=True)

index = list(set(cases_train.index) - set(merge.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)

# print('merge:',len(merge))
# print('case_train',len(cases_train))

cur_loc = np.setdiff1d(location.Country_Region.unique(),cur_loc.Country_Region.unique())
cur_loc = location.loc[location.Country_Region.isin(cur_loc)]
merge = pd.merge(cases_train, cur_loc, how='left', left_on=['country'], right_on=['Country_Region'])
merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
result = pd.concat([result, merge])
result.drop(['Lat', 'Long_','Province_State','Country_Region','Combined_Key'], axis=1,inplace=True)

index = list(set(cases_train.index) - set(merge.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)

# print('merge:',len(merge))
# print('case_train',len(cases_train))

merge = pd.merge(cases_train, location, how='left', left_on=['province'], right_on=['Province_State'])
merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
result = pd.concat([result, merge])
result.drop(['Lat', 'Long_','Province_State','Country_Region','Combined_Key'], axis=1,inplace=True)

index = list(set(cases_train.index) - set(merge.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)


# print('merge:',len(merge))
# print('case_train',len(cases_train))


freq_country = cases_train['country'].value_counts().to_frame()
freq_country.reset_index(inplace=True)

cur_loc = list(freq_country['index'])
cur_loc = location.loc[location['Country_Region'].isin(cur_loc)]
cur_loc = location.groupby('Country_Region').agg('mean')
cur_loc.reset_index(inplace=True)
cur_loc.drop(['Lat', 'Long_'], axis=1,inplace=True)
cur_loc['Case-Fatality_Ratio'] = (cur_loc.Deaths/cur_loc.Confirmed)*100


merge = pd.merge(cases_train, cur_loc, how='left', left_on=['country'], right_on=['Country_Region'])
merge.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], how='all',inplace=True)
result = pd.concat([result, merge])
result.drop(['Country_Region'], axis=1,inplace=True)

index = list(set(cases_train.index) - set(merge.index))
cases_train = cases_train.iloc[index]
cases_train.reset_index(inplace=True,drop=True)

# print('merge:',len(merge))
# print('case_train',len(cases_train))

cases_train['Confirmed'] = result.Confirmed.mean()
cases_train['Deaths'] = result.Deaths.mean()
cases_train['Recovered'] = result.Recovered.mean()
cases_train['Active'] = result.Active.mean()
cases_train['Incidence_Rate'] = result.Incidence_Rate.mean()
cases_train['Case-Fatality_Ratio'] = result['Case-Fatality_Ratio'].mean()

result = pd.concat([result, cases_train])


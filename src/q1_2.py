# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:25:16 2021

@author: wlian
"""

import pandas as pd

cases_train = pd.read_csv('../data/cases_train.csv')
cases_test = pd.read_csv('../data/cases_test.csv')
location = pd.read_csv('../data/location.csv')

freq_age = cases_train['age'].value_counts().to_frame()
freq_age.reset_index(inplace=True)

freq_prov = cases_train['province'].value_counts().to_frame()
freq_prov.reset_index(inplace=True)

freq_country = cases_train['country'].value_counts().to_frame()
freq_country.reset_index(inplace=True)

freq_date = cases_train['date_confirmation'].value_counts().to_frame()
freq_date.reset_index(inplace=True)
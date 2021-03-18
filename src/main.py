# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:17:51 2021

@author: wlian
"""

from q1_2 import q1_2
from q1_3 import q1_3
from q1_4 import q1_4
from q1_5 import q1_5

import pandas as pd
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    # q1_2()
    
    # q1_3()
    
    # q1_4()
    
    # q1_5("train")
    
    # q1_5("test")
    

    train = pd.read_csv(r'../results/cases_train_preprocessed.csv',dtype=object)
    test =  pd.read_csv(r'../results/cases_test_preprocessed.csv',dtype=object)
    df_tr, df_te = train_test_split(train, test_size=0.2, random_state = 1)
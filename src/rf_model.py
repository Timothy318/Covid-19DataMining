# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:19:00 2021

@author: wlian
"""

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, accuracy_score
from sklearn.metrics import f1_score
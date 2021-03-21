#KNN model evaluation for milestone 2 (2.3, 2.4)
#Bianca Blackwell 301304375
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

knn = pickle.load(open('knn_classifier.pkl', 'rb'))
y_te = pd.read_csv('y_te.csv',dtype = object)
x_te = pd.read_csv('x_te.csv')
print(y_te)
print(x_te)
prediction = knn.predict(x_te)
print(confusion_matrix(y_te,prediction))
print(classification_report(y_te,prediction))

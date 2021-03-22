#KNN model evaluation for milestone 2 (2.3, 2.4)
#Bianca Blackwell 301304375
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
# from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pickle

#2.3 "load the saved models from task 2.2, ..."
#assuming i'm not allowed to keep working in the same file then....
#I only see us uploading .main though, so this can be streamlined.
classifier = pickle.load(open('../models/knn_classifier.pkl', 'rb'))
y_te = pd.read_csv('./knn/y_te.csv',dtype = object)
x_te = pd.read_csv('./knn/x_te.csv')
y_tr = pd.read_csv('./knn/y_tr.csv',dtype = object)
x_tr = pd.read_csv('./knn/x_tr.csv')

print("Predicting on validation data with knn model, hang on, this might take a minute.")
prediction = classifier.predict(x_te)
print("Finished Prediction, calculating metrics.")
print(confusion_matrix(y_te,prediction))
print(classification_report(y_te,prediction))


print("Predicting on training data with knn model, hang on, this might take a minute.")
prediction = classifier.predict(x_tr)
print("Finished Prediction, calculating metrics.")
print(confusion_matrix(y_tr,prediction))
print(classification_report(y_tr,prediction))

#Graphs -> decided metrics were good, no graphs needed
# plot_decision_regions(x_te,y_te, clf=clf, legend = 2)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Knn with k = 5')
# plt.show()
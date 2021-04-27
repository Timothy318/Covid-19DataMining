# Cmpt459Project

## Milestone 1  
The code contains external package call basemap to generate the heatmap. Please install it if you wishes to see the output of the heatmap otherwise comment out any section containing such related code. There are other packages used as well and few of which require network connection. The stopword dictionary also needs to be download via ntlk as well, instructions will be prompt upon error for the download. The running time of the codes in total may take up to 5 min depending on the network connection, please wait patiently.  

## Milestone 2  
To execute any of our model without training, please download the pkl files and the preprocessed file in the google drive link. Put the preprocessed file into the data folder and the pkl files into the model folder. To execute a specific model, run the *_model.py/ipynb directly. Running the main.py will execute all training / evaluation for the catboost and knn and it will take a long time for knn (20 min on validation, 2hr on training). Model for random forest is done in juypter notebook and the results can be reproduced with the the run all cell button. Please also create the knn folder inside src folder if it is not there. Plot might not display correctly if the code is run from terminal but it will display correct in spyder. Please also install catboost package or any other required package indicated by any possible errors

## Milestone 3
There are three predictions files in the result folder but the one for submission is the one without any suffix append to it, which is equivalent to predictionCB.txt
Since there is no restriction on the location of the data, we have not included a folder. Please move it to the correct file directory upon error. 
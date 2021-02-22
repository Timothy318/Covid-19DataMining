import numpy as np 
import pandas as pd 	
import re

#import cleaned and transformed data
location_dat = pd.read_csv("results/location_transformed.csv")
dat_test = pd.read_csv("results/cases_test_preprocessed.csv")
dat_train = pd.read_csv("results/cases_train_preprocessed.csv")

#split location_dat based on rows with provinces, and rows without.
location_dat_prov = location_dat.loc[location_dat['Province_State'].notnull()]
location_dat_none = location_dat.loc[location_dat['Province_State'].isnull()]

#add a new column to case data, 'Key' to be the new combined key
dat_test['Key'] = dat_test['province'].str.cat(dat_test['country'], sep =", ")
#print(dat_test)
dat_train['Key'] = dat_train['province'].str.cat(dat_train['country'], sep =", ")
#print(dat_train)

#super ugly, but easier this way to just shove it all in, then remove in last step(slower though)
cols = ["age","sex","province","country","latitude","longitude","date_confirmation","additional_information",
	"source","outcome","age_range_ind","age_range","Key","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered",
	"Active","Combined_Key","Incidence_Rate","Case-Fatality_Ratio"]
join_test = pd.DataFrame(columns = cols)
#first, join on location_dat_prov (more specific) with Combined_Key
#if that fails, join on location_dat_none (with only country)
count = 0
count_good = 0 
for index,row in dat_test.iterrows():
	count = count +1
	if count > 46500: # comment this out, or change it to whatever the size of the dataset
	 	break
	#print("location Combined_Key: " + str(location_dat_prov['Combined_Key'].str.contains(row['Combined_Key'], flags=re.IGNORECASE, regex=False).any()))
	#print("row Combined_Key: " + str(row['Combined_Key']))
	#print("matching location Row: " + str(location_dat_prov[location_dat_prov['Combined_Key'] == row['Combined_Key']]))
	#matches province-specific location data
	if location_dat_prov['Combined_Key'].str.contains(row['Key'], flags=re.IGNORECASE, regex=False).any():
		#now, merge this row with the correct locations row, then put it into join_test!
		loc_row = location_dat_prov[location_dat_prov['Combined_Key'] == row['Key']].values.flatten().tolist()
		#print("Elements in loc_row: " + str(len(loc_row)))
		new_line = row.tolist() + loc_row
		#print("This is my new line: " + str(new_line) )
		#print("It has.. " + str(len(new_line)) + " elements")
		df = pd.DataFrame([new_line], columns = cols)
		join_test = join_test.append(df, ignore_index = True)
		#print(join_test)
		count_good = count_good + 1
		print(index)
		continue
	#matches province-non-specific location data
	elif location_dat_none['Combined_Key'].str.contains(row['country'], flags=re.IGNORECASE, regex=False).any():
		loc_row = location_dat_none[location_dat_none['Combined_Key'] == row['country']].values.flatten().tolist()
		#print("Elements in loc_row: " + str(len(loc_row)))
		#print(row.tolist())
		new_line = row.tolist() + loc_row
		#print("This is my new line: " + str(new_line) )
		#print("It has.. " + str(len(new_line)) + " elements")
		df = pd.DataFrame([new_line], columns = cols)
		join_test = join_test.append(df, ignore_index = True)
		count_good = count_good + 1
		print(index)
print("Finished Matching Test Data")
print("This is my count of matches out of " + str(len(dat_test.index)) + ": " + str(count_good))
#Remove the columns we dont want now (move this to the start so everything is faster)
remove_cols = ["province","country","latitude","longitude","Key"]
join_test = join_test.drop(columns = remove_cols)
join_test.to_csv("results/join_test.csv",index=False)




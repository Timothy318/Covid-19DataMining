import numpy as np
import pandas as pd

location_dat = pd.read_csv('location.csv')
bool_find = location_dat['Country_Region'].str.contains('US')
#quick cleaning, as I've noticed negative values in my active cases, which makes no sense. (you can't have negative covid)
bad_data = location_dat[["Confirmed","Deaths","Recovered","Active"]]
#bad_count = bad_data[bad_data < 0].count()
#print(bad_count)
location_dat = location_dat[(location_dat['Active'] >= 0) | (location_dat['Active'].isnull())]

#isolate US data
US_loc = location_dat.loc[bool_find]
US_inv = location_dat.loc[~bool_find]

#transformation on US_loc
#split US_loc by state
states = US_loc['Province_State'].unique().tolist()
#states.remove("Recovered")
#print(states)
states_df = dict(tuple(US_loc.groupby('Province_State')))

#all data frozen to 9/20/2020 4:22:56 AM
date = '9/20/2020 4:22:56 AM'
for loc in states:
	#mean of lat/long
	lat = states_df[loc]['Lat'].sum()/states_df[loc]['Lat'].size
	long_ = states_df[loc]['Long_'].sum()/states_df[loc]['Long_'].size
	#sum of confirmed cases
	confirmed = states_df[loc]['Confirmed'].sum()
	#sum of deaths
	deaths = states_df[loc]['Deaths'].sum()
	#sum of recovered
	recovered = states_df[loc]['Recovered'].sum()
	#sum of active
	active = states_df[loc]['Active'].sum()
	#sum incidence rate
	inc = states_df[loc]['Incidence_Rate'].sum()
	#new combined key
	combined_key = loc + ", United States"
	#re-calculate case-fataility (death/all) *100
	case_fat = (deaths/confirmed)*100
	#add row to US_inv DF, save to .csv
	new_line = pd.DataFrame([[loc,"United States",date,lat,long_,confirmed,deaths,recovered,active,combined_key,inc,case_fat]], columns = ["Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incidence_Rate","Case-Fatality_Ratio"])
	US_inv = pd.concat([US_inv, new_line], axis=0)

US_inv.to_csv("results/location_transformed.csv",index=False)
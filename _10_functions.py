import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from _12_LWE_function import *
from _13_GWPstar_functions import * 

#%% define how the string is constructed to name aircraft (a combination of aircraft name and year of datapoint)
def format_scenario_name(aircraft, year):
    # what this string looks like is arbitrary -- all that matters is that it's consistent
    string = str(aircraft)+', '+str(int(year))
    return string

#%% function to read LCIA files into individual dataframes per (combination of) scenario(s)
def read_LCIAs(file_name):
    df = pd.read_excel(file_name)
    # drop the nan row
    df = df.dropna(how='all')
    # remove line breaks from all cells in the DataFrame
    df = df.replace('\n', ' ', regex=True)
    
    #adjust reference flow labels and make them the indexes
    # read first column to list
    row_labels = df.iloc[:, 0].tolist()
    # modify the row labels
    modified_row_labels = []
    for label in row_labels:
        label = str(label)
        modified_label = label.split(" | ")[0].strip()
        modified_row_labels.append(modified_label)  
    # make the row_labels the new index
    df.index = modified_row_labels
    # drop the first column, since it's no longer needed
    df = df.iloc[:, 1:]
    
    # adjust the column labels (i.e., the impact categories) and split dataframe based on scenarios
    # Store the original column labels
    original_column_labels = df.columns[0:].tolist()
    # Remove unnamed, extract parts between commas and add number to duplicates
    filtered_list = []
    count_dict = {}
    
    for item in original_column_labels:
        if item.startswith("Unnamed"):
            continue
        part = item.split(",")[1].strip()
        # build in exception for "custom" LCIA methods
        if part == "custom": part = item.split("custom,",1)[1].strip()
        if part in count_dict:
            count_dict[part] += 1
            part += " " + str(count_dict[part])
        else:
            count_dict[part] = 0
        filtered_list.append(part)
    
    # Convert column labels to a range of integers
    df = df.set_axis(range(len(df.columns)), axis=1, copy=False)
    # Get the scenario names from the first row
    scenarios = df.iloc[0].astype(str).tolist()
    # Create a dictionary to store the separate dataframes
    separate_dataframes = {}
    # Prepare to build a list of all aircraft and years
    aircraft = []
    years = []
    
    # Iterate over the scenarios
    for scenario in scenarios:
        # Create a boolean mask to filter columns with the current scenario
        mask = df.iloc[0] == scenario
        # Filter the dataframe based on the mask
        df_here = df.loc[:, mask.tolist()].copy()
        # Get rid of first row
        df_here = df_here.iloc[1:]
        # Rename column labels with impact categories
        df_here.rename(columns=dict(zip(df_here.columns, filtered_list)), inplace=True)
        # If necessary, expand list of aircraft and years
        aircraft_here = scenario.split("'")[1].strip()
        year_here = scenario.split("'")[3].strip()
        year_here = float(year_here[-4:])
        if aircraft_here not in aircraft: aircraft.append(aircraft_here)
        if year_here not in years: years.append(year_here)
        # Add dataframe to dict
        separate_dataframes[format_scenario_name(aircraft_here,year_here)] = df_here
    
    return separate_dataframes, aircraft, years

#%% build full timeline of LCI(A) data based on start year and end year
def expand_LCIAs(file_name, y_start, y_stop):
    # extract scenarios into dict of individual scenarios
    dataframes_data, aircraft, years_data = read_LCIAs(file_name)
    # create a dictionary to store the yearly dataframes
    dataframes_yearly = {}
    for a in aircraft:
        # fill in dict starting from each year and moving up, skipping the final year
        for i in range(len(years_data)-1):
            data_string_here = format_scenario_name(a,years_data[i])
            data_string_next = format_scenario_name(a, years_data[i+1]) 
            data_here = dataframes_data[data_string_here]
            data_next = dataframes_data[data_string_next]
            year_here = years_data[i]
            # fill in gaps between data points with linear interpolation
            while year_here < years_data[i+1]:
                df_here = data_here + (data_next - data_here)*(year_here - years_data[i])/(years_data[i+1] - years_data[i])
                year_string = format_scenario_name(a, year_here)
                dataframes_yearly[year_string] = df_here
                year_here += 1
        # finally, add the data for the last year in the dataset
        year_string = format_scenario_name(a, year_here)  
        dataframes_yearly[year_string] = data_next
        
    # change the dictionary of yearly dataframes to match the timeline of y_start and y_stop
    dataframes_timeline = {}
    for a in aircraft:
        i = y_start
        # account for the possibility that timeline starts before first data point
        while i <= years_data[0]:
            string_here = format_scenario_name(a, i)
            string_data = format_scenario_name(a, years_data[0])
            dataframes_timeline[string_here] = dataframes_yearly[string_data]
            i += 1
        # duplicating dictionary made earlier to timeline dictionary
        while i <= years_data[-1]:
            string_here = format_scenario_name(a, i)
            dataframes_timeline[string_here] = dataframes_yearly[string_here]
            i += 1
        # account for possibility that timeline ends after last data point
        while i <= y_stop:
            string_here = format_scenario_name(a, i)
            string_data = format_scenario_name(a, years_data[-1])      
            dataframes_timeline[string_here] = dataframes_yearly[string_data]
            i += 1
            
    return dataframes_timeline, aircraft

#%% function used in the creation of dataframes for plant/market processes
def plant_years(y_start, y_end):
    return y_start - 30, y_end

#%% to align data formats (and as placeholder, in case construction data of plants becomes time-dependent), extend plants_dataframes_yearly
def reformat_yearly_data(dataframes_temp, y_start, y_end, cases, string):
    y_plants_start, y_plants_end = plant_years(y_start, y_end)
    cases = []
    for i in range(y_plants_start, y_plants_end + 1):
        cases.append(str(i)+' '+string)
    dataframes_timeline = {}
    for i in cases:
        # fill in dict starting from each year and moving up
        for key, df_here in dataframes_temp.items():
            year_here = float(key.split(",")[1].strip())
            year_string = format_scenario_name(i, year_here)
            dataframes_timeline[year_string] = df_here
    
    return dataframes_timeline, cases

#%% function to assist in loading files from file names
def use_file_names(y_start, y_end, aircraft_file, wind_file, grid_file, market_file):
    
    aircraft_dataframes_yearly, aircraft_names = expand_LCIAs(aircraft_file, y_start, y_end)
    plant_dataframes_yearly_temp_wind, plants_wind = expand_LCIAs(wind_file, y_start, y_end)
    plant_dataframes_yearly_temp_grid, plants_grid = expand_LCIAs(grid_file, y_start, y_end)
    market_dataframes_yearly_temp, markets = expand_LCIAs(market_file, y_start, y_end)
    
    plant_dataframes_yearly_wind, plants = reformat_yearly_data(plant_dataframes_yearly_temp_wind, y_start, y_end, plants_wind, 'plants')
    plant_dataframes_yearly_grid, plants = reformat_yearly_data(plant_dataframes_yearly_temp_grid, y_start, y_end, plants_grid, 'plants')
    market_dataframes_yearly, markets = reformat_yearly_data(market_dataframes_yearly_temp, y_start, y_end, markets, 'plants')
    
    return aircraft_names, aircraft_dataframes_yearly, plants, plant_dataframes_yearly_wind, plant_dataframes_yearly_grid, market_dataframes_yearly

#%% flight builder: time series of RPK based on growth factor
def build_flights(y_start, y_stop, rpk_0, growth):
    flights = []
    for i in range(y_stop-y_start+1): flights.append(rpk_0*(1+growth)**(i+1))
    return flights

#%% occupation builder: make list of same length as RPK which embodies the ratio of RPK/ASK
# current implementation is that occupation changes linearly over time. could also be implemented as a different function.
def build_occupation(occupation_y0, y0, occupation_y1, y1, occupation_y2, y2):
    # use case will typically be that y0 and y2 are the start and end years, respectively
    occupation = []
    for i in range(y1-y0+1):
        occupation.append(occupation_y0 + i*(occupation_y1-occupation_y0)/(y1-y0))
    for i in range(y2-y1):
        i = i + 1
        occupation.append(occupation_y1 + i*(occupation_y2-occupation_y1)/(y2-y1))
    
    return occupation

#%% entry list builder: determines for a given range of years what aircraft enters service
def entry_list(aircraft_char, y_start, y_stop, h2_share):
    entries = []
    for t in range(y_stop-y_start):
        y_here = t + y_start # from t (position in time series) to y_here (current year)
        req_eis = aircraft_char.loc['EIS'] <= y_here # filter out aircraft not entering service yet
        # filter aircraft into fuel types + only keep most recent model
        req_hc = aircraft_char.loc['fuel type'] == 'hydrocarbon'
        chosen_hc = aircraft_char.loc['EIS'] == max(req_eis*req_hc*aircraft_char.loc['EIS'])
        chosen_hc = chosen_hc*req_hc    
        req_h2 = aircraft_char.loc['fuel type'] == 'hydrogen'
        chosen_h2 = aircraft_char.loc['EIS'] == max(req_eis*req_h2*aircraft_char.loc['EIS'])
        chosen_h2 = chosen_h2*req_h2
        # create list of most recent model(s), taking into account split between HC and H2 (based on RPK)
        if sum(chosen_h2) > 0:
            entries.append(h2_share*chosen_h2 + (1 - h2_share)*chosen_hc)
        else: entries.append(1*chosen_hc)
        
    entries = pd.DataFrame(entries)
    
    return entries

#%% fleet builder: build fleet in terms of number and types of aircraft servicing a particular RPK time series
def build_fleet(aircraft_char, max_age_0, ages_0, rpk, occupation, y_start, h2_share):
    entries_0 = entry_list(aircraft_char, y_start - max_age_0, y_start, h2_share)   # list of aircraft (per age) in the starting fleet
    entries_t = entry_list(aircraft_char, y_start, y_start + len(rpk), h2_share) # list of which aircraft comes into the fleet, starting from 
    fleet = []          # timeline list where each entry is the number of aircraft of each generation in the fleet (e.g.: [100, 10, 0] shows that there are three aircraft generations, with a mix of 1st and 2nd generation in the fleet that year)
    inflow = []         # timeline list like fleet, but only showing aircraft entering service
    outflow = []        # timeline list like fleet, but only showing aircraft being decomissioned
    fleet_age = pd.DataFrame(columns = aircraft_char.columns) # summation of the dataframe is equal to entries of the fleet list. distributed by age to enable fleet renewal modeling.
    # start by building the fleet as we expect it in the starting year (based on demand)
    fleet_use = pd.DataFrame(columns = aircraft_char.columns) # temporary dataframe to characterise use of fleet to meet RPK
    for i in range(max_age_0): 
        fleet_use.loc[-1] = [0]*len(aircraft_char.columns) # adding a row
        fleet_use.index = fleet_use.index + 1  # shifting index
        fleet_use.sort_index(inplace=True)  # more recent aircraft now has lowest index (= lowest age)
        fleet_use.loc[0] = ages_0[i]*aircraft_char.loc['seats']*aircraft_char.loc['yearly distance']*occupation[0]*entries_0.loc[i]
    
    x_0 = rpk[0]/fleet_use.sum().sum() # this is the factor with which the initial fleet will have to be scaled, after having built a preliminary fleet above, based on the initial age distribution
    
    for i in range(max_age_0): 
        fleet_age.loc[-1] = [0]*len(aircraft_char.columns) # adding a row
        fleet_age.index = fleet_age.index + 1  # shifting index
        fleet_age.sort_index(inplace=True) # more recent aircraft now has lowest index (= lowest age)
        fleet_age.loc[0] = ages_0[i]*x_0*entries_0.loc[i]
        
    fleet.append(fleet_age.sum())
    
    # create initial values for inflow and outflow (for outflow: assumption that decommissioning of year 1 is equal to year 0)
    inflow.append(fleet_age.loc[0])
    outflow.append(fleet_age.loc[max_age_0-1])
    
    # create fleet for subsequent years, again based on what would be needed to meet RPK
    for t in range(len(rpk)-1):
        t = t+1 # we already created the first year (start year) above with entries_0 and ages_0
        rpk_here = rpk[t]
        fleet_age.index = fleet_age.index + 1 # all aircraft become 1 year older
        outflow.append(fleet_age.loc[max_age_0]) # add aircraft about to be decommissioned to outflow list
        fleet_age.drop(max_age_0, inplace=True) # get rid of old aircraft
        rpk_before = fleet_age*aircraft_char.loc['seats']*aircraft_char.loc['yearly distance']*occupation[t]
        rpk_before = rpk_before.sum().sum()
        fleet_age.loc[0] = [0]*len(aircraft_char.columns) # add a row for any potential new aircraft
        if rpk_before < rpk_here: # add new aircraft to the fleet to meet RPK
            fleet_age.loc[0] = (rpk_here - rpk_before)*entries_t.iloc[t]/(aircraft_char.loc['seats']*aircraft_char.loc['yearly distance']*occupation[t])
        inflow.append(fleet_age.loc[0]) # add aircraft entering service to inflow list
        fleet_age.sort_index(inplace=True) # sort dataframe to have ages chronological (not strictly required, but handy)
        fleet.append(fleet_age.sum())
        
    # now, fleet, inflow and outflow are set up as a list of series. convert these to dataframes, which are much more convenient.
    fleet = pd.DataFrame(fleet)
    fleet.index = list(range(len(rpk)))
    inflow = pd.DataFrame(inflow)
    inflow.index = list(range(len(rpk)))    
    outflow = pd.DataFrame(outflow)
    outflow.index = list(range(len(rpk)))    
        
    return fleet, inflow, outflow

#%% allocate time-specific LCIA values to a timeline-based dataframe specifying economic flows
def allocate_LCIA(y_start, process, process_timeline, input_dict):
    LCIA_dict = {}
    for i in range(len(process_timeline)):
        year_here = y_start + i
        process_quant = process_timeline.iloc[i]
        process_result = []
        for j in range(len(process_timeline.columns)):
            item = process_timeline.columns[j]
            process_data = input_dict[format_scenario_name(item,year_here)].loc[process] # get the LCIA data from the input_dict for the right aircraft and time
            process_result.append(process_quant[j]*process_data)
        process_dataframe = pd.DataFrame(process_result)
        process_dataframe.index = process_timeline.columns
        LCIA_dict[format_scenario_name(process,year_here)] = process_dataframe
    return LCIA_dict

# %%fleet user: takes the fleet timeline df and translates it into fuel use dfs, divided into fuel used during LTO and during CCD 
def use_fleet(fleet, flight_distance, rpk, occupation, aircraft_char, j):

    yearly_distance = aircraft_char.loc['yearly distance']
    LTO_fuel = aircraft_char.loc['LTO fuel']
    CCD_fuel = aircraft_char.loc['reference flight CCD fuel'].apply(lambda x: x[j]) # make sure that the right CCD fuel value is taken for these flights (based on i from previous function)
    
    CCD_fuel_per_km = CCD_fuel/flight_distance # including this here now -- potentially this section of the model can be made more sophisticated
    
    LTO_fuel_timeline = [] # timeline list where each entry shows how much fuel (in MJ) is used for LTO per aircraft
    CCD_fuel_timeline = [] # timeline list where each entry shows how much fuel (in MJ) is used for CCD per aircraft
    for i in fleet.index:
        rpk_capacity = fleet.loc[i]*aircraft_char.loc['yearly distance']*aircraft_char.loc['seats']*occupation[i]
        rpk_demand = rpk[i]
        rpk_adjustment = rpk_demand/(rpk_capacity.sum())
        flights_here = fleet.loc[i]*yearly_distance*rpk_adjustment/flight_distance # number of flights each aircraft type takes, based on usage estimates
        fuel_LTO_here = flights_here*LTO_fuel # MJ fuel for LTO per aircraft type in this year
        fuel_CCD_here = flights_here*flight_distance*CCD_fuel_per_km # MJ fuel for CCD per aircraft type in this year
        LTO_fuel_timeline.append(fuel_LTO_here)
        CCD_fuel_timeline.append(fuel_CCD_here)
    
    LTO_fuel_timeline = pd.DataFrame(LTO_fuel_timeline)
    LTO_fuel_timeline.index = fleet.index
    CCD_fuel_timeline = pd.DataFrame(CCD_fuel_timeline)
    CCD_fuel_timeline.index = fleet.index
    
    return LTO_fuel_timeline, CCD_fuel_timeline

#%% AAF timeline builder: turns a dataframe of SAF milestones into a year-by-year timeline
# note that there is a possible edge case not accounted for, which is if aaf_0 falls after a goal_year but does not meet the goal for that year.
def aaf_share_list(y_start, y_stop, aaf_0, aaf_milestones, aircraft_char, LTO_fuel, CCD_fuel, no_decreases):
    aaf_goal_timeline = [aaf_0] # prime timeline, starting with share in initial year
    year_next = y_start + 1 # the year following, for which we determine what the share should be to be on track with the milestones
    aaf_now = aaf_0 # the share in the year previous to "year_next"
    for index, row in aaf_milestones.iterrows():
        goal_year = row.loc['year']
        goal_share = row.loc['AAF share']
        while year_next <= goal_year:
            if aaf_now < goal_share: 
                aaf_next = aaf_now + (goal_share - aaf_now)/(goal_year - year_next + 1)
            else: aaf_next = aaf_now
            aaf_goal_timeline.append(aaf_next)
            aaf_now = aaf_next
            year_next += 1
    # in the case that the end year reaches beyond the timeline of goals, repeat the final goal to match length
    while year_next <= y_stop:
        aaf_goal_timeline.append(aaf_next)
        year_next += 1
    # in the case that the end year is reached before the timeline of goals, shorten the timeline list
    if year_next > y_stop: aaf_goal_timeline = aaf_goal_timeline[:y_stop - y_start + 1]
    
    # next part of the function: based on the amount of hydrogen planes flying, determine what the share of hydrogen is in the fuel mix 
    aircraft_char = aircraft_char.T
    hc_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrocarbon'])].index
    h2_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrogen'])].index
    hydrogen_share = []
    year_here = y_start
    for i in LTO_fuel.index:
        hydrogen_share_here = (LTO_fuel.loc[i,h2_aircraft].sum() + CCD_fuel.loc[i,h2_aircraft].sum())/(LTO_fuel.loc[i, :].sum() + CCD_fuel.loc[i, :].sum())
        hydrogen_share.append(hydrogen_share_here)
    
    # final part of the function: determine how much of the hydrocarbon fuel share should be SAF in order to meet the AAF share goal timeline
    missing_share = np.subtract(aaf_goal_timeline, hydrogen_share)
    missing_share = np.array([0 if i < 0 else i for i in missing_share])
    notice = [] # list of years to print in case no_decreases is False and SAF share decreases
    for i in range(len(missing_share) - 1):
        if missing_share[i] > missing_share[i+1]:
            if no_decreases == True:
                missing_share[i+1] = missing_share[i]
            else: notice.append(y_start+i+1)
    
    saf_share = np.array([missing_share[i]*1/(1- hydrogen_share[i]) for i in range(len(missing_share))])
    saf_share[np.isnan(saf_share)] = 0
    
    if len(notice) != 0: logging.info(f"Instance(s) of SAF share in total fuel decreasing dectected in year(s): {notice}")
    
    return aaf_goal_timeline, hydrogen_share, saf_share

#%% function that takes the fuel shares and total fuel use as inputs and creates seperate dataframes for each fuel
def fuel_quantities(aircraft_char, fuel, hydrogen_share, saf_share):
    aircraft_char = aircraft_char.T
    hc_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrocarbon'])].index
    h2_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrogen'])].index
    
    fuel_fossil = []
    fuel_saf = []
    fuel_hydrogen = []
    for i in range(len(fuel)):
        fuel_fossil.append(fuel.loc[i,hc_aircraft]*(1 - saf_share[i]))
        fuel_saf.append(fuel.loc[i,hc_aircraft]*saf_share[i])
        fuel_hydrogen.append(fuel.loc[i,h2_aircraft])
        
    fuel_fossil = pd.DataFrame(fuel_fossil)
    fuel_fossil.index = fuel.index
    fuel_saf = pd.DataFrame(fuel_saf)
    fuel_saf.index = fuel.index
    fuel_hydrogen = pd.DataFrame(fuel_hydrogen)
    fuel_hydrogen.index = fuel.index
    
    return fuel_fossil, fuel_saf, fuel_hydrogen

#%% like entry_list, but for fuel plants
def fuel_entry_list(plant_char, y_start, y_stop):
    entries = []
    for t in range(y_stop-y_start):
        y_here = t + y_start # from t (position in time series) to y_here (current year)
        req_eis = plant_char.loc['EIS'] <= y_here # filter out aircraft not entering service yet
        chosen_plant = plant_char.loc['EIS'] == max(req_eis*plant_char.loc['EIS'])
        # create list of most recent model(s), taking into account split between HC and H2 (based on RPK)
        entries.append(1*chosen_plant)
        
    entries = pd.DataFrame(entries)
    entries.index = list(range(y_stop-y_start))
    return entries

#%% based on the fuels needed to power aircraft, determine what the fuel production fleets look like
def fuel_fleet_builder(plant_char, ages_0, max_age, total_fuel, y_start, y_stop, production_index):
    entries_0 = fuel_entry_list(plant_char, y_start - max_age, y_start)   # list of plants (per age) in the starting fleet
    entries_t = fuel_entry_list(plant_char, y_start, y_start + len(total_fuel)) # list of which plants comes into the fleet
    fleet = []          
    inflow = []         
    outflow = []
    fleet_age = pd.DataFrame(columns = plant_char.columns) # summation of the dataframe is equal to entries of the fleet list. distributed by age to enable fleet renewal modeling.
    # start by building the fleet as we expect it in the starting year (based on demand)
    fleet_use = pd.DataFrame(columns = plant_char.columns) # temporary dataframe to characterise use of fleet to meet MJ H2 demand
    for i in range(max_age): 
        fleet_use.loc[-1] = [0]*len(plant_char.columns) # adding a row
        fleet_use.index = fleet_use.index + 1  # shifting index
        fleet_use.sort_index(inplace=True)  # more recent aircraft now has lowest index (= lowest age)
        fleet_use.loc[0] = ages_0[i]*plant_char.loc[production_index]*entries_0.loc[i]
    
    x_0 = total_fuel[0]/fleet_use.sum().sum() # this is the factor with which the initial fleet will have to be scaled, after having built a preliminary fleet above, based on the initial age distribution
    
    for i in range(max_age): 
        fleet_age.loc[-1] = [0]*len(plant_char.columns) # adding a row
        fleet_age.index = fleet_age.index + 1  # shifting index
        fleet_age.sort_index(inplace=True) # more recent aircraft now has lowest index (= lowest age)
        fleet_age.loc[0] = ages_0[i]*x_0*entries_0.loc[i]
    
    fleet.append(fleet_age.sum())
    
    # create initial values for inflow and outflow (for outflow: assumption that decommissioning of year 1 is equal to year 0)
    inflow.append(fleet_age.loc[0])
    outflow.append(fleet_age.loc[max_age-1])
    
    # create fleet for subsequent years, again based on what would be needed to meet MJ H2 demand
    for t in range(len(total_fuel)-1):
        t = t+1 # we already created the first year (start year) above with entries_0 and ages_0
        mj_here = total_fuel[t]
        fleet_age.index = fleet_age.index + 1 # all plants become 1 year older
        outflow.append(fleet_age.loc[max_age]) # add plants about to be decommissioned to outflow list
        fleet_age.drop(max_age, inplace=True) # get rid of old plants
        mj_before = fleet_age*plant_char.loc[production_index]
        mj_before = mj_before.sum().sum()
        fleet_age.loc[0] = [0]*len(plant_char.columns) # add a row for any potential new plants
        if mj_before < mj_here: # add new plants to the fleet to meet MJ H2 demand
            fleet_age.loc[0] = (mj_here - mj_before)*entries_t.iloc[t]/plant_char.loc[production_index]
        inflow.append(fleet_age.loc[0]) # add plants entering service to inflow list
        fleet_age.sort_index(inplace=True) # sort dataframe to have ages chronological (not strictly required, but handy)
        fleet.append(fleet_age.sum())
        
    # now, fleet, inflow and outflow are set up as a list of series. convert these to dataframes, which are much more convenient.
    fleet = pd.DataFrame(fleet)
    fleet.index = list(range(len(total_fuel)))
    inflow = pd.DataFrame(inflow)
    inflow.index = list(range(len(total_fuel)))    
    outflow = pd.DataFrame(outflow)
    outflow.index = list(range(len(total_fuel)))    
    
    return fleet, inflow, outflow

#%% function to isolate one impact category from dictionary of results
def impact_cat_result(LCIA_result, impact_cat):
    result = []
    for key, df in LCIA_result.items(): result.append(df[impact_cat].sum())
    return np.array(result)

#%% cirrus calculator: calculate how much cirrus is produced
# expressed in fossil cirrus km-eq, so relative to the RF which would be caused by aircraft flying on fossil kerosene
def cirrus_calc(hydrogen_share, saf_share, fleet, aircraft_char, cirrus_rf_change_h2, no_aaf_change=False):
    h2_fossil = 13.73 # hydrogen content of fossil kerosene
    h2_saf = 15.29 # hydrogen content of SAF
    # estimate of ice particle number based on fuel mix
    mix = np.array(saf_share)
    h2_mix = h2_fossil*(1-mix) + h2_saf*mix
    ice_particles = 9.407e23*np.e**(np.log(0.2475)*h2_mix)
    # based on estimated change in ice particles, estimate change in cirrus RF
    ice_particles_change = 1 - ice_particles/ice_particles[0]
    cirrus_rf_change = 1 - 0.048*19.2**ice_particles_change
    distance_total = fleet*aircraft_char.loc['yearly distance']
    # figure out how much the km-eq value is based on distance of hydrocarbon aircraft
    aircraft_char = aircraft_char.T
    hc_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrocarbon'])].index
    h2_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrogen'])].index
    if no_aaf_change == False:
        cirrus_eq = []
        for i in range(len(distance_total)):
            cirrus_here = distance_total.iloc[i].loc[hc_aircraft].sum()*cirrus_rf_change[i] + distance_total.iloc[i].loc[h2_aircraft].sum()*cirrus_rf_change_h2
            cirrus_eq.append(cirrus_here)
    # sensitivity analysis which assumes that AAF does not influence cirrus impacts
    if no_aaf_change == True:
        cirrus_eq = []
        for i in range(len(distance_total)):
            cirrus_here = distance_total.iloc[i].sum()
            cirrus_eq.append(cirrus_here)

    return cirrus_eq

def calc_cirrus_change(saf_share, fleet, aircraft_char, cirrus_rf_change_h2, no_aaf_change=False):
    if no_aaf_change == True: cirrus_change = pd.Series([1]*len(fleet))
    else: 
        h2_fossil = 13.73 # hydrogen content of fossil kerosene
        h2_saf = 15.29 # hydrogen content of SAF
        # estimate of ice particle number based on fuel mix
        mix = np.array(saf_share)
        h2_mix = h2_fossil*(1-mix) + h2_saf*mix
        ice_particles = 9.407e23*np.e**(np.log(0.2475)*h2_mix)
        # based on estimated change in ice particles, estimate change in cirrus RF
        ice_particles_change = 1 - ice_particles/ice_particles[0]
        cirrus_rf_change = 1 - 0.048*19.2**ice_particles_change
        distance_total = fleet*aircraft_char.loc['yearly distance']
        # figure out how much the km-eq value is based on distance of hydrocarbon aircraft
        aircraft_char = aircraft_char.T
        hc_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrocarbon'])].index
        h2_aircraft = aircraft_char.loc[aircraft_char['fuel type'].isin(['hydrogen'])].index
        hc_distance = distance_total[hc_aircraft].sum(axis=1)
        h2_distance = distance_total[h2_aircraft].sum(axis=1)
        cirrus_change = (
            hc_distance.values * cirrus_rf_change + 
            h2_distance.values * cirrus_rf_change_h2
        ) / distance_total.sum(axis=1).values

    return pd.Series(cirrus_change)

def perform_GWPstar_calc(flight_start_year, flight_end_year, lwe_input, distance_total, fuel_effect_correction_contrails):
    # prepare objects
    params = ModelParameters(flight_start_year, flight_end_year)
    ERF_co2 = SimplifiedERFCo2(parameters=params)
    ERF_nox = ERFNox(parameters=params)
    ERF_other = ERFOthers(parameters=params)
    ERF_extra = ERFExtra(parameters=params)
    years = ERF_co2.years

    # extract emissions, including contrail fuel correction -- note that emissions should be in Tg, except for contrails (km)
    co2_emissions = pd.Series((lwe_input.iloc[:, 12].values + lwe_input.iloc[:, 13].values)* 1e-9, index=years)
    nox_emissions = pd.Series(lwe_input.iloc[:, 14].values * 1e-9, index=years)
    soot_emissions = pd.Series(lwe_input.iloc[:, 15].values * 1e-9, index=years)
    h2o_emissions = pd.Series(lwe_input.iloc[:, 17].values * 1e3 * 1e-9, index=years) # from m3 to Tg
    sulfur_emissions = pd.Series(lwe_input.iloc[:, 16].values * 1e-9, index=years)
    ch4_emissions = pd.Series(lwe_input.iloc[:, 5].values * 1e-9, index=years)
    h2_emissions = pd.Series(lwe_input.iloc[:, 2].values * 1e-9, index=years)

    total_aircraft_distance = pd.Series(distance_total.values, index=years)
    operations_contrails_gain =pd.Series([0]*len(lwe_input), index=years)
    fuel_effect_correction_contrails = pd.Series(fuel_effect_correction_contrails.values, index=years)

    # calculate ERF for emissions
    annual_co2_erf, co2_erf = ERF_co2.compute(co2_emissions)
    nox_short_term_o3_increase_erf, nox_long_term_o3_decrease_erf, nox_ch4_decrease_erf, nox_stratospheric_water_vapor_decrease_erf, nox_erf = ERF_nox.compute(nox_emissions, erf_coefficient_nox_short_term_o3_increase, erf_coefficient_nox_long_term_o3_decrease, erf_coefficient_nox_ch4_decrease, erf_coefficient_nox_stratospheric_water_vapor_decrease)
    contrails_erf, soot_erf, h2o_erf, sulfur_erf, aerosol_erf = ERF_other.compute(soot_emissions, h2o_emissions, sulfur_emissions, erf_coefficient_contrails, erf_coefficient_soot, erf_coefficient_h2o, erf_coefficient_sulfur, total_aircraft_distance, operations_contrails_gain, fuel_effect_correction_contrails)
    ch4_erf, h2_erf = ERF_extra.compute(ch4_emissions, h2_emissions, erf_coefficient_ch4, erf_coefficient_h2)

    # combine results for next step
    df_climate = pd.concat([ERF_co2.df_climate, ERF_nox.df_climate, ERF_other.df_climate, ERF_extra.df_climate], axis=1)
    total_erf = co2_erf + contrails_erf + h2o_erf + nox_erf + soot_erf + sulfur_erf + ch4_erf + h2_erf
    df_climate["total_erf"] = total_erf
    climate_calc = TemperatureGWPStar(parameters=params)
    climate_calc.df_climate = df_climate

    # calculate temperature change
    climate_calc_results = climate_calc.compute(
        contrails_gwpstar_variation_duration,
        contrails_gwpstar_s_coefficient,
        nox_short_term_o3_increase_gwpstar_variation_duration,
        nox_short_term_o3_increase_gwpstar_s_coefficient,
        nox_long_term_o3_decrease_gwpstar_variation_duration,
        nox_long_term_o3_decrease_gwpstar_s_coefficient,
        nox_ch4_decrease_gwpstar_variation_duration,
        nox_ch4_decrease_gwpstar_s_coefficient,
        nox_stratospheric_water_vapor_decrease_gwpstar_variation_duration,
        nox_stratospheric_water_vapor_decrease_gwpstar_s_coefficient,
        soot_gwpstar_variation_duration,
        soot_gwpstar_s_coefficient,
        h2o_gwpstar_variation_duration,
        h2o_gwpstar_s_coefficient,
        sulfur_gwpstar_variation_duration,
        sulfur_gwpstar_s_coefficient,
        contrails_erf,
        nox_short_term_o3_increase_erf,
        nox_long_term_o3_decrease_erf,
        nox_ch4_decrease_erf,
        nox_stratospheric_water_vapor_decrease_erf,
        soot_erf,
        h2o_erf,
        sulfur_erf,
        co2_erf,
        total_erf,
        co2_emissions,
        tcre_coefficient,
        ch4_gwpstar_variation_duration,
        ch4_gwpstar_s_coefficient,
        ch4_erf,
        h2_gwpstar_variation_duration,
        h2_gwpstar_s_coefficient,
        h2_erf,
    )

    return climate_calc.df_climate

#%% function to turn a list of dictionaries (as created by quantifying a number of activities) into a single dataframe of years and impact categories
def total_LCIA(list_of_dic):
    # prime an array with the right dimensions, matching number of impact categories and number of years
    row_num = len(list_of_dic[0].keys())
    col_num = len(list_of_dic[0][list(list_of_dic[0].keys())[0]].columns)
    LCIA_array = np.zeros([row_num, col_num])
    
    # cycle through list of dictionaries
    for i in range(len(list_of_dic)):
        # cycle through each dictionary
        loc = 0
        for key, df in list_of_dic[i].items():
            LCIA_array[loc] = LCIA_array[loc] + np.array(df.apply(np.sum, axis=0))
            loc = loc + 1
    
    LCIA_df = pd.DataFrame(LCIA_array, columns = list_of_dic[0][list(list_of_dic[0].keys())[0]].columns)
    
    return LCIA_df

#%% function to turn a list of dictionaries (as created by quantifying a number of activities) into a single dictionary
def list_of_dic_to_dic(list_of_dic):
    # prime an array with the right dimensions, matching number of impact categories and number of years
    row_num = len(list_of_dic[0][list(list_of_dic[0].keys())[0]].index)
    col_num = len(list_of_dic[0][list(list_of_dic[0].keys())[0]].columns)
    LCIA_array = np.zeros([row_num, col_num])
    
    # cycle through list of dictionaries
    for i in range(len(list_of_dic)):
        # cycle through each dictionary
        loc = 0
        for key, df in list_of_dic[i].items():
            LCIA_array[loc] = LCIA_array[loc] + np.array(df.apply(np.sum, axis=0))
            loc = loc + 1
    
    LCIA_dic = {}
    loc = 0
    for key, df in list_of_dic[0].items():
        key_here = key.split(",")[-1].strip()
        LCIA_dic[key_here] = pd.DataFrame([LCIA_array[loc]], columns = list_of_dic[0][list(list_of_dic[0].keys())[0]].columns)
        loc = loc + 1
    
    return LCIA_dic

#%% similar function to aaf_share_list, but more simple: calculates consumption of process based on linear change across a number of years
def consumption_list(performance_milestones, y_start, y_stop, performance_0):
    year_next = y_start
    consumption_now = performance_0 # the share in the year previous to "year_next"
    consumption_list = []
    for index, row in performance_milestones.iterrows():
        goal_year = row.loc['year']
        goal_consumption = row.iloc[1] # use iloc to account for changing index based on application
        while year_next <= goal_year:
            consumption_next = consumption_now + (goal_consumption - consumption_now)/(goal_year - year_next + 1)
            consumption_list.append(consumption_next)
            consumption_now = consumption_next
            year_next += 1
    # in the case that the end year reaches beyond the timeline of goals, repeat the final goal to match length
    while year_next <= y_stop:
        consumption_list.append(consumption_next)
        year_next += 1
    # in the case that the end year is reached before the timeline of goals, shorten the timeline list
    if year_next > y_stop: consumption_list = consumption_list[:y_stop - y_start + 1]
    
    return np.array(consumption_list)

#%% function to use a fleet to quantify a generic variable (comparable to use_fleet function for aircraft) IMPORTANT: make sure units are aligned in the plant_char variable!
def generic_fleet_use(plant_char, consumption_index, production_index, fleet, total_demand = []):
    consumption = plant_char.loc[consumption_index]
    production = plant_char.loc[production_index]
    
    consumption_timeline = [] # timeline list where each entry shows how much is used
    for i in fleet.index:
        if list(total_demand) == []: # i.e., no total_demand is given -- we do not care about the demand, only about the capacity
            production_here = fleet.loc[i]*production # quantity produced in total, divided over plant fleet
            consumption_here = production_here*consumption # quantity required for production
            consumption_timeline.append(consumption_here)
        else:
            capacity_here = fleet.loc[i]*production
            demand_here = total_demand[i]
            capacity_adjustment = demand_here/capacity_here.sum()
            consumption_here = capacity_here*consumption*capacity_adjustment
            consumption_timeline.append(consumption_here)
    
    consumption_timeline = pd.DataFrame(consumption_timeline)
    consumption_timeline.index = fleet.index
    
    return consumption_timeline

#%% function to more easily calculate a large amount of LCIA results based on timeline dataframes and to combine them into one dataframe
def calculate_LCIAs_from_list(flow_list, label_list, dataframes_yearly, y_start):
    flow_results = [] # start of a list of dictionaries for each flow
    for i in range(len(flow_list)):
        flow_results.append(allocate_LCIA(y_start, label_list[i], flow_list[i], dataframes_yearly))
    
    return flow_results

#%% establish characteristics of fuel plants and use them based on fuel demand
def define_and_use_fuel_plants(y_start, y_end, y_plants_start, y_plants_end, plants, hydrogen_method, total_hydrogen, total_saf, PEM_elec_progression, PEM_elec_0, PEM_occupation, PEM_water, PEM_h2_escaped, DAC_elec_progression, DAC_elec_0, DAC_sorbent_progression, DAC_sorbent_0, FT_elec, FT_co2, FT_h2, FT_co2_emissions, LIQ_elec_progression, LTQ_elec_0, LIQ_h2_escaped, plant_dataframes_yearly, process_dataframes_yearly):
    # variable used for utility
    h2_energy_density = 120 # MJ/kg
    
    # establish characteristics of PEM plant
    PEM_plant_attributes = ['EIS', 'max age', 'occupation', 'electricity: electrolysis', 'water consumption', 'electricity: total', 'hydrogen escaped to air', 'yearly production']
    PEM_EIS = list(range(y_plants_start, y_plants_end + 1)) # entry into service of each plant [year]
    PEM_max_age = 20 # how long each plant is assumed to operate [years]
    PEM_ages_0 = [0]*(PEM_max_age - 1) + [1] # distribution in number of plants at start of analysis; note: len(ages_0) MUST be equal to max_age!
    PEM_production = np.array([148216*h2_energy_density]*(y_plants_end - y_plants_start + 1)) # production per year of each plant [MJH2/year]
    PEM_elec = consumption_list(PEM_elec_progression, y_plants_start, y_plants_end, PEM_elec_0) # electricity consumption per MJ of H2 producted [kWh/MJH2]
    PEM_elec_total = PEM_elec + 3.2/h2_energy_density # account for 3.2 kWh/kg additional electricity to compress H2 before transport
    PEM_prod_net = PEM_production*(1 - PEM_h2_escaped) # the H2 that can be obtained from a plant per year when accounting for transportation losses
    PEM_plant_char = pd.DataFrame([PEM_EIS, 
                    [PEM_max_age]*(y_plants_end - y_plants_start + 1), 
                    PEM_occupation/PEM_prod_net, 
                    PEM_elec/PEM_prod_net*PEM_production,
                    PEM_water/PEM_prod_net*PEM_production,
                    PEM_elec_total/PEM_prod_net*PEM_production,
                    PEM_h2_escaped/h2_energy_density,
                    PEM_prod_net],
                    index = PEM_plant_attributes, columns = plants)
    
    # establish characteristics of DAC plant
    DAC_elec = consumption_list(DAC_elec_progression, y_plants_start, y_plants_end, DAC_elec_0) # electricity consumption per kg of CO2 producted [kWh/kg]
    DAC_sorbent = consumption_list(DAC_sorbent_progression, y_plants_start, y_plants_end, DAC_sorbent_0) # sorbent consumption per kg of CO2 producted [kg/kg]
    DAC_plant_attributes = ['EIS', 'max age', 'occupation', 'yearly production', 'electricity: DAC', 'sorbent consumption', 'electricity: total', 'CO2 uptake']
    DAC_EIS = list(range(y_plants_start, y_plants_end + 1)) # entry into service of each plant [year]
    DAC_max_age = 20 # how long each plant is assumed to operate [years]
    DAC_ages_0 = [0]*(DAC_max_age - 1) + [1] # distribution in number of plants at start of analysis; note: len(ages_0) MUST be equal to max_age!
    DAC_occupation = 2.4e3 # area which each plant occupies [m2]
    DAC_production = np.array([1e8]*(y_plants_end - y_plants_start + 1)) # production per year of each plant [kg CO2/year]
    DAC_elec_total = DAC_elec + 0.2 # account for 0.2 kWh/kg additional electricity to compress CO2 before transport
    DAC_CO2_uptake = [-1]*(y_plants_end - y_plants_start + 1) # kg/kg -- negative to simplify further calculations (update: negative; emissions: positive)
    DAC_plant_char = pd.DataFrame([DAC_EIS, 
                    [DAC_max_age]*(y_plants_end - y_plants_start + 1), 
                    np.array([DAC_occupation]*(y_plants_end - y_plants_start + 1))/DAC_production, 
                    DAC_production, 
                    DAC_elec,
                    DAC_sorbent,
                    DAC_elec_total,
                    DAC_CO2_uptake],
                    index = DAC_plant_attributes, columns = plants)
    
    # establish characteristics of FT plant
    FT_plant_attributes = ['EIS', 'max age', 'occupation', 'yearly production', 'electricity consumption', 'CO2 consumption', 'H2 consumption', 'CO2 emissions']
    FT_EIS = list(range(y_plants_start, y_plants_end + 1)) # entry into service of each plant [year]
    FT_max_age = 30 # how long each plant is assumed to operate [years]
    FT_ages_0 = [0]*(FT_max_age - 1) + [1] # distribution in number of plants at start of analysis; note: len(ages_0) MUST be equal to max_age!
    FT_occupation = 3.733e4 # area which each plant occupies [m2]
    FT_production = np.array([2.354e11]*(y_plants_end - y_plants_start + 1)) # production per year of each plant [MJ/year]
    FT_plant_char = pd.DataFrame([FT_EIS, 
                    [FT_max_age]*(y_plants_end - y_plants_start + 1), 
                    np.array([FT_occupation]*(y_plants_end - y_plants_start + 1))/FT_production, 
                    FT_production, 
                    FT_elec,
                    FT_co2,
                    FT_h2,
                    FT_co2_emissions],
                    index = FT_plant_attributes, columns = plants)
    
    # establish characteristics of liquefaction plant
    LIQ_plant_attributes = ['EIS', 'max age', 'occupation', 'electricity consumption', 'H2 consumption', 'hydrogen escaped to air', 'yearly production']
    LIQ_EIS = list(range(y_plants_start, y_plants_end + 1)) # entry into service of each plant [year]
    LIQ_max_age = 20 # how long each plant is assumed to operate [years]
    LIQ_ages_0 = [0]*(FT_max_age - 1) + [1] # distribution in number of plants at start of analysis; note: len(ages_0) MUST be equal to max_age!
    LIQ_occupation = 0 # area which each plant occupies [m2]
    LIQ_production = np.array([1e8]*(y_plants_end - y_plants_start + 1)) # production per year of each plant [MJ/year]
    LIQ_elec = consumption_list(LIQ_elec_progression, y_plants_start, y_plants_end, LTQ_elec_0) # electricity consumption per MJ of fuel producted [kWh/MJ]
    LIQ_h2 = np.array([1]*(y_plants_end - y_plants_start + 1)) # MJ H2 consumption per MJ liquified H2 [MJ/MJ]
    LIQ_prod_net = LIQ_production*(1 - LIQ_h2_escaped) # the H2 that can be obtained from a plant per year when accounting for boil-off losses [MJ/MJ]
    LIQ_plant_char = pd.DataFrame([LIQ_EIS, 
                    [LIQ_max_age]*(y_plants_end - y_plants_start + 1), 
                    np.array([LIQ_occupation]*(y_plants_end - y_plants_start + 1))/LIQ_prod_net, 
                    LIQ_elec/LIQ_prod_net*LIQ_production,
                    LIQ_h2/LIQ_prod_net*LIQ_production,
                    LIQ_h2_escaped/h2_energy_density,
                    LIQ_prod_net],
                    index = LIQ_plant_attributes, columns = plants)
    
    # if hydrogen_method == process, hydrogen production itself does not get a fleet
    if hydrogen_method == 'process':
        LIQ_fleet_H2, LIQ_inflow_H2, LIQ_outflow_H2 = fuel_fleet_builder(LIQ_plant_char, LIQ_ages_0, LIQ_max_age, total_hydrogen, y_start, y_end, 'yearly production')
        LIQ_H2_demand_timeline = generic_fleet_use(LIQ_plant_char, 'H2 consumption', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in total MJ
        
        FT_fleet, FT_inflow, FT_outflow = fuel_fleet_builder(FT_plant_char, FT_ages_0, FT_max_age, total_saf, y_start, y_end, 'yearly production')
        FT_CO2_demand_timeline = generic_fleet_use(FT_plant_char, 'CO2 consumption', 'yearly production', FT_fleet, total_saf) # in total kg
        FT_H2_demand_timeline = generic_fleet_use(FT_plant_char, 'H2 consumption', 'yearly production', FT_fleet, total_saf) # in total MJ
        total_FT_CO2 = FT_CO2_demand_timeline.sum(axis = 1)
        
        DAC_fleet_FT, DAC_inflow_FT, DAC_outflow_FT = fuel_fleet_builder(DAC_plant_char, DAC_ages_0, DAC_max_age, total_FT_CO2, y_start, y_end, 'yearly production')
        
        # and subsequenty, create total timeline dataframes for all input/output flows
        LIQ_H2_occupation_timeline = generic_fleet_use(LIQ_plant_char, 'occupation', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in m2
        LIQ_H2_elec_demand_timeline = generic_fleet_use(LIQ_plant_char, 'electricity consumption', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in m2
        LIQ_H2_h2_emissions_timeline = generic_fleet_use(LIQ_plant_char, 'hydrogen escaped to air', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in m2
        
        FT_occupation_timeline = generic_fleet_use(FT_plant_char, 'occupation', 'yearly production', FT_fleet, total_saf) # in m2
        FT_elec_demand_timeline = generic_fleet_use(FT_plant_char, 'electricity consumption', 'yearly production', FT_fleet, total_saf) # in kWh
        FT_co2_emissions_timeline = generic_fleet_use(FT_plant_char, 'CO2 emissions', 'yearly production', FT_fleet, total_saf) # in kg
        DAC_FT_occupation_timeline = generic_fleet_use(DAC_plant_char, 'occupation', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in m2
        DAC_FT_sorbent_demand_timeline = generic_fleet_use(DAC_plant_char, 'sorbent consumption', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in kg
        DAC_FT_elec_demand_timeline = generic_fleet_use(DAC_plant_char, 'electricity: total', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in kWh
        DAC_FT_co2_uptake_timeline = generic_fleet_use(DAC_plant_char, 'CO2 uptake', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in kg
        
        # determine flows for production of H2 fuel
        h2_occupations = [LIQ_H2_occupation_timeline]
        h2_occupation_flows = ['Environmental flow, land occupation']*len(h2_occupations)
        h2_elec_demands = [LIQ_H2_elec_demand_timeline]
        h2_elec_flows = ['Electricity, medium voltage']*len(h2_elec_demands)
        h2_h2_emissions = [LIQ_H2_h2_emissions_timeline, LIQ_H2_demand_timeline*0.01/h2_energy_density]
        h2_h2_flows = ['Environmental flow, hydrogen to air']*len(h2_h2_emissions)
        h2_process_emissions = [LIQ_H2_demand_timeline*(1 + 0.01/h2_energy_density)]
        h2_process_flows = ['Hydrogen']*len(h2_process_emissions)
        
        # determine flows for production of e-fuel
        saf_inflows = [FT_inflow, DAC_inflow_FT]
        saf_inflow_flows = ['FT plant construction', 'DAC system construction']
        saf_outflows = [FT_outflow, DAC_outflow_FT]
        saf_outflow_flows = ['FT plant end-of-life', 'DAC system end-of-life']
        saf_occupations = [FT_occupation_timeline, DAC_FT_occupation_timeline]
        saf_occupation_flows = ['Environmental flow, land occupation']*len(saf_occupations)
        saf_elec_demands = [FT_elec_demand_timeline, DAC_FT_elec_demand_timeline]
        saf_elec_flows = ['Electricity, medium voltage']*len(saf_elec_demands)
        saf_sorbent_demand = [DAC_FT_sorbent_demand_timeline]
        saf_sorbent_flows = ['Sorbent']*len(saf_sorbent_demand)
        saf_h2_emissions = [FT_H2_demand_timeline*0.01/h2_energy_density]
        saf_h2_flows = ['Environmental flow, hydrogen to air']*len(saf_h2_emissions)
        saf_co2_emissions_net = [FT_co2_emissions_timeline, DAC_FT_co2_uptake_timeline]
        saf_co2_flows = ['Environmental flow, carbon dioxide to air']*len(saf_co2_emissions_net)
        saf_process_emissions = [FT_H2_demand_timeline*(1 + 0.01/h2_energy_density)]
        saf_process_flows = ['Hydrogen']*len(saf_process_emissions)
        
        # determine LCIAs for production of H2 fuel
        h2_occupation_LCIAs = calculate_LCIAs_from_list(h2_occupations, h2_occupation_flows, plant_dataframes_yearly, y_start)
        h2_elec_LCIAs = calculate_LCIAs_from_list(h2_elec_demands, h2_elec_flows, plant_dataframes_yearly, y_start)
        h2_h2_LCIAs = calculate_LCIAs_from_list(h2_h2_emissions, h2_h2_flows, plant_dataframes_yearly, y_start)
        h2_process_LCIAs = calculate_LCIAs_from_list(h2_process_emissions, h2_process_flows, process_dataframes_yearly, y_start)
        
        # determine LCIAs for production of e-fuel
        saf_inflows_LCIAs = calculate_LCIAs_from_list(saf_inflows, saf_inflow_flows, plant_dataframes_yearly, y_start)
        saf_outflows_LCIAs = calculate_LCIAs_from_list(saf_outflows, saf_outflow_flows, plant_dataframes_yearly, y_start)
        saf_occupation_LCIAs = calculate_LCIAs_from_list(saf_occupations, saf_occupation_flows, plant_dataframes_yearly, y_start)
        saf_elec_LCIAs = calculate_LCIAs_from_list(saf_elec_demands, saf_elec_flows, plant_dataframes_yearly, y_start)
        saf_sorbent_LCIAs = calculate_LCIAs_from_list(saf_sorbent_demand, saf_sorbent_flows, plant_dataframes_yearly, y_start)
        saf_h2_LCIAs = calculate_LCIAs_from_list(saf_h2_emissions, saf_h2_flows, plant_dataframes_yearly, y_start)
        saf_co2_LCIAs = calculate_LCIAs_from_list(saf_co2_emissions_net, saf_co2_flows, plant_dataframes_yearly, y_start)
        saf_process_LCIAs = calculate_LCIAs_from_list(saf_process_emissions, saf_process_flows, process_dataframes_yearly, y_start)
        
        # combine LCIAs (can be expanded based on graphing needs)
        LCIA_saf_wtt = saf_inflows_LCIAs + saf_outflows_LCIAs + saf_occupation_LCIAs + saf_elec_LCIAs + saf_sorbent_LCIAs + saf_h2_LCIAs + saf_co2_LCIAs + saf_process_LCIAs
        LCIA_h2_wtt = h2_occupation_LCIAs + h2_elec_LCIAs + h2_h2_LCIAs + h2_process_LCIAs
        
        LCIA_saf_wtt_infra = saf_inflows_LCIAs + saf_outflows_LCIAs + saf_occupation_LCIAs
        LCIA_saf_wtt_ops = saf_elec_LCIAs + saf_sorbent_LCIAs + saf_h2_LCIAs + saf_co2_LCIAs + saf_process_LCIAs
        LCIA_h2_wtt_infra = h2_occupation_LCIAs
        LCIA_h2_wtt_ops = h2_elec_LCIAs + h2_h2_LCIAs + h2_process_LCIAs
        
        # create these variables to align with how this function is structured -- however, they cannot be used
        electricity_demands = 0
        electricity_by_fuel = 0
        
        # hydrogen demands
        total_LIQ_H2 = LIQ_H2_demand_timeline.sum(axis = 1)
        total_FT_H2 = FT_H2_demand_timeline.sum(axis = 1)
        hydrogen_demands = pd.DataFrame(np.array([total_FT_H2*(1 + 0.01), total_LIQ_H2*(1 + 0.01)]).T, columns = ['E-fuel', 'Liquid hydrogen fuel'], index = np.arange(len(total_LIQ_H2)))
       
    # if hydrogen_method == fleet, fleets are constructed for PEM plants as well
    if hydrogen_method == 'fleet':
        LIQ_fleet_H2, LIQ_inflow_H2, LIQ_outflow_H2 = fuel_fleet_builder(LIQ_plant_char, LIQ_ages_0, LIQ_max_age, total_hydrogen, y_start, y_end, 'yearly production')
        LIQ_H2_demand_timeline = generic_fleet_use(LIQ_plant_char, 'H2 consumption', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in total MJ
        total_LIQ_H2 = LIQ_H2_demand_timeline.sum(axis = 1)
        
        PEM_fleet_H2, PEM_inflow_H2, PEM_outflow_H2 = fuel_fleet_builder(PEM_plant_char, PEM_ages_0, PEM_max_age, total_LIQ_H2, y_start, y_end, 'yearly production')
        
        FT_fleet, FT_inflow, FT_outflow = fuel_fleet_builder(FT_plant_char, FT_ages_0, FT_max_age, total_saf, y_start, y_end, 'yearly production')
        FT_CO2_demand_timeline = generic_fleet_use(FT_plant_char, 'CO2 consumption', 'yearly production', FT_fleet, total_saf) # in total kg
        FT_H2_demand_timeline = generic_fleet_use(FT_plant_char, 'H2 consumption', 'yearly production', FT_fleet, total_saf) # in total MJ
        total_FT_H2 = FT_H2_demand_timeline.sum(axis = 1)
        total_FT_CO2 = FT_CO2_demand_timeline.sum(axis = 1)
        
        PEM_fleet_FT, PEM_inflow_FT, PEM_outflow_FT = fuel_fleet_builder(PEM_plant_char, PEM_ages_0, PEM_max_age, total_FT_H2, y_start, y_end, 'yearly production')
        DAC_fleet_FT, DAC_inflow_FT, DAC_outflow_FT = fuel_fleet_builder(DAC_plant_char, DAC_ages_0, DAC_max_age, total_FT_CO2, y_start, y_end, 'yearly production')
        
        # and subsequenty, create total timeline dataframes for all input/output flows
        LIQ_H2_occupation_timeline = generic_fleet_use(LIQ_plant_char, 'occupation', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in m2
        LIQ_H2_elec_demand_timeline = generic_fleet_use(LIQ_plant_char, 'electricity consumption', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in m2
        LIQ_H2_h2_emissions_timeline = generic_fleet_use(LIQ_plant_char, 'hydrogen escaped to air', 'yearly production', LIQ_fleet_H2, total_hydrogen) # in m2
        PEM_H2_occupation_timeline = generic_fleet_use(PEM_plant_char, 'occupation', 'yearly production', PEM_fleet_H2, total_LIQ_H2) # in m2
        PEM_H2_water_demand_timeline = generic_fleet_use(PEM_plant_char, 'water consumption', 'yearly production', PEM_fleet_H2, total_LIQ_H2) # in m2
        PEM_H2_elec_demand_timeline = generic_fleet_use(PEM_plant_char, 'electricity: total', 'yearly production', PEM_fleet_H2, total_LIQ_H2) # in m2
        PEM_H2_h2_emissions_timeline = generic_fleet_use(PEM_plant_char, 'hydrogen escaped to air', 'yearly production', PEM_fleet_H2, total_LIQ_H2) # in m2
        
        FT_occupation_timeline = generic_fleet_use(FT_plant_char, 'occupation', 'yearly production', FT_fleet, total_saf) # in m2
        FT_elec_demand_timeline = generic_fleet_use(FT_plant_char, 'electricity consumption', 'yearly production', FT_fleet, total_saf) # in kWh
        FT_co2_emissions_timeline = generic_fleet_use(FT_plant_char, 'CO2 emissions', 'yearly production', FT_fleet, total_saf) # in kg
        PEM_FT_occupation_timeline = generic_fleet_use(PEM_plant_char, 'occupation', 'yearly production', PEM_fleet_FT, total_FT_H2) # in m2
        PEM_FT_water_demand_timeline = generic_fleet_use(PEM_plant_char, 'water consumption', 'yearly production', PEM_fleet_FT, total_FT_H2) # in m2
        PEM_FT_elec_demand_timeline = generic_fleet_use(PEM_plant_char, 'electricity: total', 'yearly production', PEM_fleet_FT, total_FT_H2) # in m2
        PEM_FT_h2_emissions_timeline = generic_fleet_use(PEM_plant_char, 'hydrogen escaped to air', 'yearly production', PEM_fleet_FT, total_FT_H2) # in m2
        DAC_FT_occupation_timeline = generic_fleet_use(DAC_plant_char, 'occupation', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in m2
        DAC_FT_sorbent_demand_timeline = generic_fleet_use(DAC_plant_char, 'sorbent consumption', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in kg
        DAC_FT_elec_demand_timeline = generic_fleet_use(DAC_plant_char, 'electricity: total', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in kWh
        DAC_FT_co2_uptake_timeline = generic_fleet_use(DAC_plant_char, 'CO2 uptake', 'yearly production', DAC_fleet_FT, total_FT_CO2) # in kg
        
        # determine flows for production of H2 fuel
        h2_inflows = [PEM_inflow_H2]
        h2_inflow_flows = ['PEM electrolyzer construction']
        h2_outflows = [PEM_outflow_H2]
        h2_outflow_flows = ['PEM electrolyzer end-of-life']
        h2_occupations = [PEM_H2_occupation_timeline, LIQ_H2_occupation_timeline]
        h2_occupation_flows = ['Environmental flow, land occupation']*len(h2_occupations)
        h2_elec_demands = [PEM_H2_elec_demand_timeline, LIQ_H2_elec_demand_timeline]
        h2_elec_flows = ['Electricity, medium voltage']*len(h2_elec_demands)
        h2_water_demands = [PEM_H2_water_demand_timeline]
        h2_water_flows = ['Water, deionised']*len(h2_water_demands)
        h2_h2_emissions = [PEM_H2_h2_emissions_timeline, LIQ_H2_h2_emissions_timeline]
        h2_h2_flows = ['Environmental flow, hydrogen to air']*len(h2_h2_emissions)
        
        # determine flows for production of e-fuel
        saf_inflows = [FT_inflow, DAC_inflow_FT, PEM_inflow_FT]
        saf_inflow_flows = ['FT plant construction', 'DAC system construction', 'PEM electrolyzer construction']
        saf_outflows = [FT_outflow, DAC_outflow_FT, PEM_outflow_FT]
        saf_outflow_flows = ['FT plant end-of-life', 'DAC system end-of-life', 'PEM electrolyzer end-of-life']
        saf_occupations = [FT_occupation_timeline, DAC_FT_occupation_timeline, PEM_FT_occupation_timeline]
        saf_occupation_flows = ['Environmental flow, land occupation']*len(saf_occupations)
        saf_elec_demands = [FT_elec_demand_timeline, DAC_FT_elec_demand_timeline, PEM_FT_elec_demand_timeline]
        saf_elec_flows = ['Electricity, medium voltage']*len(saf_elec_demands)
        saf_water_demands = [PEM_FT_water_demand_timeline]
        saf_water_flows = ['Water, deionised']*len(saf_water_demands)
        saf_sorbent_demand = [DAC_FT_sorbent_demand_timeline]
        saf_sorbent_flows = ['Sorbent']*len(saf_sorbent_demand)
        saf_h2_emissions = [PEM_FT_h2_emissions_timeline]
        saf_h2_flows = ['Environmental flow, hydrogen to air']*len(saf_h2_emissions)
        saf_co2_emissions_net = [FT_co2_emissions_timeline, DAC_FT_co2_uptake_timeline]
        saf_co2_flows = ['Environmental flow, carbon dioxide to air']*len(saf_co2_emissions_net)
        
        # determine LCIAs for production of H2 fuel
        h2_inflows_LCIAs = calculate_LCIAs_from_list(h2_inflows, h2_inflow_flows, plant_dataframes_yearly, y_start)
        h2_outflows_LCIAs = calculate_LCIAs_from_list(h2_outflows, h2_outflow_flows, plant_dataframes_yearly, y_start)
        h2_occupation_LCIAs = calculate_LCIAs_from_list(h2_occupations, h2_occupation_flows, plant_dataframes_yearly, y_start)
        h2_elec_LCIAs = calculate_LCIAs_from_list(h2_elec_demands, h2_elec_flows, plant_dataframes_yearly, y_start)
        h2_water_LCIAs = calculate_LCIAs_from_list(h2_water_demands, h2_water_flows, plant_dataframes_yearly, y_start)
        h2_h2_LCIAs = calculate_LCIAs_from_list(h2_h2_emissions, h2_h2_flows, plant_dataframes_yearly, y_start)
        
        # determine LCIAs for production of e-fuel
        saf_inflows_LCIAs = calculate_LCIAs_from_list(saf_inflows, saf_inflow_flows, plant_dataframes_yearly, y_start)
        saf_outflows_LCIAs = calculate_LCIAs_from_list(saf_outflows, saf_outflow_flows, plant_dataframes_yearly, y_start)
        saf_occupation_LCIAs = calculate_LCIAs_from_list(saf_occupations, saf_occupation_flows, plant_dataframes_yearly, y_start)
        saf_elec_LCIAs = calculate_LCIAs_from_list(saf_elec_demands, saf_elec_flows, plant_dataframes_yearly, y_start)
        saf_water_LCIAs = calculate_LCIAs_from_list(saf_water_demands, saf_water_flows, plant_dataframes_yearly, y_start)
        saf_sorbent_LCIAs = calculate_LCIAs_from_list(saf_sorbent_demand, saf_sorbent_flows, plant_dataframes_yearly, y_start)
        saf_h2_LCIAs = calculate_LCIAs_from_list(saf_h2_emissions, saf_h2_flows, plant_dataframes_yearly, y_start)
        saf_co2_LCIAs = calculate_LCIAs_from_list(saf_co2_emissions_net, saf_co2_flows, plant_dataframes_yearly, y_start)
        
        # combine LCIAs (can be expanded based on graphing needs)
        LCIA_saf_wtt = saf_inflows_LCIAs + saf_outflows_LCIAs + saf_occupation_LCIAs + saf_elec_LCIAs + saf_water_LCIAs + saf_sorbent_LCIAs + saf_h2_LCIAs + saf_co2_LCIAs
        LCIA_h2_wtt =  h2_inflows_LCIAs + h2_outflows_LCIAs + h2_occupation_LCIAs + h2_elec_LCIAs + h2_water_LCIAs + h2_h2_LCIAs
        
        LCIA_saf_wtt_infra = saf_inflows_LCIAs + saf_outflows_LCIAs + saf_occupation_LCIAs
        LCIA_saf_wtt_ops = saf_elec_LCIAs + saf_water_LCIAs + saf_sorbent_LCIAs + saf_h2_LCIAs + saf_co2_LCIAs
        LCIA_h2_wtt_infra = h2_inflows_LCIAs + h2_outflows_LCIAs + h2_occupation_LCIAs
        LCIA_h2_wtt_ops = h2_elec_LCIAs + h2_water_LCIAs + h2_h2_LCIAs
        
        # electricity demands
        electricity_h2_h2 = (PEM_H2_elec_demand_timeline).sum(axis = 1)
        electricity_h2_ft = (PEM_FT_elec_demand_timeline).sum(axis = 1)
        electricity_dac = (DAC_FT_elec_demand_timeline).sum(axis = 1)
        electricity_liq = (LIQ_H2_elec_demand_timeline).sum(axis = 1)
        electricity_ft = (FT_elec_demand_timeline).sum(axis = 1)
        electricity_demands = pd.DataFrame(np.array([electricity_h2_h2 + electricity_h2_ft, electricity_dac, electricity_liq, electricity_ft]).T, columns = ['Water electrolysis', 'Direct air capture', 'Hydrogen liquefaction', 'Fischer-Tropsch process'], index = np.arange(len(electricity_ft)))
        electricity_by_fuel = pd.DataFrame(np.array([electricity_h2_ft + electricity_dac + electricity_ft, electricity_h2_h2 + electricity_liq]).T, columns = ['E-fuel', 'Liquid hydrogen fuel'], index = np.arange(len(electricity_ft)))
        
        # hydrogen demands
        hydrogen_demands = pd.DataFrame(np.array([total_FT_H2*(1 + 0.01), total_LIQ_H2*(1 + 0.01)]).T, columns = ['E-fuel', 'Liquid hydrogen fuel'], index = np.arange(len(total_LIQ_H2)))
        
    # create additional variables for export to plotting
    # fuel production capacity
    FT_capacity_timeline = generic_fleet_use(FT_plant_char, 'max age', 'yearly production', FT_fleet)/FT_max_age # in MJ
    LH2_capacity_timeline = generic_fleet_use(LIQ_plant_char, 'max age', 'yearly production', LIQ_fleet_H2)/LIQ_max_age # in MJ
    FT_capacity = (FT_capacity_timeline).sum(axis = 1)
    LH2_capacity = (LH2_capacity_timeline).sum(axis = 1)
    fuel_capacities = pd.DataFrame(np.array([FT_capacity, LH2_capacity]).T, columns = ['E-fuel', 'Liquid hydrogen fuel'], index = np.arange(len(FT_capacity)))
    
    # fuel production capacity added
    FT_capacity_added_timeline = generic_fleet_use(FT_plant_char, 'max age', 'yearly production', FT_inflow)/FT_max_age # in MJ
    LH2_capacity_added_timeline = generic_fleet_use(LIQ_plant_char, 'max age', 'yearly production', LIQ_inflow_H2)/LIQ_max_age # in MJ
    FT_capacity_added = (FT_capacity_added_timeline).sum(axis = 1)
    LH2_capacity_added = (LH2_capacity_added_timeline).sum(axis = 1)
    fuel_capacities_added = pd.DataFrame(np.array([FT_capacity_added, LH2_capacity_added]).T, columns = ['E-fuel', 'Liquid hydrogen fuel'], index = np.arange(len(FT_capacity_added)))    
    
    return [LCIA_saf_wtt, LCIA_h2_wtt, LCIA_saf_wtt_infra, LCIA_saf_wtt_ops, LCIA_h2_wtt_infra, LCIA_h2_wtt_ops, electricity_demands, fuel_capacities, electricity_by_fuel, fuel_capacities_added, hydrogen_demands]

#%% some functions that are primarily for plotting
def impact_cat_result_per_aircraft(LCIA_result, impact_cat):
    result = []
    for key, df in LCIA_result.items(): result.append(df[impact_cat])
    return pd.DataFrame(result, index = list(range(len(result))))

def div_per_aircraft(distr_per_aircraft, total):
    share_per_aircraft = distr_per_aircraft.div(distr_per_aircraft.sum(axis = 1), axis = 0) # normalise distribution into a share
    total_per_aircraft = share_per_aircraft.mul(total, axis = 0) # multiply share by total
    
    return total_per_aircraft
    
def extract_processes(list_of_dic, impact_cat, i_list):
    impacts_total = pd.DataFrame(0, columns = list_of_dic[0][list(list_of_dic[0].keys())[0]].index, index = np.arange(len(list_of_dic[0])))
    
    for i in i_list:
        impacts_here = impact_cat_result_per_aircraft(list_of_dic[i], impact_cat)
        impacts_total = impacts_total.add(impacts_here, fill_value=0)
        
    return impacts_total

#%% large function combining all previous functions to more easily execute a series of scenarios
def single_type_scenario(flight_start_year, flight_end_year, aircraft_dataframes_yearly, aircraft_char, rpk_segments, max_age_0, ages_0, aaf_0, aaf_milestones, no_decreases, occupation, cirrus_rf_change_h2, plant_start_year, plant_end_year, plants, PEM_elec_progression, PEM_elec_0, PEM_occupation, PEM_water, PEM_h2_escaped, DAC_elec_progression, DAC_elec_0, DAC_sorbent_progression, DAC_sorbent_0, FT_elec, FT_co2, FT_h2, FT_co2_emissions, LIQ_elec_progression, LTQ_elec_0, LIQ_h2_escaped, hydrogen_method, plant_dataframes_yearly, process_dataframes_yearly):
    # several functions are run for each of the destination pairs (scenarios connecting demand, flight distance, and hydrogen share)
    rpk = np.zeros(flight_end_year - flight_start_year + 1)
    fleet = pd.DataFrame(0, columns = aircraft_char.columns, index = np.arange(len(rpk)))
    inflow = pd.DataFrame(0, columns = aircraft_char.columns, index = np.arange(len(rpk)))
    outflow = pd.DataFrame(0, columns = aircraft_char.columns, index = np.arange(len(rpk)))
    fuel_LTO = pd.DataFrame(0, columns = aircraft_char.columns, index = np.arange(len(rpk)))
    fuel_CCD = pd.DataFrame(0, columns = aircraft_char.columns, index = np.arange(len(rpk)))
    for index, segment in rpk_segments.items():
        # build flights
        rpk_here = segment['yearly RPK']
        
        # create selection from aircraft_char that matches aircraft type and range of the segment at hand
        segment_aircraft_char = aircraft_char.T
        segment_aircraft_char = segment_aircraft_char[(segment_aircraft_char['aircraft type'] == segment['aircraft type']) & (segment_aircraft_char['range'] >= segment['nominal flight distance'])]
        segment_aircraft_char = segment_aircraft_char.T
        
        # build fleet over time
        fleet_here, inflow_here, outflow_here = build_fleet(segment_aircraft_char, max_age_0, ages_0, rpk_here, occupation, flight_start_year, segment['H2 share']) # number of each type of aircraft in fleet as timeseries
        
        # build LCI(A) of use phase over time
        # create dataframes of fuel demand [MJ] -- fuel type agnostic
        fuel_LTO_here, fuel_CCD_here = use_fleet(fleet_here, segment['nominal flight distance'], rpk_here, occupation, segment_aircraft_char, index)
    
        rpk = rpk + rpk_here
        fleet = fleet.add(fleet_here, fill_value=0)
        inflow = inflow.add(inflow_here, fill_value=0)
        outflow = outflow.add(outflow_here, fill_value=0)
        fuel_LTO = fuel_LTO.add(fuel_LTO_here, fill_value=0)
        fuel_CCD = fuel_CCD.add(fuel_CCD_here, fill_value=0)
    
    # aaf_goal_timeline: timeline of desired AAF share over time, based on initial share and milestone years
    # hydrogen_share: share of hydrogen w.r.t. to total fuel mix
    # saf_share: how much **of the hydrocarbon fuel share** should be SAF in order to meet the AAF share goal timeline
    aaf_goal_timeline, hydrogen_share, saf_share = aaf_share_list(flight_start_year, flight_end_year, aaf_0, aaf_milestones, aircraft_char, fuel_LTO, fuel_CCD, no_decreases) # expressed as share of total fuel demand per year [MJ/MJ]
    fuel_LTO_fossil, fuel_LTO_saf, fuel_LTO_hydrogen = fuel_quantities(aircraft_char, fuel_LTO, hydrogen_share, saf_share)
    fuel_CCD_fossil, fuel_CCD_saf, fuel_CCD_hydrogen = fuel_quantities(aircraft_char, fuel_CCD, hydrogen_share, saf_share)
    
    # add fuel quantities together (for later graphing)
    total_fossil = (fuel_LTO_fossil + fuel_CCD_fossil).sum(axis = 1)
    total_saf = (fuel_LTO_saf + fuel_CCD_saf).sum(axis = 1)
    total_hydrogen = (fuel_LTO_hydrogen + fuel_CCD_hydrogen).sum(axis = 1)
    total_fuel = pd.DataFrame([total_fossil, total_saf, total_hydrogen], index = ['Fossil kerosene', 'E-fuel', 'Hydrogen fuel']).T
    
    # add fuels together but still seperated per aircraft (for later graphing)
    fossil_by_aircraft = fuel_LTO_fossil + fuel_CCD_fossil
    saf_by_aircraft = fuel_LTO_saf + fuel_CCD_saf
    hydrogen_by_aircraft = fuel_LTO_hydrogen + fuel_CCD_hydrogen 
    total_fuel_by_aircraft = fossil_by_aircraft.add(saf_by_aircraft, fill_value=0).add(hydrogen_by_aircraft, fill_value=0)
    
    # build LCI(A) of aircraft system over time
    LCIA_inflow = allocate_LCIA(flight_start_year, 'Aircraft manufacturing', inflow, aircraft_dataframes_yearly)
    LCIA_outflow = allocate_LCIA(flight_start_year, 'Aircraft end-of-life', outflow, aircraft_dataframes_yearly)
    
    # get LCIA results of fuel production systems
    LCIAs_fuels_wtt = define_and_use_fuel_plants(flight_start_year, flight_end_year, plant_start_year, plant_end_year, plants, hydrogen_method, total_hydrogen, total_saf, PEM_elec_progression, PEM_elec_0, PEM_occupation, PEM_water, PEM_h2_escaped, DAC_elec_progression, DAC_elec_0, DAC_sorbent_progression, DAC_sorbent_0, FT_elec, FT_co2, FT_h2, FT_co2_emissions, LIQ_elec_progression, LTQ_elec_0, LIQ_h2_escaped, plant_dataframes_yearly, process_dataframes_yearly)
    
    # use fuel quantities abtained above to multiply LCI(A)s
    LCIA_fossil_wtt = allocate_LCIA(flight_start_year, 'Fossil kerosene', fuel_LTO_fossil + fuel_CCD_fossil, aircraft_dataframes_yearly)
    LCIA_fossil_LTO_ttw = allocate_LCIA(flight_start_year, 'Fuel use, fossil kerosene, LTO', fuel_LTO_fossil, aircraft_dataframes_yearly)
    LCIA_fossil_CCD_ttw = allocate_LCIA(flight_start_year, 'Fuel use, fossil kerosene, CCD', fuel_CCD_fossil, aircraft_dataframes_yearly)
    
    LCIA_saf_wtt = list_of_dic_to_dic(LCIAs_fuels_wtt[0])
    LCIA_saf_LTO_ttw = allocate_LCIA(flight_start_year, 'Fuel use, syn-kerosene, LTO', fuel_LTO_saf, aircraft_dataframes_yearly)
    LCIA_saf_CCD_ttw = allocate_LCIA(flight_start_year, 'Fuel use, syn-kerosene, CCD', fuel_CCD_saf, aircraft_dataframes_yearly)
    
    LCIA_h2_wtt = list_of_dic_to_dic(LCIAs_fuels_wtt[1])
    LCIA_h2_LTO_ttw = allocate_LCIA(flight_start_year, 'Fuel use, H2 turbine, LTO', fuel_LTO_hydrogen, aircraft_dataframes_yearly)
    LCIA_h2_CCD_ttw = allocate_LCIA(flight_start_year, 'Fuel use, H2 turbine, CCD', fuel_CCD_hydrogen, aircraft_dataframes_yearly)
    
    # alterenative perspective on wtt
    LCIA_saf_wtt_infra = list_of_dic_to_dic(LCIAs_fuels_wtt[2])
    LCIA_saf_wtt_ops = list_of_dic_to_dic(LCIAs_fuels_wtt[3])
    LCIA_h2_wtt_infra = list_of_dic_to_dic(LCIAs_fuels_wtt[4])
    LCIA_h2_wtt_ops = list_of_dic_to_dic(LCIAs_fuels_wtt[5])
    
    # create a list of all the dictionaries created above
    # list_of_dic = [LCIA_inflow, LCIA_outflow, LCIA_fossil_wtt, LCIA_fossil_LTO_ttw, LCIA_fossil_CCD_ttw, LCIA_saf_wtt, LCIA_saf_LTO_ttw, LCIA_saf_CCD_ttw, LCIA_h2_wtt, LCIA_h2_LTO_ttw, LCIA_h2_CCD_ttw]
    list_of_dic = [LCIA_inflow, LCIA_outflow, LCIA_fossil_wtt, LCIA_fossil_LTO_ttw, LCIA_fossil_CCD_ttw, LCIA_saf_wtt_infra, LCIA_saf_wtt_ops, LCIA_saf_LTO_ttw, LCIA_saf_CCD_ttw, LCIA_h2_wtt_infra, LCIA_h2_wtt_ops, LCIA_h2_LTO_ttw, LCIA_h2_CCD_ttw]
    
    # turn list of dictionaries into single dataframe of impacts over time
    LCIA_df = total_LCIA(list_of_dic)
    
    # calculate RF impact with LWE
    lwe_input = pd.DataFrame()
    lwe_input['years'] = list(range(flight_start_year, flight_end_year + 1))
    
    for impact_cat in aircraft_dataframes_yearly[list(aircraft_dataframes_yearly.keys())[0]].columns[16:28]: # these indexes are specific to the way the LCIA results are exported from the AB
        lwe_inflow = impact_cat_result(LCIA_inflow, impact_cat)
        lwe_outflow = impact_cat_result(LCIA_outflow, impact_cat)
        lwe_fossil_wtt = impact_cat_result(LCIA_fossil_wtt, impact_cat)
        lwe_saf_wtt = impact_cat_result(LCIA_saf_wtt, impact_cat)
        lwe_h2_wtt = impact_cat_result(LCIA_h2_wtt, impact_cat)
        lwe_total = lwe_inflow + lwe_outflow + lwe_fossil_wtt + lwe_saf_wtt + lwe_h2_wtt
        lwe_input[impact_cat+' (ground)'] = lwe_total
      
    for impact_cat in aircraft_dataframes_yearly[list(aircraft_dataframes_yearly.keys())[0]].columns[27:]: # these indexes are specific to the way the LCIA results are exported from the AB
        lwe_fossil_LTO_ttw = impact_cat_result(LCIA_fossil_LTO_ttw, impact_cat)
        lwe_fossil_CCD_ttw = impact_cat_result(LCIA_fossil_CCD_ttw, impact_cat)
        lwe_saf_LTO_ttw = impact_cat_result(LCIA_saf_LTO_ttw, impact_cat)
        lwe_saf_CCD_ttw = impact_cat_result(LCIA_saf_CCD_ttw, impact_cat)
        lwe_h2_LTO_ttw = impact_cat_result(LCIA_h2_LTO_ttw, impact_cat)
        lwe_h2_CCD_ttw = impact_cat_result(LCIA_h2_CCD_ttw, impact_cat)
        lwe_total = lwe_fossil_LTO_ttw + lwe_fossil_CCD_ttw + lwe_saf_LTO_ttw + lwe_saf_CCD_ttw + lwe_h2_LTO_ttw + lwe_h2_CCD_ttw
        lwe_input[impact_cat+' (flight)'] = lwe_total
     
    # do LWE calculations
    lwe_input['cirrus'] = cirrus_calc(hydrogen_share, saf_share, fleet, aircraft_char, cirrus_rf_change_h2)
       
    RF, RF_low, RF_high = emissions_to_LWE(lwe_input, flight_start_year, flight_end_year)
    RF_basic = RF.iloc[:,0:2]
    
    LCIA_df['Radiative forcing'] = list(RF.sum(axis = 1))
    LCIA_df['Radiative forcing (5%)'] = list(RF_low.sum(axis = 1))
    LCIA_df['Radiative forcing (95%)'] = list(RF_high.sum(axis = 1))
    LCIA_df['Radiative forcing (excl. aviation non-CO$_2$)'] = list(RF_basic.sum(axis = 1))

    # additional RF calculations for sensitivity analysis
    no_aaf_change_lwe_input = lwe_input.copy()
    no_aaf_change_lwe_input['cirrus'] = cirrus_calc(hydrogen_share, saf_share, fleet, aircraft_char, cirrus_rf_change_h2, no_aaf_change=True)
    RF_no_AAF_change, RF_low_no_AAF_change, RF_high_no_AAF_change = emissions_to_LWE(no_aaf_change_lwe_input, flight_start_year, flight_end_year)

    LCIA_df['Radiative forcing (excl. AAF change to AIC, mean)'] = list(RF_no_AAF_change.sum(axis = 1))
    LCIA_df['Radiative forcing (excl. AAF change to AIC, 5%)'] = list(RF_low_no_AAF_change.sum(axis = 1))
    LCIA_df['Radiative forcing (excl. AAF change to AIC, 95%)'] = list(RF_high_no_AAF_change.sum(axis = 1))

    # do GWP* calculations
    distance_total = (fleet*aircraft_char.loc['yearly distance']).fillna(0).sum(axis=1)
    fuel_effect_correction_contrails = calc_cirrus_change(saf_share, fleet, aircraft_char, cirrus_rf_change_h2)
    GWPstar_df_climate = perform_GWPstar_calc(flight_start_year, flight_end_year, lwe_input, distance_total, fuel_effect_correction_contrails)

    # and again for sensitivity analysis
    fuel_effect_correction_contrails_sens = calc_cirrus_change(saf_share, fleet, aircraft_char, cirrus_rf_change_h2, no_aaf_change=True)
    GWPstar_df_climate_sens = perform_GWPstar_calc(flight_start_year, flight_end_year, lwe_input, distance_total, fuel_effect_correction_contrails_sens)

    # export additional variables for plotting
    electricity_demands = LCIAs_fuels_wtt[6]
    fuel_capacities = LCIAs_fuels_wtt[7]
    electricity_by_fuel = LCIAs_fuels_wtt[8]
    fuel_capacities_added = LCIAs_fuels_wtt[9]
    hydrogen_demands = LCIAs_fuels_wtt[10]
    
    # the RPK supplied by each aircraft type, per year
    rpk_per_aircraft = fleet*np.array([occupation]*len(fleet.columns)).T*aircraft_char.loc['seats']*aircraft_char.loc['yearly distance']
    
    # the MJ required per RPK, per year, split into HC and LH2
    rpk_per_aircraft[rpk_per_aircraft == 0] = np.nan
    mj_per_rpk = total_fuel_by_aircraft.div(rpk_per_aircraft)
    mj_per_rpk['Weighted average'] = total_fuel.sum(axis = 1)/rpk_per_aircraft.sum(axis = 1)
    
    rpk_hc = (rpk_per_aircraft*(aircraft_char.loc['fuel type'] == 'hydrocarbon')).sum(axis = 1)
    rpk_h2 = (rpk_per_aircraft*(aircraft_char.loc['fuel type'] == 'hydrogen')).sum(axis = 1)
    rpk_hc[rpk_hc==0] = np.nan
    rpk_h2[rpk_h2==0] = np.nan
    
    # additionally required for full picture
    rpk_hc_fossil = rpk_hc*(1 - saf_share)
    rpk_hc_saf = rpk_hc*saf_share
    rpk_total_per_fuel = pd.DataFrame(np.array([rpk_hc_fossil, rpk_hc_saf, rpk_h2]).T, columns = ['Fossil kerosene', 'E-fuel', 'Hydrogen fuel'], index = np.arange(len(rpk_h2)))
    
    return [list_of_dic, LCIA_df, RF, RF_low, RF_high, RF_basic, fleet, inflow, total_fuel, electricity_demands, fuel_capacities, electricity_by_fuel, fuel_capacities_added, rpk_per_aircraft, mj_per_rpk, rpk_total_per_fuel, fossil_by_aircraft, saf_by_aircraft, hydrogen_by_aircraft, hydrogen_demands, RF_no_AAF_change, RF_low_no_AAF_change, RF_high_no_AAF_change, aircraft_char, GWPstar_df_climate, GWPstar_df_climate_sens]
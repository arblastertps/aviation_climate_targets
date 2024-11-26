import pandas as pd
import numpy as np
import os
import pickle 
import logging
logging.basicConfig(level=logging.INFO)

from _10_functions import *
from _11_define_scenarios import * 
from _21_plot_scenarios import *

#%% function in which a variaty of scenario variables (constant across scenarios) are defined and the scenario is run
def run_scenario(y_start, y_end, y_init, hydrogen_method, aircraft_names, aircraft_dataframes_yearly, plants, plant_dataframes_yearly, process_dataframes_yearly, growth_scenario, occuptation_improvements, aaf_advanced, hydrogen_introduced, aircraft_performance, lh2_aircraft_performance, PEM_performance, DAC_performance, FT_performance, LIQ_performance):    
    # starting passenger-kilometer distance for flights [RPK]
    rpk_2019 = [1.71e11, 3.58e11, 1.24e11, 2.34e10,  # intra-EU+ NB flights
                   4.39e9, 3.95e10, 7.47e10, 6.58e10,   # extra-EU+ NB flights
                   1.49e10, 1.21e11, 1.80e11, 2.27e11,  # extra-EU+ WB flights
                   9.85e9]                              # intra-EU+ WB flights
 
    # turn initial RPK into lists according to scenario
    rpk_lists = rpk_from_growth_scenario(y_init, y_start, y_end, rpk_2019, growth_scenario)
    
    # create lists of destinations and matching RPK demand for this type of aircraft
    # for narrow-body aircraft: intra-EU flights suitable for hydrogen, intra-EU flights not suitable for hydrogen, NB EU-departing flights
    if hydrogen_introduced == True:
        h2_potential = 1 # share of intra-EU RPK which could be serviced by hydrogen aircraft
    else: h2_potential = 0
            
    # distance of example flights [km]
    flight_distance = [500, 1500, 2500, 3500,
                       500, 1500, 2500, 3500,
                       2000, 5000, 7000, 9000,
                       2000]   
    # share of RPK which are met by hydrogen aircraft (provided sufficient hydrogen aircraft are available) [-]
    h2_share = [h2_potential, h2_potential, h2_potential, h2_potential,
                0, 0, 0, 0,
                h2_potential*0.5, h2_potential*0.5, h2_potential*0.5, h2_potential*0.5,
                h2_potential]
    ac_type = ['NB', 'NB', 'NB', 'NB',
               'NB', 'NB', 'NB', 'NB',
               'WB', 'WB', 'WB', 'WB',
               'WB']
    
    segment_attributes = ['yearly RPK', 'nominal flight distance', 'H2 share', 'aircraft type']
    rpk_segments =  pd.DataFrame([rpk_lists, flight_distance, h2_share, ac_type], index = segment_attributes)
        
    # fuel efficiency of conventional aircraft
    NB_old_fuel_LTO = 32322.584 # MJ fuel per LTO cycle
    NB_old_fuel_CCD = np.array([7.19e4, 1.90e5, 3.09e5, 4.30e5, # MJ fuel per CCD cycle for reference flight distances
                                7.19e4, 1.90e5, 3.09e5, 4.30e5, 
                                0, 0, 0, 0, # placeholders for distance buckets that are exclusive to WB aircraft
                                0]) 
    NB_new_fuel_LTO = 25988.512 
    NB_new_fuel_CCD = np.array([6.40e4, 1.64e5, 2.66e5, 3.70e5,
                                6.40e4, 1.64e5, 2.66e5, 3.70e5,
                                0, 0, 0, 0,
                                0]) 
    WB_old_fuel_LTO = 102204.120
    WB_old_fuel_CCD = np.array([0, 0, 0, 0, # placeholders for distance buckets that are exclusive to NB aircraft
                                0, 0, 0, 0,
                                7.11e5, 1.77e6, 2.46e6, 3.18e6,
                                7.11e5]) 
    WB_new_fuel_LTO = 84175.080
    WB_new_fuel_CCD = np.array([0, 0, 0, 0,
                                0, 0, 0, 0,
                                5.84e5, 1.44e6, 2.01e6, 2.59e6,
                                5.84e5]) 
    
    # fuel efficiency of future hydrocarbon aircraft
    NB_upcoming_improvement, NB_future_improvement, WB_upcoming_improvement, WB_future_improvement = define_AC_performance(aircraft_performance)

    NB_upcoming_LTO = NB_new_fuel_LTO*NB_upcoming_improvement
    NB_upcoming_CCD = NB_new_fuel_CCD*NB_upcoming_improvement
    NB_future_LTO = NB_new_fuel_LTO*NB_future_improvement
    NB_future_CCD = NB_new_fuel_CCD*NB_future_improvement
    
    WB_upcoming_LTO = WB_new_fuel_LTO*WB_upcoming_improvement
    WB_upcoming_CCD = WB_new_fuel_CCD*WB_upcoming_improvement
    WB_future_LTO = WB_new_fuel_LTO*WB_future_improvement
    WB_future_CCD = WB_new_fuel_CCD*WB_future_improvement
    
    # fuel efficiency of future hydrogen aircraft
    NB_upcoming_H2_change, NB_future_H2_change, WB_future_H2_change = define_HC_performance(lh2_aircraft_performance)
    
    NB_upcoming_h2_LTO = NB_upcoming_LTO*NB_upcoming_H2_change
    NB_upcoming_h2_CCD = NB_upcoming_CCD*NB_upcoming_H2_change
    NB_future_h2_LTO = NB_future_LTO*NB_future_H2_change
    NB_future_h2_CCD = NB_future_CCD*NB_future_H2_change
    
    WB_future_h2_LTO = WB_future_LTO*WB_future_H2_change
    WB_future_h2_CCD = WB_future_CCD*WB_future_H2_change
    
    # yearly distance
    NB_yrly_dist = 5e7/22.1 # from Cox et al. (2018) value for LNB aircraft
    WB_yrly_dist = 9e7/22.1 # from Cox et al. (2018) value for SWB aircraft
    
    # characterise aircraft further
    aircraft_attributes= ['seats', 'EIS', 'aircraft type', 'fuel type', 'range', 'yearly distance', 'max. age', 'LTO fuel', 'reference flight CCD fuel']
    aircraft_char = pd.DataFrame([[180, 189, 189, 189, 189, 189, 
                                   360, 350, 350, 350, 350], # seats [-]
                   [1988, 2016, 2035, 2050, 2035, 2050, 
                    1995, 2015, 2035, 2050, 2050], # year of entry into service [year]
                   ['NB', 'NB', 'NB', 'NB', 'NB', 'NB', 
                    'WB', 'WB', 'WB', 'WB', 'WB'], # either narrow-body (NB) or wide-body (WB)
                   ['hydrocarbon', 'hydrocarbon', 'hydrocarbon', 'hydrocarbon', 'hydrogen', 'hydrogen', 
                    'hydrocarbon', 'hydrocarbon', 'hydrocarbon', 'hydrocarbon', 'hydrogen'], # the type of fuel (hydrocarbon or hydrogen) used in the aircraft
                   [10000, 10000, 10000, 10000, 2000, 3000, 
                    10000, 10000, 10000, 10000, 10000], # aircraft range under typical operations [km/flight]
                   [NB_yrly_dist, NB_yrly_dist, NB_yrly_dist, NB_yrly_dist, NB_yrly_dist, NB_yrly_dist, 
                    WB_yrly_dist, WB_yrly_dist, WB_yrly_dist, WB_yrly_dist, WB_yrly_dist], # yearly distance flown for 1 plane [km]
                   [22, 22, 22, 22, 22, 22,
                    22, 22, 22, 22, 22], # age after which the aircraft is retired [years]. NOTE: this variable is not currently functional; only max_age_0 is currently used; requires changes to build_fleet function to enable.
                   [NB_old_fuel_LTO, NB_new_fuel_LTO, NB_upcoming_LTO, NB_future_LTO, NB_upcoming_h2_LTO, NB_future_h2_LTO, 
                    WB_old_fuel_LTO, WB_new_fuel_LTO, WB_upcoming_LTO, WB_future_LTO, WB_future_h2_LTO], # fuel use per LTO phase [MJ]
                   [NB_old_fuel_CCD, NB_new_fuel_CCD, NB_upcoming_CCD, NB_future_CCD, NB_upcoming_h2_CCD, NB_future_h2_CCD, 
                    WB_old_fuel_CCD, WB_new_fuel_CCD, WB_upcoming_CCD, WB_future_CCD, WB_future_h2_CCD]], index = aircraft_attributes, columns = aircraft_names) # fuel use per CCD phase for initial flight distance [MJ]
        
    # age variables
    max_age_0 = 22 # age *after* which aircraft are retired
    ages_0 = [1/max_age_0]*max_age_0 # distribution in number of aircraft at start of analysis; note: len(ages_0) MUST be equal to max_age!

    # build occupation outside of the main function (important to keep in mind when trying to combine multiple scenarios)
    occupation_0 = 0.8
    if occuptation_improvements == True:
        occupation_2050 = 0.9
        occupation_end = 0.9
    else:
        occupation_2050 = occupation_0
        occupation_end = occupation_0
        
    occupation = build_occupation(occupation_0, y_start, occupation_2050, 2050, occupation_end, y_end) # share of available-seat-kilometers used for revenue-passenger-kilometers, as timeseries which aligns with RPK timeseries

    # variables for alternative fuel share
    aaf_0 = 0.0005 # share of saf in total fuel at start time (from the ICCT)
    no_decreases = False # if True, do not decrease the share of SAF, even if the AAF goal is already met with share of hydrogen
    cirrus_rf_change_h2 = 0.3405 # what is the RF impact of hydrogen aircraft (per km flown) compared to an aircraft using fossil kerosene
    #^0.3405 to match with reduction that e-fuel causes. Literature says that for both, 0-60% could be expected (Kossarev et al., 2023)
    
    if aaf_advanced == 'ReFuelEU base':
        aaf_milestones = pd.DataFrame([[2025, 2030, 2035, 2040, 2045, 2050], [0.02, 0.06, 0.2, 0.34, 0.42, 0.7]], ['year', 'AAF share']).T # milestones based on ReFuelEU Aviation (except for 2060, that one I added myself)
    if aaf_advanced == 'ReFuelEU extended':
        aaf_milestones = pd.DataFrame([[2025, 2030, 2035, 2040, 2045, 2050, 2060], [0.02, 0.06, 0.2, 0.34, 0.42, 0.7, 1]], ['year', 'AAF share']).T # milestones based on ReFuelEU Aviation (except for 2060, that one I added myself)
    if aaf_advanced == 'no ReFuelEU':
        aaf_milestones = pd.DataFrame([[2060], [aaf_0]], ['year', 'AAF share']).T # do not change AAF share

    # define performance of fuel infrastructure (low, mid, or high)
    y_plants_start, y_plants_end = plant_years(y_start, y_end)
    PEM_elec_progression, PEM_elec_0, PEM_occupation, PEM_water, PEM_h2_escaped = define_PEM_performance(PEM_performance, y_plants_start, y_plants_end)
    DAC_elec_progression, DAC_elec_0, DAC_sorbent_progression, DAC_sorbent_0 = define_DAC_performance(DAC_performance, y_plants_start, y_plants_end)
    FT_elec, FT_co2, FT_h2, FT_co2_emissions = define_FT_performance(FT_performance, y_plants_start, y_plants_end)
    LIQ_elec_progression, LTQ_elec_0, LIQ_h2_escaped  = define_LIQ_performance(LIQ_performance, y_plants_start, y_plants_end)
       
    #execute functions
    scenario_results = single_type_scenario(y_start, y_end, aircraft_dataframes_yearly, aircraft_char, rpk_segments, max_age_0, ages_0, aaf_0, aaf_milestones, no_decreases, occupation, cirrus_rf_change_h2, y_plants_start, y_plants_end, plants, PEM_elec_progression, PEM_elec_0, PEM_occupation, PEM_water, PEM_h2_escaped, DAC_elec_progression, DAC_elec_0, DAC_sorbent_progression, DAC_sorbent_0, FT_elec, FT_co2, FT_h2, FT_co2_emissions, LIQ_elec_progression, LTQ_elec_0, LIQ_h2_escaped, hydrogen_method, plant_dataframes_yearly, process_dataframes_yearly)

    return scenario_results + [y_start]

#%% universal scenario variables
# define start and end years
initiation_year = 2019 # start year of 2019 is virtually hard-coded into the model -- RPK buckets and growth lists depend on it
flight_start_year = 2024 # including covid results in strange behaviour from some parameters, so start year excludes worst of the pandemic
flight_end_year = 2070

#%% pre-load files so that, when a large number of scenarios is run, each file only has to be loaded in once
# get LCIA data, list of aircraft, and list of years

aircraft_file_2p5C = 'LCIA_building_blocks/aircraft-SSP2-NDC.xlsx'
wind_file_2p5C = 'LCIA_building_blocks/plants-SSP2-NDC-wind.xlsx'
grid_file_2p5C = 'LCIA_building_blocks/plants-SSP2-NDC-grid.xlsx'
market_file_2p5C = 'LCIA_building_blocks/hydrogen-market-SSP2-NDC.xlsx'

aircraft_file_1p7C = 'LCIA_building_blocks/aircraft-SSP2-PkBudg1150.xlsx'
wind_file_1p7C = 'LCIA_building_blocks/plants-SSP2-PkBudg1150-wind.xlsx'
grid_file_1p7C = 'LCIA_building_blocks/plants-SSP2-PkBudg1150-grid.xlsx'
market_file_1p7C = 'LCIA_building_blocks/hydrogen-market-SSP2-PkBudg1150.xlsx'

aircraft_file_1p4C = 'LCIA_building_blocks/aircraft-SSP1-PkBudg500.xlsx'
wind_file_1p4C = 'LCIA_building_blocks/plants-SSP1-PkBudg500-wind.xlsx'
grid_file_1p4C = 'LCIA_building_blocks/plants-SSP1-PkBudg500-grid.xlsx'
market_file_1p4C = 'LCIA_building_blocks/hydrogen-market-SSP1-PkBudg500.xlsx'

aircraft_names_2p5C, aircraft_dataframes_yearly_2p5C, plant_names_2p5C, plant_dataframes_yearly_wind_2p5C, plant_dataframes_yearly_grid_2p5C, market_dataframes_yearly_2p5C = use_file_names(flight_start_year, flight_end_year, aircraft_file_2p5C, wind_file_2p5C, grid_file_2p5C, market_file_2p5C)
aircraft_names_1p7C, aircraft_dataframes_yearly_1p7C, plant_names_1p7C, plant_dataframes_yearly_wind_1p7C, plant_dataframes_yearly_grid_1p7C, market_dataframes_yearly_1p7C = use_file_names(flight_start_year, flight_end_year, aircraft_file_1p7C, wind_file_1p7C, grid_file_1p7C, market_file_1p7C)
aircraft_names_1p4C, aircraft_dataframes_yearly_1p4C, plant_names_1p4C, plant_dataframes_yearly_wind_1p4C, plant_dataframes_yearly_grid_1p4C, market_dataframes_yearly_1p4C = use_file_names(flight_start_year, flight_end_year, aircraft_file_1p4C, wind_file_1p4C, grid_file_1p4C, market_file_1p4C)

hydrogen_method = 'fleet'
hydrogen_method = 'process'

#%% run selected scenario(s) -- takes around 20-30 seconds per scenario, depending on hardware

# define scenario variables considered
# note: some of these variables are overwritten when performing "background sensitivity"
# note also: when performing "foreground sensitivity" (the results of which include the illustrative scenarios), all possible combinations of provided scenario variables are executed
scenarios_pathways = ['1.7C']  # narrowed down from full possibilities: ['1.4C', '1.7C', '2.5C']
hydrogen_sources = ['grid'] # narrowed down from full possibilities: ['market', 'grid', 'wind']
growth_scenarios = ['high growth', 'base growth', 'low growth', 'degrowth']
aircraft_techs = ['low', 'mid', 'high']
capacity_impl = [True, False]
fuel_techs = ['low', 'mid', 'high']
lh2_techs = ['low', 'mid', 'high']
e_fuel_impl = ['no ReFuelEU', 'ReFuelEU base', 'ReFuelEU extended']
hydrogen_impl = [True, False]

# determine which sets of scenario combinations are executed
# both should be set to False if provided .pkl files are being used
foreground_sensitivity = False
background_sensitivity = False

if foreground_sensitivity == True:
    i = 0
    all_scenario_results = []
    all_scenario_names = []
    for e_fuel in e_fuel_impl:
        if e_fuel == 'no ReFuelEU': # only run a few scenarios here
            pathway                      = scenarios_pathways[0]
            hydrogen_method              = 'fleet'
            hydrogen_source              = 'grid'
            aircraft_names               = aircraft_names_1p7C
            aircraft_dataframes_yearly   = aircraft_dataframes_yearly_1p7C
            plants                       = plant_names_1p7C
            plant_dataframes_yearly      = plant_dataframes_yearly_grid_1p7C
            process_dataframes_yearly    = market_dataframes_yearly_1p7C
            fuel_tech                    = 'low'
            PEM_performance = DAC_performance = FT_performance = LIQ_performance = fuel_tech
            hydrogen = False
            lh2_tech = 'low'
            for aircraft_tech in aircraft_techs:
                for growth in growth_scenarios:
                    for capacity in capacity_impl:
                        scenario_results_here = run_scenario(flight_start_year, flight_end_year, initiation_year, hydrogen_method, aircraft_names, aircraft_dataframes_yearly, plants, plant_dataframes_yearly, process_dataframes_yearly, growth, capacity, e_fuel, hydrogen, aircraft_tech, lh2_tech, PEM_performance, DAC_performance, FT_performance, LIQ_performance)
                        scenario_name_here = scenario_name_generator(pathway, hydrogen_source, growth, aircraft_tech, lh2_tech, fuel_tech, capacity, e_fuel, hydrogen)
                        all_scenario_results.append(scenario_results_here)
                        all_scenario_names.append(scenario_name_here)
                        logging.info(f"Calculated scenario #{i}")
                        i += 1
        else:
            for pathway in scenarios_pathways:
                for hydrogen_source in hydrogen_sources:
                    hydrogen_method = 'fleet'
                    if hydrogen_source == 'market': hydrogen_method = 'process'
                    if pathway == '1.4C':    
                        aircraft_names               = aircraft_names_1p4C
                        aircraft_dataframes_yearly   = aircraft_dataframes_yearly_1p4C
                        plants                       = plant_names_1p4C
                        plant_dataframes_yearly      = plant_dataframes_yearly_grid_1p4C
                        if hydrogen_source == 'wind': plant_dataframes_yearly = plant_dataframes_yearly_wind_1p4C
                        process_dataframes_yearly    = market_dataframes_yearly_1p4C
                    if pathway == '1.7C':    
                        aircraft_names               = aircraft_names_1p7C
                        aircraft_dataframes_yearly   = aircraft_dataframes_yearly_1p7C
                        plants                       = plant_names_1p7C
                        plant_dataframes_yearly      = plant_dataframes_yearly_grid_1p7C
                        if hydrogen_source == 'wind': plant_dataframes_yearly = plant_dataframes_yearly_wind_1p7C
                        process_dataframes_yearly    = market_dataframes_yearly_1p7C    
                    if pathway == '2.5C':    
                        aircraft_names               = aircraft_names_2p5C
                        aircraft_dataframes_yearly   = aircraft_dataframes_yearly_2p5C
                        plants                       = plant_names_2p5C
                        plant_dataframes_yearly      = plant_dataframes_yearly_grid_2p5C
                        if hydrogen_source == 'wind': plant_dataframes_yearly = plant_dataframes_yearly_wind_2p5C
                        process_dataframes_yearly    = market_dataframes_yearly_2p5C
                    for growth in growth_scenarios:
                        for aircraft_tech in aircraft_techs:
                            for capacity in capacity_impl:
                                for fuel_tech in fuel_techs:
                                    PEM_performance = DAC_performance = FT_performance = LIQ_performance = fuel_tech
                                    for hydrogen in hydrogen_impl:
                                        if hydrogen == False: # only one scenario past this point, since difference in LH2_tech has no effect
                                            lh2_tech = 'low'
                                            scenario_results_here = run_scenario(flight_start_year, flight_end_year, initiation_year, hydrogen_method, aircraft_names, aircraft_dataframes_yearly, plants, plant_dataframes_yearly, process_dataframes_yearly, growth, capacity, e_fuel, hydrogen, aircraft_tech, lh2_tech, PEM_performance, DAC_performance, FT_performance, LIQ_performance)
                                            scenario_name_here = scenario_name_generator(pathway, hydrogen_source, growth, aircraft_tech, lh2_tech, fuel_tech, capacity, e_fuel, hydrogen)
                                            all_scenario_results.append(scenario_results_here)
                                            all_scenario_names.append(scenario_name_here)
                                            logging.info(f"Calculated scenario #{i}")
                                            i += 1
                                        else:
                                            for lh2_tech in lh2_techs:                               
                                                scenario_results_here = run_scenario(flight_start_year, flight_end_year, initiation_year, hydrogen_method, aircraft_names, aircraft_dataframes_yearly, plants, plant_dataframes_yearly, process_dataframes_yearly, growth, capacity, e_fuel, hydrogen, aircraft_tech, lh2_tech, PEM_performance, DAC_performance, FT_performance, LIQ_performance)
                                                scenario_name_here = scenario_name_generator(pathway, hydrogen_source, growth, aircraft_tech, lh2_tech, fuel_tech, capacity, e_fuel, hydrogen)
                                                all_scenario_results.append(scenario_results_here)
                                                all_scenario_names.append(scenario_name_here)
                                                logging.info(f"Calculated scenario #{i}")
                                                i += 1
    
    with open('pickled_scenario_results/scenario_results.pkl', 'wb') as f:
        pickle.dump(all_scenario_results, f)
        f.close()
        
    with open('pickled_scenario_results/scenario_names.pkl', 'wb') as f:
        pickle.dump(all_scenario_names, f)
        f.close()

# redefine selected variable values as considered for "background sensitivity"
capacity_impl = [False]
fuel_techs = ['mid']
scenarios_pathways = ['1.4C', '1.7C', '2.5C']
hydrogen_sources = ['market', 'grid', 'wind']

if background_sensitivity == True:
    i = 0
    all_scenario_results_bck = []
    all_scenario_names_bck = []
    for e_fuel in e_fuel_impl:
        for pathway in scenarios_pathways:
            for hydrogen_source in hydrogen_sources:
                hydrogen_method = 'fleet'
                if hydrogen_source == 'market': hydrogen_method = 'process'
                if pathway == '1.4C':    
                    aircraft_names               = aircraft_names_1p4C
                    aircraft_dataframes_yearly   = aircraft_dataframes_yearly_1p4C
                    plants                       = plant_names_1p4C
                    plant_dataframes_yearly      = plant_dataframes_yearly_grid_1p4C
                    if hydrogen_source == 'wind': plant_dataframes_yearly = plant_dataframes_yearly_wind_1p4C
                    process_dataframes_yearly    = market_dataframes_yearly_1p4C
                if pathway == '1.7C':    
                    aircraft_names               = aircraft_names_1p7C
                    aircraft_dataframes_yearly   = aircraft_dataframes_yearly_1p7C
                    plants                       = plant_names_1p7C
                    plant_dataframes_yearly      = plant_dataframes_yearly_grid_1p7C
                    if hydrogen_source == 'wind': plant_dataframes_yearly = plant_dataframes_yearly_wind_1p7C
                    process_dataframes_yearly    = market_dataframes_yearly_1p7C    
                if pathway == '2.5C':    
                    aircraft_names               = aircraft_names_2p5C
                    aircraft_dataframes_yearly   = aircraft_dataframes_yearly_2p5C
                    plants                       = plant_names_2p5C
                    plant_dataframes_yearly      = plant_dataframes_yearly_grid_2p5C
                    if hydrogen_source == 'wind': plant_dataframes_yearly = plant_dataframes_yearly_wind_2p5C
                    process_dataframes_yearly    = market_dataframes_yearly_2p5C
                for growth in growth_scenarios:
                    for aircraft_tech in aircraft_techs:
                        for capacity in capacity_impl:
                            for fuel_tech in fuel_techs:
                                for hydrogen in hydrogen_impl:
                                    fuel_tech_here = fuel_tech
                                    hydrogen_here = hydrogen
                                    if e_fuel == 'no ReFuelEU': 
                                        hydrogen_here = False
                                        fuel_tech_here = 'low'
                                    if hydrogen_here == False: # only one scenario past this point, since difference in LH2_tech has no effect
                                        lh2_tech = 'low'
                                        PEM_performance = DAC_performance = FT_performance = LIQ_performance = fuel_tech_here
                                        scenario_results_here = run_scenario(flight_start_year, flight_end_year, initiation_year, hydrogen_method, aircraft_names, aircraft_dataframes_yearly, plants, plant_dataframes_yearly, process_dataframes_yearly, growth, capacity, e_fuel, hydrogen_here, aircraft_tech, lh2_tech, PEM_performance, DAC_performance, FT_performance, LIQ_performance)
                                        scenario_name_here = scenario_name_generator(pathway, hydrogen_source, growth, aircraft_tech, lh2_tech, fuel_tech_here, capacity, e_fuel, hydrogen_here)
                                        all_scenario_results_bck.append(scenario_results_here)
                                        all_scenario_names_bck.append(scenario_name_here)
                                        logging.info(f"Calculated scenario #{i}")
                                        i += 1
                                    else:
                                        for lh2_tech in lh2_techs:
                                            PEM_performance = DAC_performance = FT_performance = LIQ_performance = fuel_tech_here
                                            scenario_results_here = run_scenario(flight_start_year, flight_end_year, initiation_year, hydrogen_method, aircraft_names, aircraft_dataframes_yearly, plants, plant_dataframes_yearly, process_dataframes_yearly, growth, capacity, e_fuel, hydrogen_here, aircraft_tech, lh2_tech, PEM_performance, DAC_performance, FT_performance, LIQ_performance)
                                            scenario_name_here = scenario_name_generator(pathway, hydrogen_source, growth, aircraft_tech, lh2_tech, fuel_tech_here, capacity, e_fuel, hydrogen_here)
                                            all_scenario_results_bck.append(scenario_results_here)
                                            all_scenario_names_bck.append(scenario_name_here)
                                            logging.info(f"Calculated scenario #{i}")
                                            i += 1
                                            
    with open('pickled_scenario_results/scenario_results_bck.pkl', 'wb') as f:
        pickle.dump(all_scenario_results_bck, f)
        f.close()
        
    with open('pickled_scenario_results/scenario_names_bck.pkl', 'wb') as f:
        pickle.dump(all_scenario_names_bck, f)
        f.close()

#%% load .pkl files

with open('pickled_scenario_results/scenario_results_bck.pkl', 'rb') as f:
    all_scenario_results_bck = pd.Series(pickle.load(f))
with open('pickled_scenario_results/scenario_names_bck.pkl', 'rb') as f:
    all_scenario_names_bck = pd.Series(pickle.load(f))
    
with open('pickled_scenario_results/scenario_results.pkl', 'rb') as f:
    all_scenario_results = pd.Series(pickle.load(f))
with open('pickled_scenario_results/scenario_names.pkl', 'rb') as f:
    all_scenario_names = pd.Series(pickle.load(f))

#%% prepare variables used to plot results
LU_red = '#be1908'
LU_orange = '#f46e32'
LU_turqoise = '#34a3a9'
LU_light_blue = '#5cb1eb'
LU_violet = '#b02079'
LU_green = '#2c712d'
LU_blue = '#001158'

# the original figures use the font Minion Pro, but a more common font such as Georgia also works
plt.rcParams['font.family'] = 'Minion Pro'

folder_figures = 'figures/'
folder_csvs = 'figures_data/'

if not os.path.isdir(folder_figures): os.makedirs(folder_figures)
if not os.path.isdir(folder_csvs): os.makedirs(folder_csvs)

#%% plot the final figures
# comment out lines to select which figures to generate
year_of_interest = 2070 # the year in which emission targets are evaluated (always compared to 2050)
years_graph = list(range(flight_start_year,flight_end_year + 1))

# create plots for main results sections
plot_timeline_results(flight_start_year, flight_end_year, folder_figures, all_scenario_names, all_scenario_results, folder_csvs)

# create heatmaps to check emission targets
heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4 = make_target_heat_map(year_of_interest, flight_start_year, flight_end_year, all_scenario_names, all_scenario_results)
plot_target_heat_map(heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4, folder_figures)
# create four big plots for foreground sensitivity analysis
heat_map_foreground_df_1, heat_map_foreground_df_2, heat_map_foreground_df_3, heat_map_foreground_df_4 = make_foreground_heat_map(year_of_interest, flight_start_year, flight_end_year, all_scenario_names, all_scenario_results)
plot_foreground_heat_map(heat_map_foreground_df_1, heat_map_foreground_df_2, heat_map_foreground_df_3, heat_map_foreground_df_4, folder_figures)
# create nine 2x2 plots for background sensitivity analysis
for background in ['1.4C', '1.7C', '2.5C']:
    for hydrogen_source in ['market', 'grid', 'wind']:
        heat_map_background_df_1, heat_map_background_df_2, heat_map_background_df_3, heat_map_background_df_4 = make_background_heat_map(year_of_interest, flight_start_year, flight_end_year, all_scenario_names_bck, all_scenario_results_bck, background, hydrogen_source)
        plot_background_heat_map(heat_map_background_df_1, heat_map_background_df_2, heat_map_background_df_3, heat_map_background_df_4, folder_figures, background, hydrogen_source)

# create two additional plots to visualise RPK and AAF trajectories
plotting_rpk_aaf = False
if plotting_rpk_aaf == True:
    fig_name_rpk_aaf = 'Fig1_rpk_aaf_overview'
    colours = [LU_turqoise, LU_green, LU_orange, LU_red, LU_light_blue, LU_violet]
    fig, (ax_rpk, ax_aaf) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,1]},figsize = (8,3))
    
    # lines for plotting RPK
    fuel_scenarios = [['no ReFuelEU', False, 'low']]
    ac_scenarios = [['mid',False,'high growth'],
                    ['mid',False,'base growth'],
                    ['mid',False,'low growth'],
                    ['mid',False,'degrowth']]
    background = '1.7C'
    hydrogen_source = 'grid'
    
    scenarios_rpk_per_aircraft = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    for i in range(len(ac_scenarios)):
        ac_tech = ac_scenarios[i][0]
        ops_imp = ac_scenarios[i][1]
        growth =  ac_scenarios[i][2]
        for j in range(len(fuel_scenarios)):
            aaf_impl = fuel_scenarios[j][0]
            h2ac_impl = fuel_scenarios[j][1]
            aaf_tech =  fuel_scenarios[j][2]
            h2ac_tech = 'low'
            scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, aaf_tech, ops_imp, aaf_impl, h2ac_impl)
            scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
            scenarios_rpk_per_aircraft[i][j] = scenario_result_here[13]
                
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            ax_rpk.plot(years_graph, scenarios_rpk_per_aircraft[i][j].sum(axis=1)/1e12, color=colours[i], lw = 1, alpha = 1)

    # lines for plotting AAF
    aaf_timeline_as_is      = aaf_share_for_plt('ReFuelEU base', 2024, 2070)
    aaf_timeline_extended   = aaf_share_for_plt('ReFuelEU extended', 2024, 2070)
    
    ax_aaf.plot(years_graph, aaf_timeline_extended, color=colours[2], lw = 1, alpha = 1)
    ax_aaf.plot(years_graph, aaf_timeline_as_is, color=colours[0], lw = 1, alpha = 1, linestyle = '--')
    
    # formatting plots
    ax_rpk.set_title('(b) Air traffic volume [$\mathregular{10^{12}}$ RPK]', fontweight="bold", fontsize=14)
    ax_aaf.set_title('(c) Alternative aviation fuel share [%]', fontweight="bold", fontsize=14)
 
    ax_rpk.set_ylim(bottom=0)  
    ax_aaf.set_ylim(0,101) 
    ax_rpk.set_xlim(2024,2070)
    ax_aaf.set_xlim(2024,2070)
    
    # make legend
    handles, labels = ax_rpk.get_legend_handles_labels()
    rpk_names = ['high growth', 'base growth', 'low growth', 'degrowth']
    for j in range(len(rpk_names)):
        label = rpk_names[j].replace('\n ','')
        linestyle = '-'
        labels.append(label)
        handles.append(Line2D([0,1],[0,1],linestyle=linestyle, color=colours[j], lw=1))
    by_label = dict(zip(labels, handles))
    ax_rpk.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    
    handles, labels = ax_aaf.get_legend_handles_labels()
    labels.append('ReFuelEU Aviation as-is')
    labels.append('ReFuelEU Aviation extended')
    handles.append(Line2D([0,1],[0,1],linestyle='--', color=colours[0], lw=1))
    handles.append(Line2D([0,1],[0,1],linestyle='-', color=colours[2], lw=1))
    by_label = dict(zip(labels, handles))
    ax_aaf.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)
    
    plt.subplots_adjust(wspace=0.15)
    fig_name = folder_figures+fig_name_rpk_aaf+'.pdf'
    fig.savefig(fig_name,bbox_inches='tight')
    fig_name = folder_figures+fig_name_rpk_aaf+'.png'
    fig.savefig(fig_name,bbox_inches='tight',dpi=600)

    # create and save figure data
    data = {
        "Year": years_graph,
        "AAF share, ReFuelEU Aviation as-is": aaf_timeline_as_is,
        "AAF share, ReFuelEU Aviation extended": aaf_timeline_extended
    }

    ac_scenario_names = ['high growth',
                    'base growth',
                    'low growth',
                    'degrowth']
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            data[f"Air traffic volume [RPK] - {ac_scenario_names[i]}"] = scenarios_rpk_per_aircraft[i][j].sum(axis=1).reset_index(drop=True)

    df = pd.DataFrame(data)

    csv_file_name = folder_csvs+fig_name_rpk_aaf+'.csv'
    df.to_csv(csv_file_name, index=False) 
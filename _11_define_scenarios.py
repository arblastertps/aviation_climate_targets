import pandas as pd
import numpy as np
from _10_functions import *

#%% the scenario variables (traffic growth, technological performance, etc.) are defined by calling the below functions
# generate RPK based on initial values (buckets) and growth trajectory
def rpk_from_growth_scenario(y_init, y_start, y_end, rpk_buckets, growth_scenario):
    # growth scenarios
    # all values are given relative to 2019 (= 1)
    if growth_scenario == 'high growth':
        # list obtained by combining EUROCONTROl data as described in text
        growth_list = [1, # 2019
                        0.43,
                        0.54,
                        0.87,
                        1.01,
                        1.14,
                        1.18,
                        1.22,
                        1.26,
                        1.31,
                        1.34] # 2029
        growth_rate = 0.018 # 2030 and onward
    if growth_scenario == 'base growth':
        # list obtained by combining EUROCONTROl data as described in text
        growth_list = [1, # 2019
                    0.43,
                    0.54,
                    0.87,
                    1.00,
                    1.08,
                    1.11,
                    1.14,
                    1.16,
                    1.19,
                    1.20] # 2029
        growth_rate = 0.012 # 2030 and onward
        
    if growth_scenario == 'low growth':
        # list obtained by combining EUROCONTROl data as described in text
        growth_list = [1, # 2019
                        0.43,
                        0.54,
                        0.87,
                        0.98,
                        1.03,
                        1.04,
                        1.05,
                        1.06,
                        1.07,
                        1.07] # 2029
        growth_rate = 0.006 # 2030 and onward
    
    if growth_scenario == 'degrowth':
        # list obtained by combining EUROCONTROl data (2019-2023) with degrowth scenario as described in text
        growth_list = [1, # 2019
                        0.43,
                        0.54,
                        0.87,
                        0.98,
                        0.95,
                        0.92,
                        0.90,
                        0.87,
                        0.84,
                        0.82,
                        0.79,
                        0.77,
                        0.75,
                        0.72,
                        0.70] # 2034
        growth_rate = 0 # 2035 and onward
        
    # the chosen growth scenario is used to transform the RPK buckets into a timeline list
    rpk_lists = []
    for rpk_0 in rpk_buckets:
        rpk_first_part = [i*rpk_0 for i in growth_list]
        rpk_second_part = build_flights(y_init + len(rpk_first_part), y_end, rpk_first_part[-1], growth_rate)
        rpk_first_part = rpk_first_part[(y_start-y_init):]
        rpk_lists.append(rpk_first_part + rpk_second_part)
    
    return rpk_lists
# define technological performance of proton exchange membrane electrolysis plants
def define_PEM_performance(low_mid_high, y_plants_start, y_plants_end):
    # utility variables for fuel demand
    h2_energy_density = 120 # MJ/kg
    
    if low_mid_high == 'low':
        PEM_elec_progression = pd.DataFrame([[2019, 2030, 2050], np.array([59.5, 52.9, 49.8])/h2_energy_density], ['year', 'electricity consumption']).T # progression of electricity consumption per MJ H2 [kWh/MJH2]
        PEM_elec_0 = 59.5/h2_energy_density # 50 kWh/kgH2, converted to kWh/MJH2
        PEM_occupation = [110]*(y_plants_end - y_plants_start + 1) # area which each plant occupies [m2]
        PEM_water = np.array([14/h2_energy_density]*(y_plants_end - y_plants_start + 1)) # water consumption per MJ of H2 producted [kg/MJH2]
        PEM_h2_escaped = np.array([0.01]*(y_plants_end - y_plants_start + 1)) # hydrogen escaped along the transporation chain between this plant and the next [MJ/MJ]
    
    if low_mid_high == 'mid':
        PEM_elec_progression = pd.DataFrame([[2019, 2030, 2050], np.array([54.6, 51.3, 45.7])/h2_energy_density], ['year', 'electricity consumption']).T # progression of electricity consumption per MJ H2 [kWh/MJH2]
        PEM_elec_0 = 54.6/h2_energy_density # 50 kWh/kgH2, converted to kWh/MJH2
        PEM_occupation = [90]*(y_plants_end - y_plants_start + 1) # area which each plant occupies [m2]
        PEM_water = np.array([10/h2_energy_density]*(y_plants_end - y_plants_start + 1)) # water consumption per kg of H2 producted [kg/MJH2]
        PEM_h2_escaped = np.array([0.01]*(y_plants_end - y_plants_start + 1)) # hydrogen escaped along the transporation chain between this plant and the next [MJ/MJ]
            
    if low_mid_high == 'high':
        PEM_elec_progression = pd.DataFrame([[2019, 2030, 2050], np.array([49.8, 49.0, 45.0])/h2_energy_density], ['year', 'electricity consumption']).T # progression of electricity consumption per MJ H2 [kWh/MJH2]
        PEM_elec_0 = 49.8/h2_energy_density # 50 kWh/kgH2, converted to kWh/MJH2
        PEM_occupation = [50]*(y_plants_end - y_plants_start + 1) # area which each plant occupies [m2]
        PEM_water = np.array([9/h2_energy_density]*(y_plants_end - y_plants_start + 1)) # water consumption per kg of H2 producted [kg/MJH2]
        PEM_h2_escaped = np.array([0.01]*(y_plants_end - y_plants_start + 1)) # hydrogen escaped along the transporation chain between this plant and the next [MJ/MJ]
    
    return PEM_elec_progression, PEM_elec_0, PEM_occupation, PEM_water, PEM_h2_escaped
# define technological performance of direct air capture plants
def define_DAC_performance(low_mid_high, y_plants_start, y_plants_end):
    if low_mid_high == 'low':
        DAC_elec_progression = pd.DataFrame([[2020, 2070], [1.132, 1.132]], ['year', 'electricity consumption']).T # progression of electricity consumption per kg CO2 [kWh/kg]
        DAC_elec_0 = 1.132 # kWh/kg
        DAC_sorbent_progression = pd.DataFrame([[2020, 2070], [0.0075, 0.0075]], ['year', 'sorbent consumption']).T # progression of sorbent consumption per kg CO2 [kg/kg]
        DAC_sorbent_0 = 0.003 # kg/kg
        
    if low_mid_high == 'mid':
        DAC_elec_progression = pd.DataFrame([[2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070], 
                                             [1.132,
            0.979875759,
            0.960575941,
            0.949484052,
            0.941705938,
            0.935725753,
            0.930874086,
            0.926796293,
            0.923281893,
            0.92019579,
            0.917446138]], ['year', 'electricity consumption']).T # progression of electricity consumption per kg CO2 [kWh/kg]
        DAC_elec_0 = 1.132 # kWh/kg
        DAC_sorbent_progression = pd.DataFrame([[2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070], 
                                                [0.003,
            0.002543769,
            0.002491854,
            0.002462625,
            0.002442392,
            0.002426984,
            0.002414578,
            0.002404216,
            0.002395334,
            0.002387571,
            0.002380684]], ['year', 'sorbent consumption']).T # progression of sorbent consumption per kg CO2 [kg/kg]
        DAC_sorbent_0 = 0.003 # kg/kg
        
    if low_mid_high == 'high':
        DAC_elec_progression = pd.DataFrame([[2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070], 
                                             [0.5,
            0.375296878,
            0.361106835,
            0.353117455,
            0.347587132,
            0.343375604,
            0.339984662,
            0.337152492,
            0.334724728,
            0.332602842,
            0.330720169]], ['year', 'electricity consumption']).T # progression of electricity consumption per kg CO2 [kWh/kg]
        DAC_elec_0 = 0.7 # kWh/kg
        DAC_sorbent_progression = pd.DataFrame([[2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070], 
                                                [0.003,
            0.001975451,
            0.001904487,
            0.001867912,
            0.001843965,
            0.00182646,
            0.001812817,
            0.001801725,
            0.001792434,
            0.001784476,
            0.00177754]], ['year', 'sorbent consumption']).T # progression of sorbent consumption per kg CO2 [kg/kg]
        DAC_sorbent_0 = 0.003 # kg/kg
    
    return DAC_elec_progression, DAC_elec_0, DAC_sorbent_progression, DAC_sorbent_0
# define technological performance of Fischer-Trophsch plants
def define_FT_performance(low_mid_high, y_plants_start, y_plants_end): 
    # utility variables for fuel demand
    h2_energy_density = 120 # MJ/kg
    saf_energy_density = 45 # MJ/kg
    
    if low_mid_high == 'low':  
        FT_elec = np.array([0.02]*(y_plants_end - y_plants_start + 1)) # electricity consumption per MJ of fuel producted [kWh/MJ]
        FT_co2 = np.array([4.212/saf_energy_density]*(y_plants_end - y_plants_start + 1)) # CO2 (from DAC) consumption per MJ of fuel producted [kg/MJ]
        FT_h2 = np.array([2.3]*(y_plants_end - y_plants_start + 1)) # H2 consumption per MJ of fuel producted [MJ/MJ]
        FT_co2_emissions = np.array([1.108/saf_energy_density]*(y_plants_end - y_plants_start + 1)) # CO2 emissions (CO2 from DAC not used for fuel) per MJ of fuel produced [kg/MJ]
    
    if low_mid_high == 'mid':  
        FT_elec = np.array([0.02]*(y_plants_end - y_plants_start + 1)) # electricity consumption per MJ of fuel producted [kWh/MJ]
        FT_co2 = np.array([3.880/saf_energy_density]*(y_plants_end - y_plants_start + 1)) # CO2 (from DAC) consumption per MJ of fuel producted [kg/MJ]
        FT_h2 = np.array([1.59]*(y_plants_end - y_plants_start + 1)) # H2 consumption per MJ of fuel producted [MJ/MJ]
        FT_co2_emissions = np.array([0.776/saf_energy_density]*(y_plants_end - y_plants_start + 1)) # CO2 emissions (CO2 from DAC not used for fuel) per MJ of fuel produced [kg/MJ]
    
    if low_mid_high == 'high':  
        FT_elec = np.array([0.02]*(y_plants_end - y_plants_start + 1)) # electricity consumption per MJ of fuel producted [kWh/MJ]
        FT_co2 = np.array([3.422/saf_energy_density]*(y_plants_end - y_plants_start + 1)) # CO2 (from DAC) consumption per MJ of fuel producted [kg/MJ]
        FT_h2 = np.array([1.37]*(y_plants_end - y_plants_start + 1)) # H2 consumption per MJ of fuel producted [MJ/MJ]
        FT_co2_emissions = np.array([0.318/saf_energy_density]*(y_plants_end - y_plants_start + 1)) # CO2 emissions (CO2 from DAC not used for fuel) per MJ of fuel produced [kg/MJ]
    
    return FT_elec, FT_co2, FT_h2, FT_co2_emissions
# define technological performance of H2 liquefaction plants
def define_LIQ_performance(low_mid_high, y_plants_start, y_plants_end):     
    # utility variables for fuel demand
    h2_energy_density = 120 # MJ/kg
    
    if low_mid_high == 'low':
        LIQ_elec_progression = pd.DataFrame([[2020, 2050], np.array([15, 15])/h2_energy_density], ['year', 'electricity consumption']).T # progression of electricity consumption per MJ H2 [kWh/MJH2]
        LTQ_elec_0 = 15/h2_energy_density
        LIQ_h2_escaped = np.array([0.01]*(y_plants_end - y_plants_start + 1)) # hydrogen escaped due to boil-off [MJ/MJ]
    
    if low_mid_high == 'mid':
        LIQ_elec_progression = pd.DataFrame([[2020, 2030], np.array([10.4, 6])/h2_energy_density], ['year', 'electricity consumption']).T # progression of electricity consumption per MJ H2 [kWh/MJH2]
        LTQ_elec_0 = 10.4/h2_energy_density
        LIQ_h2_escaped = np.array([0.01]*(y_plants_end - y_plants_start + 1)) # hydrogen escaped due to boil-off [MJ/MJ]
        
    if low_mid_high == 'high':
        LIQ_elec_progression = pd.DataFrame([[2020, 2050], np.array([6, 6])/h2_energy_density], ['year', 'electricity consumption']).T # progression of electricity consumption per MJ H2 [kWh/MJH2]
        LTQ_elec_0 = 6/h2_energy_density
        LIQ_h2_escaped = np.array([0.01]*(y_plants_end - y_plants_start + 1)) # hydrogen escaped due to boil-off [MJ/MJ]
    
    return LIQ_elec_progression, LTQ_elec_0, LIQ_h2_escaped
# define technological performance of future hydrocarbon-powered aircraft
def define_AC_performance(tech_improvements):
    # values are relative to the performance of the initial generation of aircraft
    if tech_improvements == 'low': # BAU scenario from Cox et al. 2018
        NB_upcoming_improvement = 0.87
        NB_future_improvement = 0.78
        WB_upcoming_improvement = 0.87
        WB_future_improvement = 0.78
        
    if tech_improvements == 'mid': # based on evolutionary estimates from Grewe et al. 2021
        NB_upcoming_improvement = 0.78
        NB_future_improvement = 0.62
        WB_upcoming_improvement = 0.82
        WB_future_improvement = 0.66
            
    if tech_improvements == 'high': # based on OPT scenario from Cox et al. 2018
        NB_upcoming_improvement = 0.7
        NB_future_improvement = 0.5
        WB_upcoming_improvement = 0.7
        WB_future_improvement = 0.5
    
    return NB_upcoming_improvement, NB_future_improvement, WB_upcoming_improvement, WB_future_improvement
# define technological performance of future H2-powered aircraft
def define_HC_performance(tech_improvements):
    # values are relative to the performance of the initial generation of aircraft
    if tech_improvements == 'low': # low progress from LTAG
        NB_upcoming_H2_change = 1.2
        NB_future_H2_change = 1.2
        WB_future_H2_change = 1.4
        
    if tech_improvements == 'mid': # medium progress from LTAG
        NB_upcoming_H2_change = 1.15
        NB_future_H2_change = 1.15
        WB_future_H2_change = 1
            
    if tech_improvements == 'high': # higher progress from LTAG
        NB_upcoming_H2_change = 0.95
        NB_future_H2_change = 0.95
        WB_future_H2_change = 0.9
    
    return NB_upcoming_H2_change, NB_future_H2_change, WB_future_H2_change

#%% some functions used when running a large number of scenarios
def scenario_name_generator(pathway, hydrogen_source, growth, aircraft_tech, lh2_tech, fuel_tech, capacity, e_fuel, hydrogen):
    scenario_name = pathway+' pathway; hydrogen from '+hydrogen_source+'; '+growth+'; '+aircraft_tech+' aircraft development ('+lh2_tech+' relative development of LH$_2$ aircraft); '+capacity*'capacity improved; '+'AAF with '+e_fuel+'; '+hydrogen*'LH$_2$ aircraft implemented; '+fuel_tech+' fuel technology development.'
    return scenario_name

def retrieve_scenario_results(scenario_name, all_scenario_names, all_scenario_results):
    return all_scenario_results[list(all_scenario_names).index(scenario_name)]
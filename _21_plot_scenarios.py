import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from copy import deepcopy
import seaborn as sns

from _10_functions import *
from _11_define_scenarios import * 
from _12_LWE_function import *

#%% function used to process scenario results by determining LCIA results per MJ fuel
def calculate_impact_per_mj(total_fuel, list_of_dic, impact_cat):
    fuel_impacts = extract_processes(list_of_dic, impact_cat, list(range(2,len(list_of_dic)))) # 0 and 1 are excluded, since that are aircraft impacts
    sum_of_impacts = fuel_impacts.sum(axis = 1)
    sum_of_fuel = total_fuel.sum(axis = 1)
    
    total_impact_mj = sum_of_impacts/sum_of_fuel
    
    return total_impact_mj

#%% functions for plotting the various heatmaps
# make dataframes with values for heatmaps
def make_target_heat_map(year_of_interest, flight_start_year, flight_end_year, all_scenario_names, all_scenario_results):
    heat_map_columns = ['no ReFuelEU with \n no H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n no H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n low-performance H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n mid-performance H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n high-performance H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n no H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n low-performance H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n mid-performance H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n high-performance H$\mathregular{_2}$ AC'] # variables 
    heat_map_column_characteristics = [['no ReFuelEU', False, 'low'], 
                                       ['ReFuelEU base', False, 'low'], ['ReFuelEU base', True, 'low'], ['ReFuelEU base', True, 'mid'], ['ReFuelEU base', True, 'high'],
                                       ['ReFuelEU extended', False, 'low'], ['ReFuelEU extended', True, 'low'], ['ReFuelEU extended', True, 'mid'], ['ReFuelEU extended', True, 'high']]
    heat_map_rows = []
    heat_map_row_characteristics = []
    ac_level = ['low', 'mid', 'high']
    ac_level_name = ['business-as-usual', 'optimistic', 'breakthrough']
    for growth in ['high growth', 'base growth', 'low growth', 'degrowth']:
        for ops_imp in [False]: #[False, True]:
            for i in range(len(ac_level)):
                for aaf_level in ['mid']: #['low', 'mid', 'high']:
                    heat_map_rows.append(ac_level_name[i]+' HC AC \n performance with '+growth)
                    heat_map_row_characteristics.append([ac_level[i], aaf_level, ops_imp, growth])
            
    heat_map_budget = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_just_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_excl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_incl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    
    # global variables across heat map
    background = '1.7C'
    hydrogen_source = 'grid'
    # target: make lines to plot ICAO/IATA target
    years = list(range(flight_start_year,flight_end_year + 1))
    max_line = []
    for year in years:
        corisa_baseline = 127.5e9 # running the model results in 150 Mton CO2 in 2019; new CORSIA baseline is 85% of 2019 (note: EASA cites 147 Mtons in 2019).
        if year <= 2035: max_line.append(corisa_baseline) 
        elif year <= 2050: max_line.append(corisa_baseline*(2050-year)/(2050-2035))
        else: max_line.append(0.0)
            
    max_RF_input = np.zeros([len(years),19])
    max_RF_input[:,0] = years
    max_RF_input[:,12] = max_line
    max_RF_input = pd.DataFrame(max_RF_input)
    RF, RF_low, RF_high = emissions_to_LWE(max_RF_input, years[0], years[-1])
    RF_CO2_target = RF.iloc[:,0] # variable showing RF for each year when using emissions limit
    
    # prime variables to extract results
    time = year_of_interest - flight_start_year
    for i in range(len(heat_map_rows)):
        ac_tech = heat_map_row_characteristics[i][0]
        aaf_tech =  heat_map_row_characteristics[i][1]
        ops_imp = heat_map_row_characteristics[i][2]
        growth =  heat_map_row_characteristics[i][3]
        for j in range(len(heat_map_columns)):
            aaf_impl = heat_map_column_characteristics[j][0]
            h2ac_impl = heat_map_column_characteristics[j][1]
            h2ac_tech = heat_map_column_characteristics[j][2]
            scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, aaf_tech, ops_imp, aaf_impl, h2ac_impl)
            # correct for fact that not all scenarios have been generated
            if scenario_name_here not in list(all_scenario_names): 
                scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, 'low', ops_imp, aaf_impl, h2ac_impl)
            if scenario_name_here not in list(all_scenario_names): 
                budget_result_here = warming_just_co2_result_here = warming_excl_non_co2_result_here = warming_incl_non_co2_result_here = np.nan 
            else:
                scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
                RF_input_here = np.zeros([len(years),19])
                RF_input_here[:,0] = years
                RF_input_here[:,12] = scenario_result_here[1]['Carbon dioxide']
                RF_input_here = pd.DataFrame(RF_input_here)
                RF, RF_low, RF_high = emissions_to_LWE(RF_input_here, years[0], years[-1])
                RF_CO2_here = RF.iloc[:,0] # variable showing RF for each year from just CO2
                RF_excl_non_CO2 = scenario_result_here[1]['Radiative forcing (excl. aviation non-CO$_2$)']
                RF_incl_non_CO2 = scenario_result_here[1]['Radiative forcing']
                budget_result_here = list(RF_CO2_here)[time]/list(RF_CO2_target)[time]*100 # if share of remaining budget: (list(RF_CO2_target)[time] - list(RF_CO2_here)[time])/list(RF_CO2_target)[time]*100
                warming_just_co2_result_here = list(RF_CO2_here)[time]/list(RF_CO2_here)[2050 - flight_start_year]*100
                warming_excl_non_co2_result_here = list(RF_excl_non_CO2)[time]/list(RF_excl_non_CO2)[2050 - flight_start_year]*100
                warming_incl_non_co2_result_here = list(RF_incl_non_CO2)[time]/list(RF_incl_non_CO2)[2050 - flight_start_year]*100
                
            heat_map_budget[i,j]                = budget_result_here
            heat_map_warming_just_co2[i,j]      = warming_just_co2_result_here
            heat_map_warming_excl_non_co2[i,j]  = warming_excl_non_co2_result_here
            heat_map_warming_incl_non_co2[i,j]  = warming_incl_non_co2_result_here
            
    heat_map_budget                 = pd.DataFrame(heat_map_budget, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_just_co2       = pd.DataFrame(heat_map_warming_just_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_excl_non_co2   = pd.DataFrame(heat_map_warming_excl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2   = pd.DataFrame(heat_map_warming_incl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    
    return heat_map_budget, heat_map_warming_just_co2, heat_map_warming_excl_non_co2, heat_map_warming_incl_non_co2
# make dataframes with values for heatmaps of foreground sensitivity analyses
def make_foreground_heat_map(year_of_interest, flight_start_year, flight_end_year, all_scenario_names, all_scenario_results):
    heat_map_columns = []
    heat_map_column_characteristics = []
    aaf_impl = ['no ReFuelEU', 'ReFuelEU base', 'ReFuelEU extended']
    aaf_impl_names = ['no ReFuelEU', 'ReFuelEU as-is', 'ReFuelEU extended']
    h2ac_impl = [False, True]
    h2ac_tech = ['low','mid','high']
    ops_impr = [False, True]
    for i in range(len(aaf_impl)):
        aaf_impl_state = aaf_impl[i]
        if aaf_impl_state == 'no ReFuelEU':
            for ops_impr_state in ops_impr:
                h2ac_impl_state = False
                h2ac_tech_state = 'low'
                heat_map_columns.append(aaf_impl_names[i]+' with '+'no '*(not ops_impr_state)+'op. impr. \n and no H$\mathregular{_2}$ AC')
                heat_map_column_characteristics.append([aaf_impl_state, h2ac_impl_state, h2ac_tech_state, ops_impr_state])
        else:
            for h2ac_impl_state in h2ac_impl:
                if h2ac_impl_state == False:
                    for ops_impr_state in ops_impr:
                        h2ac_tech_state = 'low'
                        heat_map_columns.append(aaf_impl_names[i]+' with '+'no '*(not ops_impr_state)+'op. impr. \n and no H$\mathregular{_2}$ AC')
                        heat_map_column_characteristics.append([aaf_impl_state, h2ac_impl_state, h2ac_tech_state, ops_impr_state])
                else:
                    for h2ac_tech_state in h2ac_tech:
                        for ops_impr_state in ops_impr:
                            heat_map_columns.append(aaf_impl_names[i]+' with '+'no '*(not ops_impr_state)+'op. impr. \n and '+h2ac_tech_state+'-performance H$\mathregular{_2}$ AC')
                            heat_map_column_characteristics.append([aaf_impl_state, h2ac_impl_state, h2ac_tech_state, ops_impr_state])    
    
    heat_map_rows = []
    heat_map_row_characteristics = []
    ac_level = ['low', 'mid', 'high']
    ac_level_name = ['business-as-usual', 'optimistic', 'breakthrough']
    aaf_level = ['low', 'mid', 'high']
    aaf_level_name = ['worst-case','base-case','best-case']
    for growth in ['high growth', 'base growth', 'low growth', 'degrowth']:
        for i in range(len(ac_level)):
            for j in range(len(aaf_level)):
                heat_map_rows.append(ac_level_name[i]+' HC AC performance with \n'+aaf_level_name[j]+' AAF performance and '+growth)
                heat_map_row_characteristics.append([ac_level[i], aaf_level[j], growth])
            
    heat_map_budget = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_just_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_excl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_incl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    
    # global variables across heat map
    background = '1.7C'
    hydrogen_source = 'grid'
    # target: make lines to plot ICAO/IATA target
    years = list(range(flight_start_year,flight_end_year + 1))
    max_line = []
    for year in years:
        corisa_baseline = 127.5e9 # running the model results in 150 Mton CO2 in 2019; new CORSIA baseline is 85% of 2019 (note: EASA cites 147 Mtons in 2019).
        if year <= 2035: max_line.append(corisa_baseline) 
        elif year <= 2050: max_line.append(corisa_baseline*(2050-year)/(2050-2035))
        else: max_line.append(0.0)
            
    max_RF_input = np.zeros([len(years),19])
    max_RF_input[:,0] = years
    max_RF_input[:,12] = max_line
    max_RF_input = pd.DataFrame(max_RF_input)
    RF, RF_low, RF_high = emissions_to_LWE(max_RF_input, years[0], years[-1])
    RF_CO2_target = RF.iloc[:,0] # variable showing RF for each year when using emissions limit
    
    # prime variables to extract results
    time = year_of_interest - flight_start_year
    for i in range(len(heat_map_rows)):
        ac_tech = heat_map_row_characteristics[i][0]
        aaf_tech =  heat_map_row_characteristics[i][1]
        growth =  heat_map_row_characteristics[i][2]
        for j in range(len(heat_map_columns)):
            aaf_impl = heat_map_column_characteristics[j][0]
            h2ac_impl = heat_map_column_characteristics[j][1]
            h2ac_tech = heat_map_column_characteristics[j][2]
            ops_imp = heat_map_column_characteristics[j][3]
            scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, aaf_tech, ops_imp, aaf_impl, h2ac_impl)
            # correct for fact that not all scenarios have been generated
            # if scenario_name_here not in list(all_scenario_names): 
            #     scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, 'low', ops_imp, aaf_impl, h2ac_impl)
            if scenario_name_here not in list(all_scenario_names): 
                budget_result_here = warming_just_co2_result_here = warming_excl_non_co2_result_here = warming_incl_non_co2_result_here = np.nan 
            else:
                scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
                RF_input_here = np.zeros([len(years),19])
                RF_input_here[:,0] = years
                RF_input_here[:,12] = scenario_result_here[1]['Carbon dioxide']
                RF_input_here = pd.DataFrame(RF_input_here)
                RF, RF_low, RF_high = emissions_to_LWE(RF_input_here, years[0], years[-1])
                RF_CO2_here = RF.iloc[:,0] # variable showing RF for each year from just CO2
                RF_excl_non_CO2 = scenario_result_here[1]['Radiative forcing (excl. aviation non-CO$_2$)']
                RF_incl_non_CO2 = scenario_result_here[1]['Radiative forcing']
                budget_result_here = list(RF_CO2_here)[time]/list(RF_CO2_target)[time]*100 # if share of remaining budget: (list(RF_CO2_target)[time] - list(RF_CO2_here)[time])/list(RF_CO2_target)[time]*100
                warming_just_co2_result_here = list(RF_CO2_here)[time]/list(RF_CO2_here)[2050 - flight_start_year]*100
                warming_excl_non_co2_result_here = list(RF_excl_non_CO2)[time]/list(RF_excl_non_CO2)[2050 - flight_start_year]*100
                warming_incl_non_co2_result_here = list(RF_incl_non_CO2)[time]/list(RF_incl_non_CO2)[2050 - flight_start_year]*100
                
            heat_map_budget[i,j]                = budget_result_here
            heat_map_warming_just_co2[i,j]      = warming_just_co2_result_here
            heat_map_warming_excl_non_co2[i,j]  = warming_excl_non_co2_result_here
            heat_map_warming_incl_non_co2[i,j]  = warming_incl_non_co2_result_here
            
    heat_map_budget                 = pd.DataFrame(heat_map_budget, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_just_co2       = pd.DataFrame(heat_map_warming_just_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_excl_non_co2   = pd.DataFrame(heat_map_warming_excl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2   = pd.DataFrame(heat_map_warming_incl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    
    return heat_map_budget, heat_map_warming_just_co2, heat_map_warming_excl_non_co2, heat_map_warming_incl_non_co2
# make dataframes with values for heatmaps of background sensitivity analyses
def make_background_heat_map(year_of_interest, flight_start_year, flight_end_year, all_scenario_names, all_scenario_results, background, hydrogen_source):
    heat_map_columns = ['no ReFuelEU with \n no H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n no H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n low-performance H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n mid-performance H$\mathregular{_2}$ AC', 'ReFuelEU as-is with \n high-performance H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n no H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n low-performance H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n mid-performance H$\mathregular{_2}$ AC', 'ReFuelEU extended with \n high-performance H$\mathregular{_2}$ AC'] # variables 
    heat_map_column_characteristics = [['no ReFuelEU', False, 'low'], 
                                       ['ReFuelEU base', False, 'low'], ['ReFuelEU base', True, 'low'], ['ReFuelEU base', True, 'mid'], ['ReFuelEU base', True, 'high'],
                                       ['ReFuelEU extended', False, 'low'], ['ReFuelEU extended', True, 'low'], ['ReFuelEU extended', True, 'mid'], ['ReFuelEU extended', True, 'high']]
    heat_map_rows = []
    heat_map_row_characteristics = []
    ac_level = ['low', 'mid', 'high']
    ac_level_name = ['business-as-usual', 'optimistic', 'breakthrough']
    for growth in ['high growth', 'base growth', 'low growth', 'degrowth']:
        for ops_imp in [False]: #[False, True]:
            for i in range(len(ac_level)):
                for aaf_level in ['mid']: #['low', 'mid', 'high']:
                    heat_map_rows.append(ac_level_name[i]+' HC AC \n performance with '+growth)
                    heat_map_row_characteristics.append([ac_level[i], aaf_level, ops_imp, growth])
            
    heat_map_budget = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_just_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_excl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_incl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    
    # target: make lines to plot ICAO/IATA target
    years = list(range(flight_start_year,flight_end_year + 1))
    max_line = []
    for year in years:
        corisa_baseline = 127.5e9 # running the model results in 150 Mton CO2 in 2019; new CORSIA baseline is 85% of 2019 (note: EASA cites 147 Mtons in 2019).
        if year <= 2035: max_line.append(corisa_baseline) 
        elif year <= 2050: max_line.append(corisa_baseline*(2050-year)/(2050-2035))
        else: max_line.append(0.0)
            
    max_RF_input = np.zeros([len(years),19])
    max_RF_input[:,0] = years
    max_RF_input[:,12] = max_line
    max_RF_input = pd.DataFrame(max_RF_input)
    RF, RF_low, RF_high = emissions_to_LWE(max_RF_input, years[0], years[-1])
    RF_CO2_target = RF.iloc[:,0] # variable showing RF for each year when using emissions limit
    
    # prime variables to extract results
    time = year_of_interest - flight_start_year
    for i in range(len(heat_map_rows)):
        ac_tech = heat_map_row_characteristics[i][0]
        aaf_tech =  heat_map_row_characteristics[i][1]
        ops_imp = heat_map_row_characteristics[i][2]
        growth =  heat_map_row_characteristics[i][3]
        for j in range(len(heat_map_columns)):
            aaf_impl = heat_map_column_characteristics[j][0]
            h2ac_impl = heat_map_column_characteristics[j][1]
            h2ac_tech = heat_map_column_characteristics[j][2]
            scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, aaf_tech, ops_imp, aaf_impl, h2ac_impl)
            # correct for fact that not all scenarios have been generated
            if scenario_name_here not in list(all_scenario_names): 
                scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, 'low', ops_imp, aaf_impl, h2ac_impl)
            if scenario_name_here not in list(all_scenario_names): 
                budget_result_here = warming_just_co2_result_here = warming_excl_non_co2_result_here = warming_incl_non_co2_result_here = np.nan 
            else:
                scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
                RF_input_here = np.zeros([len(years),19])
                RF_input_here[:,0] = years
                RF_input_here[:,12] = scenario_result_here[1]['Carbon dioxide']
                RF_input_here = pd.DataFrame(RF_input_here)
                RF, RF_low, RF_high = emissions_to_LWE(RF_input_here, years[0], years[-1])
                RF_CO2_here = RF.iloc[:,0] # variable showing RF for each year from just CO2
                RF_excl_non_CO2 = scenario_result_here[1]['Radiative forcing (excl. aviation non-CO$_2$)']
                RF_incl_non_CO2 = scenario_result_here[1]['Radiative forcing']
                budget_result_here = list(RF_CO2_here)[time]/list(RF_CO2_target)[time]*100 # if share of remaining budget: (list(RF_CO2_target)[time] - list(RF_CO2_here)[time])/list(RF_CO2_target)[time]*100
                warming_just_co2_result_here = list(RF_CO2_here)[time]/list(RF_CO2_here)[2050 - flight_start_year]*100
                warming_excl_non_co2_result_here = list(RF_excl_non_CO2)[time]/list(RF_excl_non_CO2)[2050 - flight_start_year]*100
                warming_incl_non_co2_result_here = list(RF_incl_non_CO2)[time]/list(RF_incl_non_CO2)[2050 - flight_start_year]*100
                
            heat_map_budget[i,j]                = budget_result_here
            heat_map_warming_just_co2[i,j]      = warming_just_co2_result_here
            heat_map_warming_excl_non_co2[i,j]  = warming_excl_non_co2_result_here
            heat_map_warming_incl_non_co2[i,j]  = warming_incl_non_co2_result_here
            
    heat_map_budget                 = pd.DataFrame(heat_map_budget, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_just_co2       = pd.DataFrame(heat_map_warming_just_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_excl_non_co2   = pd.DataFrame(heat_map_warming_excl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2   = pd.DataFrame(heat_map_warming_incl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    
    return heat_map_budget, heat_map_warming_just_co2, heat_map_warming_excl_non_co2, heat_map_warming_incl_non_co2
# plot dataframes to heatmaps
def plot_target_heat_map(heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4, folder):
    fig, ((ax1, ax2, axcb), (ax3, ax4, axcb2)) = plt.subplots(2,3,gridspec_kw={'width_ratios':[1,1,0.08]},figsize = (11,13))
    
    fig1 = sns.heatmap(heat_map_df_1, 
                       mask = heat_map_df_1 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = False, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax1, cbar=False, cmap = 'seismic')
    fig1 = sns.heatmap(heat_map_df_1, 
                       mask = heat_map_df_1 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = False, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax1, cbar=False, cmap = 'seismic')
    fig2 = sns.heatmap(heat_map_df_2, 
                       mask = heat_map_df_2 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = False, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax2, cbar=False, cmap = 'seismic')
    fig2 = sns.heatmap(heat_map_df_2, 
                       mask = heat_map_df_2 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = False, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax2, cbar=False, cmap = 'seismic')
    fig3 = sns.heatmap(heat_map_df_3, 
                       mask = heat_map_df_3 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax3, cbar=False, cmap = 'seismic')
    fig3 = sns.heatmap(heat_map_df_3, 
                       mask = heat_map_df_3 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax3, cbar=False, cmap = 'seismic')
    fig4 = sns.heatmap(heat_map_df_4, 
                       mask = heat_map_df_4 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = True, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax4, cbar_ax=axcb, cmap = 'seismic')
    fig4 = sns.heatmap(heat_map_df_4, 
                       mask = heat_map_df_4 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = True, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax4, cbar_ax=axcb, cmap = 'seismic')
    
    axcb2.axis('off')
    
    ax1.set_title('(a) Share of CO$\mathregular{_2}$ RF budget used by 2070 \n based on ICAO/IATA limit [%]', fontweight="bold")
    ax2.set_title('(b) RF in 2070 compared to 2050, \n CO$\mathregular{_2}$ only [%]', fontweight="bold")
    ax3.set_title('(c) RF in 2070 compared to 2050, \n excl. flight non-CO$\mathregular{_2}$ [%]', fontweight="bold")
    ax4.set_title('(d) RF in 2070 compared to 2050, \n incl. flight non-CO$\mathregular{_2}$ [%]', fontweight="bold")
    
    fig.tight_layout()
    fig_name = folder+'Sfig5_heatmap.pdf'
    fig.savefig(fig_name,bbox_inches='tight')#, dpi=600)
    fig_name = folder+'Sfig5_heatmap.png'
    fig.savefig(fig_name,bbox_inches='tight', dpi=600)   
# plot dataframes for foreground sensitivity heatmaps
def plot_foreground_heat_map(heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4, folder):
    fig1, (ax1, axcb1) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    fig2, (ax2, axcb2) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    fig3, (ax3, axcb3) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    fig4, (ax4, axcb4) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    figs = [fig1, fig2, fig3, fig4]
    
    sns.heatmap(heat_map_df_1, 
                mask = heat_map_df_1 < 100,
                annot = True, annot_kws={"alpha": 0.5},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax1, cbar=False, cmap = 'seismic')
    sns.heatmap(heat_map_df_1, 
                mask = heat_map_df_1 >= 100,
                annot = True, annot_kws={"weight": "bold"},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax1, cbar_ax=axcb1, cmap = 'seismic')
    sns.heatmap(heat_map_df_2, 
                mask = heat_map_df_2 < 100,
                annot = True, annot_kws={"alpha": 0.5},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax2, cbar=False, cmap = 'seismic')
    sns.heatmap(heat_map_df_2, 
                mask = heat_map_df_2 >= 100,
                annot = True, annot_kws={"weight": "bold"},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax2, cbar_ax=axcb2, cmap = 'seismic')
    sns.heatmap(heat_map_df_3, 
                mask = heat_map_df_3 < 100,
                annot = True, annot_kws={"alpha": 0.5},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax3, cbar=False, cmap = 'seismic')
    sns.heatmap(heat_map_df_3, 
                mask = heat_map_df_3 >= 100,
                annot = True, annot_kws={"weight": "bold"},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax3, cbar_ax=axcb3, cmap = 'seismic')
    sns.heatmap(heat_map_df_4, 
                mask = heat_map_df_4 < 100,
                annot = True, annot_kws={"alpha": 0.5},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax4, cbar=False, cmap = 'seismic')
    sns.heatmap(heat_map_df_4, 
                mask = heat_map_df_4 >= 100,
                annot = True, annot_kws={"weight": "bold"},
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax4, cbar_ax=axcb4, cmap = 'seismic')
    
    fig_names = [folder+'Sfig9_heatmap_sensitivity_budget', folder+'Xfig_heatmap_sensitivity_trend_CO2', folder+'Xfig_heatmap_sensitivity_trend_excl_nonCO2', folder+'Xfig_heatmap_sensitivity_trend_all']
    for i in range(len(fig_names)):
        figs[i].tight_layout()
        figs[i].savefig(fig_names[i]+'.pdf',bbox_inches='tight')#, dpi=600)
    for i in range(len(fig_names)):
        figs[i].tight_layout()
        figs[i].savefig(fig_names[i]+'.png',bbox_inches='tight', dpi=600)  
# plot dataframes for background sensitivity heatmaps
def plot_background_heat_map(heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4, folder, background, hydrogen_source):
    #fig, (ax1, ax2, ax3, ax4, axcb) = plt.subplots(1,5,gridspec_kw={'width_ratios':[1,1,1,1, 0.08]},figsize = (20,8))
    fig, ((ax1, ax2, axcb), (ax3, ax4, axcb2)) = plt.subplots(2,3,gridspec_kw={'width_ratios':[1,1,0.08]},figsize = (11,13))
    
    fig1 = sns.heatmap(heat_map_df_1, 
                       mask = heat_map_df_1 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = False, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax1, cbar=False, cmap = 'seismic')
    fig1 = sns.heatmap(heat_map_df_1, 
                       mask = heat_map_df_1 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = False, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax1, cbar=False, cmap = 'seismic')
    fig2 = sns.heatmap(heat_map_df_2, 
                       mask = heat_map_df_2 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = False, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax2, cbar=False, cmap = 'seismic')
    fig2 = sns.heatmap(heat_map_df_2, 
                       mask = heat_map_df_2 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = False, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax2, cbar=False, cmap = 'seismic')
    fig3 = sns.heatmap(heat_map_df_3, 
                       mask = heat_map_df_3 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax3, cbar=False, cmap = 'seismic')
    fig3 = sns.heatmap(heat_map_df_3, 
                       mask = heat_map_df_3 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax3, cbar=False, cmap = 'seismic')
    fig4 = sns.heatmap(heat_map_df_4, 
                       mask = heat_map_df_4 < 100,
                       annot = True, annot_kws={"alpha": 0.5},
                       xticklabels = True, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax4, cbar_ax=axcb, cmap = 'seismic')
    fig4 = sns.heatmap(heat_map_df_4, 
                       mask = heat_map_df_4 >= 100,
                       annot = True, annot_kws={"weight": "bold"},
                       xticklabels = True, yticklabels = False, center = 100, vmin=0, vmax=200, fmt=".0f", ax = ax4, cbar_ax=axcb, cmap = 'seismic')
    
    axcb2.axis('off')
    
    ax1.set_title('(a) Share of CO$\mathregular{_2}$ RF budget used by 2070 \n based on ICAO/IATA limit [%]', fontweight="bold")
    ax2.set_title('(b) RF in 2070 compared to 2050, \n CO$\mathregular{_2}$ only [%]', fontweight="bold")
    ax3.set_title('(c) RF in 2070 compared to 2050, \n excl. flight non-CO$\mathregular{_2}$ [%]', fontweight="bold")
    ax4.set_title('(d) RF in 2070 compared to 2050, \n incl. flight non-CO$\mathregular{_2}$ [%]', fontweight="bold")
    
    fig.tight_layout()
    heatmap_name = folder+'Xfig_heatmap_'+background+hydrogen_source
    if background=='1.7C' and hydrogen_source=='market': heatmap_name = folder+'Sfig7_heatmap_'+background+hydrogen_source
    fig_name = heatmap_name+'.pdf'
    fig.savefig(fig_name,bbox_inches='tight')#, dpi=600)
    fig_name = heatmap_name+'.png'
    fig.savefig(fig_name,bbox_inches='tight', dpi=600)  
    
#%% function for plotting all figures which have year on the x-axis
def plot_timeline_results(flight_start_year,flight_end_year, folder, all_scenario_names, all_scenario_results, folder_csvs):
    years_graph = list(range(flight_start_year,flight_end_year + 1))
    LU_red = '#be1908'
    LU_orange = '#f46e32'
    LU_turqoise = '#34a3a9'
    LU_light_blue = '#5cb1eb'
    LU_violet = '#b02079'
    LU_green = '#2c712d'
    LU_blue = '#001158'
    
    # choose scenarios to plot
    fuel_scenario_names = ['no ReFuelEU, \n no H$\mathregular{_2}$ AC', 
                            'ReFuelEU extended, \n no H$\mathregular{_2}$ AC',  
                            'ReFuelEU extended, \n with H$\mathregular{_2}$ AC']
    fuel_scenarios = [['no ReFuelEU', False, 'low'],
                      ['ReFuelEU extended', False, 'mid'],
                      ['ReFuelEU extended', True, 'mid']]
    h2ac_scenarios = ['high','mid','low']
    
    ac_scenario_names = ['breakthrough AC technology \n with high growth',
                         'optimistic AC technology \n with low growth',
                         'business-as-usual AC technology \n with degrowth']
    ac_scenarios = [['high',False,'high growth'],
                    ['mid',False,'low growth'],
                    ['low',False,'degrowth']]
    
    # global variables
    background = '1.7C'
    hydrogen_source = 'grid'
    # target: make lines to plot ICAO/IATA target
    years = list(range(flight_start_year,flight_end_year + 1))
    max_line = []
    for year in years:
        corisa_baseline = 127.5e9 # running the model results in 150 Mton CO2 in 2019; new CORSIA baseline is 85% of 2019 (note: EASA cites 147 Mtons in 2019).
        if year <= 2035: max_line.append(corisa_baseline) 
        elif year <= 2050: max_line.append(corisa_baseline*(2050-year)/(2050-2035))
        else: max_line.append(0.0)
            
    max_RF_input = np.zeros([len(years),19])
    max_RF_input[:,0] = years
    max_RF_input[:,12] = max_line
    max_RF_input = pd.DataFrame(max_RF_input)
    RF, RF_low, RF_high = emissions_to_LWE(max_RF_input, years[0], years[-1])
    RF_CO2_target = RF.iloc[:,0] # variable showing RF for each year when using emissions limit
    
    # generate results for chosen scenarios
    scenarios_list_of_dic_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_co2_per_year_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_co2_RF_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_cc_per_year_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_total_RF_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_RF_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_RF_low_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_RF_high_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_fossil_kerosene_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_hydrogen_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_fleet_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_fleet_results_max = 0
    scenarios_aaf_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_LCIA_df_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_rpk_per_aircraft = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    scenarios_fuels_results = [[0 for i in range(len(fuel_scenarios))] for j in range(len(ac_scenarios))]
    for i in range(len(ac_scenarios)):
        ac_tech = ac_scenarios[i][0]
        ops_imp = ac_scenarios[i][1]
        growth =  ac_scenarios[i][2]
        for j in range(len(fuel_scenarios)):
            aaf_impl = fuel_scenarios[j][0]
            h2ac_impl = fuel_scenarios[j][1]
            aaf_tech =  fuel_scenarios[j][2]
            h2ac_tech = 'low'
            if h2ac_impl == True: h2ac_tech = h2ac_scenarios[i]
            scenario_name_here = scenario_name_generator(background, hydrogen_source, growth, ac_tech, h2ac_tech, aaf_tech, ops_imp, aaf_impl, h2ac_impl)
            scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
            RF_input_here = np.zeros([len(years),19])
            RF_input_here[:,0] = years
            RF_input_here[:,12] = scenario_result_here[1]['Carbon dioxide']
            RF_input_here = pd.DataFrame(RF_input_here)
            RF, RF_low, RF_high = emissions_to_LWE(RF_input_here, years[0], years[-1])
            RF_CO2_here = RF.iloc[:,0] # variable showing RF for each year from just CO2
            RF_excl_non_CO2 = scenario_result_here[1]['Radiative forcing (excl. aviation non-CO$_2$)']
            RF_incl_non_CO2 = scenario_result_here[1]['Radiative forcing']
            
            scenarios_list_of_dic_results[i][j]     = scenario_result_here[0]
            scenarios_co2_per_year_results[i][j]    = scenario_result_here[1]['Carbon dioxide']/1e9
            scenarios_co2_RF_results[i][j]          = RF_CO2_here
            scenarios_cc_per_year_results[i][j]     = scenario_result_here[1]['climate change']/1e9
            scenarios_total_RF_results[i][j]        = scenario_result_here[1]['Radiative forcing']
            scenarios_LCIA_df_results[i][j]         = scenario_result_here[1]
            scenarios_RF_results[i][j]              = scenario_result_here[2]
            scenarios_RF_low_results[i][j]          = scenario_result_here[3] 
            scenarios_RF_high_results[i][j]         = scenario_result_here[4] 
            scenarios_fossil_kerosene_results[i][j] = scenario_result_here[8]['Fossil kerosene']/1e12
            scenarios_rpk_per_aircraft[i][j]        = scenario_result_here[13]
            scenarios_hydrogen_results[i][j]        = scenario_result_here[19].sum(axis=1)/1e12
            fleet_results_temp                      = scenario_result_here[6].copy()
            fleet_results_temp['gen. 0']            = fleet_results_temp['Original NB'] + fleet_results_temp['Original WB']
            fleet_results_temp['gen. 1']            = fleet_results_temp['New NB'] + fleet_results_temp['New WB']
            fleet_results_temp['gen. 2']            = fleet_results_temp['Upcoming NB'] + fleet_results_temp['Upcoming WB']
            fleet_results_temp['gen. 2 (H$\mathregular{_2}$)'] = fleet_results_temp['H$_2$ 1$^{st}$ gen. NB']
            fleet_results_temp['gen. 3']            = fleet_results_temp['Future NB'] + fleet_results_temp['Future WB']
            fleet_results_temp['gen. 3 (H$\mathregular{_2}$)'] = fleet_results_temp['H$_2$ 2$^{nd}$ gen. NB'] + fleet_results_temp['H$_2$ 1$^{st}$ gen. WB']
            fleet_results_temp.drop(['Original NB', 'Original WB', 'New NB', 'New WB', 'Upcoming NB', 'Upcoming WB', 'Future NB', 'Future WB', 'H$_2$ 1$^{st}$ gen. NB', 'H$_2$ 2$^{nd}$ gen. NB', 'H$_2$ 1$^{st}$ gen. WB'], axis=1, inplace=True)
            scenarios_fleet_results[i][j]           = fleet_results_temp
            if fleet_results_temp.sum(axis = 1).max() > scenarios_fleet_results_max: scenarios_fleet_results_max = fleet_results_temp.sum(axis = 1).max()
            scenarios_fossil_kerosene_results[i][j] = scenario_result_here[8]['Fossil kerosene']/1e12
            scenarios_aaf_results[i][j]             = scenario_result_here[8]['E-fuel']/1e12 + scenario_result_here[8]['Hydrogen fuel']/1e12
            scenarios_fuels_results[i][j]           = scenario_result_here[8]
    
    linestyles = ['--','-.','-']
    colours = [LU_turqoise, LU_orange, LU_red, LU_light_blue, LU_green, LU_violet]
    colours_ac_fleet = [LU_turqoise, LU_orange, LU_red, LU_red, LU_light_blue, LU_light_blue]
    style_list = []
    for line in linestyles:
        for colour in colours:
            style_list.append([line,colour])
    alpha = 1
    
    # first figure: CO2 & RF results
    fig1_name = 'Fig1_results_impacts'
    fig, (ax1, ax2) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,1]},figsize = (8,6))
    ax1.plot(years_graph, np.array(max_line)/1e9, linestyle=':', color='#000000', lw = 2, alpha = 1)
    ax2.plot(years_graph, RF_CO2_target, linestyle=':', color='#000000', lw = 2, alpha = 1)
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            ax1.plot(years_graph, scenarios_co2_per_year_results[i][j], linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
            ax2.plot(years_graph, scenarios_co2_RF_results[i][j], linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
            ax2.plot(years_graph, scenarios_total_RF_results[i][j], linestyle=linestyles[j], color=colours[i], lw = 1, alpha = 0.3)
    
    ax1.set_title('(a) CO$\mathregular{_2}$ emissions [Mton]', fontweight="bold", fontsize=14)
    ax2.set_title('(b) Radiative forcing [mW/m$\mathregular{^2}$]', fontweight="bold", fontsize=14)
    ax2.annotate('CO$\mathregular{_2}$ only', xy = [2030, 2], xytext = [2030, 5], arrowprops = dict(arrowstyle='->'), ha='center')
    ax2.annotate('incl. non-CO$\mathregular{_2}$ effects', xy = [2035, 14.5], xytext = [2035, 17.5], arrowprops = dict(arrowstyle='->'), ha='center')
    
    ax1.set_ylim(bottom=0)  
    ax2.set_ylim(bottom=0)  
    ax1.set_xlim(2024,2070)
    ax2.set_xlim(2024,2070)
    
    # make legend
    handles, labels = ax1.get_legend_handles_labels()
    colour = '#000000'
    for i in range(len(ac_scenario_names)):
        label = ac_scenario_names[i].replace('\n ','')
        labels.append(label)
        handles.append(Patch(facecolor=colours[i], alpha = alpha))
    by_label = dict(zip(labels, handles))    
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, -0.05), ncol=1, frameon=False)
    
    handles, labels = ax2.get_legend_handles_labels()
    for j in range(len(fuel_scenario_names)):
        label = fuel_scenario_names[j].replace('\n ','')
        linestyle = linestyles[j]
        labels.append(label)
        handles.append(Line2D([0,1],[0,1],linestyle=linestyle, color=colour, lw=1))
    labels.append('ICAO/IATA CO$\mathregular{_2}$ limit')
    handles.append(Line2D([0,1],[0,1],linestyle=':', color='#000000', lw=2))
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(-0.05, -0.05), ncol=1, frameon=False)
    
    fig.tight_layout()
    fig_name = folder+fig1_name+'.pdf'
    fig.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig1_name+'.png'
    fig.savefig(fig_name,bbox_inches='tight',dpi=600)

    # create and save figure data
    data = {
        "Year": years_graph,
        "Target CO2 emissions [kg]": np.array(max_line),
        "Target CO2 RF [mW/m^2]": RF_CO2_target.reset_index(drop=True)
    }

    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            ac_name = ac_scenario_names[i].replace('\n ','')
            fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
            scenario_name = f"{ac_name}, {fuel_name}"
            data[f"CO2 emissions [kg] - {scenario_name}"] = scenarios_co2_per_year_results[i][j].reset_index(drop=True)*1e9 # convert from Mton back to kg
            data[f"CO2 RF [mW/m^2] - {scenario_name}"] = scenarios_co2_RF_results[i][j].reset_index(drop=True)
            data[f"Total RF [mW/m^2] - {scenario_name}"] = scenarios_total_RF_results[i][j].reset_index(drop=True)

    keys_keep = list(data.keys())[:3]
    keys_sorted = sorted(list(data.keys())[3:])
    keys_new = keys_keep + keys_sorted
    data_new = {key: data[key] for key in keys_new}
    df = pd.DataFrame(data_new)

    csv_file_name = folder_csvs+fig1_name+'.csv'
    df.to_csv(csv_file_name, index=False)
    
    # second figure: fleet stacked charts
    fig2_name = 'Xfig_fleet_composition'
    fig_2, axs_2 = plt.subplots(3,2,gridspec_kw={'width_ratios':[1,1],'height_ratios':[1,1,1]},figsize = (6,6))
    labels_2 = [['(a)','(b)'],['(c)','(d)'],['(e)','(f)']]
    style_list_2 = ['','','','||||','','||||'][::-1]
    y_labels_2 = ['High growth', 'Low growth', 'Degrowth']
    x_labels_2 = ['No H$\mathregular{_2}$ aircraft', 'With H$\mathregular{_2}$ aircraft']
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios) - 1):
            ax_here = axs_2[i,j]
            # ax_here.grid(which='major', alpha=0.3)
            stacks_here = ax_here.stackplot(years_graph, scenarios_fleet_results[i][j+1].T[::-1], colors=colours_ac_fleet[::-1], labels=scenarios_fleet_results[i][j+1].columns.tolist()[::-1], alpha = alpha, linewidth=0.1)
            ax_here.set_ylim(0,scenarios_fleet_results_max*1.02)
            ax_here.set_xlim(2024,2070)
            ax_here.text(.03, .96, labels_2[i][j], ha='left', va='top', transform=ax_here.transAxes, fontweight="bold", fontsize=14)
            if i != 2:
                ax_here.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            if j != 0:
                ax_here.tick_params(axis='y', which='both', left=False, labelleft=False)
            if i == 0: 
                ax_here.set_xlabel(x_labels_2[j], fontweight="bold", fontsize=12)
                ax_here.xaxis.set_label_position('top') 
            if j == 0: 
                ax_here.set_ylabel(y_labels_2[i], fontweight="bold", fontsize=12)
            k = 0
            for stack in stacks_here:
                stack.set_hatch(style_list_2[k])
                stack.set_edgecolor('#FFFFFF')
                k += 1
    
    fig_2.suptitle('Number of aircraft [-]', fontsize=14, fontweight="bold",y=0.95)
    # make legend
    handles, labels = ax_here.get_legend_handles_labels()
    ax_here.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0, -0.1), ncol=3, frameon=False)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig_name = folder+fig2_name+'.pdf'
    fig_2.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig2_name+'.png'
    fig_2.savefig(fig_name,bbox_inches='tight',dpi=600)

    # create and save figure data
    data = {
        "Year": years_graph
    }

    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios) - 1):
            stack_data = scenarios_fleet_results[i][j+1].T[::-1]
            stack_columns = scenarios_fleet_results[i][j+1].columns.tolist()[::-1]
            for column_name in stack_columns:
                ac_name = ac_scenario_names[i].replace('\n ','')
                fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
                scenario_name = f"{ac_name}, {fuel_name}"

                column_data = scenarios_fleet_results[i][j+1][column_name]
                column_name = column_name.replace('$\mathregular{_2}$', '2')

                data[f"{scenario_name} - {column_name}"] = column_data.reset_index(drop=True)

    df = pd.DataFrame(data)
    csv_file_name = folder_csvs+fig2_name+'.csv'
    df.to_csv(csv_file_name, index=False) 
    
    # third figure: fuel demands
    fig3_name = 'Fig3_results_fuels'
    fig_3, axs_3 = plt.subplots(3,1,figsize = (6,8))
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            axs_3[0].plot(years_graph, scenarios_fossil_kerosene_results[i][j], linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
            axs_3[1].plot(years_graph, scenarios_aaf_results[i][j], linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
            axs_3[2].plot(years_graph, scenarios_hydrogen_results[i][j], linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
    axs_3[0].set_ylabel('Fossil kerosene use [EJ]', fontweight="bold", fontsize=14)
    axs_3[1].set_ylabel('AAF use [EJ]', fontweight="bold", fontsize=14)
    axs_3[2].set_ylabel('H$\mathregular{_2}$ production [EJ]', fontweight="bold", fontsize=14)
    xmin, xmax = [2024,2070]
    ymin, ymax = [0,3.2]
    axs_3[0].set_ylim(ymin, ymax) 
    axs_3[1].set_ylim(ymin, ymax) 
    axs_3[2].set_ylim(ymin, ymax) 
    axs_3[0].set_xlim(xmin, xmax)
    axs_3[1].set_xlim(xmin, xmax)
    axs_3[2].set_xlim(xmin, xmax)
    axs_3[0].text(.02, .95, '(a)', ha='left', va='top', transform=axs_3[0].transAxes, fontweight="bold", fontsize=14)
    axs_3[1].text(.02, .95, '(b)', ha='left', va='top', transform=axs_3[1].transAxes, fontweight="bold", fontsize=14)
    axs_3[2].text(.02, .95, '(c)', ha='left', va='top', transform=axs_3[2].transAxes, fontweight="bold", fontsize=14)
    axs_3[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs_3[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    #120 MJ/kg --> EJ/Mton 10e12/10e9
    h2_energy_density = 120   # EJ/Mton
    vol_2050_nze    = 44.102*1e9*h2_energy_density/1e12    # Mton in NZE scenario
    vol_2030_eu     = 20*1e9*h2_energy_density/1e12        # Mton in NZE scenario
    dx = 0.1
    axs_3[2].axhline(y = vol_2050_nze*0.2, linestyle = '--', color = '#000000', lw = 1, alpha = alpha)
    axs_3[2].axhline(y = vol_2030_eu*0.2, linestyle = '--', color = '#000000', lw = 1, alpha = alpha)
    axs_3[2].annotate('20% of European \n production in 2050', xy = [2031, vol_2050_nze*0.2 + dx], xytext = [2031, vol_2050_nze*0.2 + dx], ha='center')
    axs_3[2].annotate('20% of EU supply in 2030', xy = [2061, vol_2030_eu*0.2 + dx], xytext = [2061, vol_2030_eu*0.2 + dx], ha='center')
    
    # make legend
    handles, labels = axs_3[0].get_legend_handles_labels()
    colour = '#000000'
    for i in range(len(ac_scenario_names)):
        label = ac_scenario_names[i].replace('\n ','')
        labels.append(label)
        handles.append(Patch(facecolor=colours[i], alpha = alpha))
    for j in range(len(fuel_scenario_names)):
        label = fuel_scenario_names[j].replace('\n ','')
        linestyle = linestyles[j]
        labels.append(label)
        handles.append(Line2D([0,1],[0,1],linestyle=linestyle, color=colour, lw=1))
    by_label = dict(zip(labels, handles))    
    axs_3[2].legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    
    fig_3.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig_name = folder+fig3_name+'.pdf'
    fig_3.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig3_name+'.png'
    fig_3.savefig(fig_name,bbox_inches='tight',dpi=600)

    # create and save figure data
    data = {
        "Year": years_graph
    }

    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            ac_name = ac_scenario_names[i].replace('\n ','')
            fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
            scenario_name = f"{ac_name}, {fuel_name}"
            data[f"Fossil kerosene use [EJ] - {scenario_name}"] = scenarios_fossil_kerosene_results[i][j].reset_index(drop=True)
            data[f"AAF use [EJ] - {scenario_name}"] = scenarios_aaf_results[i][j].reset_index(drop=True)
            data[f"H2 production [EJ] - {scenario_name}"] = scenarios_hydrogen_results[i][j].reset_index(drop=True)

    keys_keep = list(data.keys())[:1]
    keys_sorted = sorted(list(data.keys())[1:])
    keys_new = keys_keep + keys_sorted
    data_new = {key: data[key] for key in keys_new}
    df = pd.DataFrame(data_new)

    csv_file_name = folder_csvs+fig3_name+'.csv'
    df.to_csv(csv_file_name, index=False) 
    
    # fourth figure: alternative impact categories
    # (creating data for figure at the same time)
    data = {
        "Year": years_graph
    }

    fig4_name = 'Sfig8_impact_categories_normalised'
    fig_4, axs_4 = plt.subplots(3,3,figsize = (9,9))
    label_counter = 0
    labels = ['(a)', '(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    y_labels = ['High growth','Low growth','Degrowth']
    x_labels = ['No ReFuelEU Aviation, \n no H$\mathregular{_2}$ aircraft','ReFuelEU Aviation extended, \n no H$\mathregular{_2}$ aircraft', 'ReFuelEU Aviation extended, \n with H$\mathregular{_2}$ aircraft']
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            # for figure image
            ax_here = axs_4[i][j]
            LCIA_df = scenarios_LCIA_df_results[i][j]
            LCIA_df_norm = LCIA_df/LCIA_df.iloc[0]
            LCIA_df_norm_selection = LCIA_df_norm.iloc[:,:15]
            counter = 0
            # for figure data
            ac_name = ac_scenario_names[i].replace('\n ','')
            fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
            scenario_name = f"{ac_name}, {fuel_name}"
            # loop through impact categories
            for name, values in LCIA_df_norm_selection.items():
                ax_here.plot(years, values, label=name, linestyle=style_list[counter][0], color=style_list[counter][1], lw = 1)
                counter += 1
                # add to figure data
                data[f"{scenario_name} - {name} [-]"] = values.reset_index(drop=True)
            ax_here.set_xlim(2024,2070)
            ax_here.set_ylim(0.04,35)
            ax_here.set_yscale('log')
            ax_here.text(.03, .96, labels[label_counter], ha='left', va='top', transform=axs_4[i][j].transAxes, fontweight="bold", fontsize=14)
            label_counter += 1
            if i != 2: ax_here.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            if i == 0: 
                ax_here.set_xlabel(x_labels[j], fontweight="bold", fontsize=12, labelpad=4)
                ax_here.xaxis.set_label_position('top')
            if j != 0: 
                ax_here.tick_params(axis='y', which='both', left=False, labelleft=False)
            else:
                ax_here.set_ylabel(y_labels[i], fontweight="bold", fontsize=12)
    
    axs_4[2][1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    
    fig_4.tight_layout()
    fig_4.suptitle('Impact relative to 2024 [-]', fontsize=14, fontweight="bold",y=0.955)
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.88)
    fig_name = folder+fig4_name+'.pdf'
    fig_4.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig4_name+'.png'
    fig_4.savefig(fig_name,bbox_inches='tight',dpi=600)

    # save figure data
    df = pd.DataFrame(data)
    csv_file_name = folder_csvs+fig4_name+'.csv'
    df.to_csv(csv_file_name, index=False) 

    # fifth figure: 4 plots for alternative impact categories
    # eutrophication: terrestrial
    # material resources: metals/minerals
    # particulate matter formation
    # ecotoxicity: freshwater
    fig5_name = 'Xfig_selected_impact_categories'
    fig_5, axs_5 = plt.subplots(2,2,figsize = (8,8))
    labels = ['(a)','(b)','(c)','(d)']
    scales = [6,9,3,12]
    units = ['kg Sb-eq','mol N-eq','disease incidence','CTUe']
    impact_cat_selected = ['material resources: metals/minerals', 'eutrophication: terrestrial', 'particulate matter formation', 'ecotoxicity: freshwater']
    impact_cat_counter = 0
    # (creating data for figure at the same time)
    data = {
        "Year": years_graph
    }
    for a in range(2):
        for b in range(2):
            ax_here = axs_5[a][b]
            scales_here = scales[impact_cat_counter]
            impact_cat_here = impact_cat_selected[impact_cat_counter]
            ax_here.set_xlabel(labels[impact_cat_counter]+' '+impact_cat_here+'\n [$\mathregular{10^{'+str(scales_here)+'}}$ '+units[impact_cat_counter]+']', fontweight="bold", fontsize=14, labelpad=6)
            ax_here.xaxis.set_label_position('top') 
            for i in range(len(ac_scenarios)):
                for j in range(len(fuel_scenarios)):
                    ax_here.plot(years, scenarios_LCIA_df_results[i][j][impact_cat_here]/(10**scales_here), linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
                    # add to figure data
                    ac_name = ac_scenario_names[i].replace('\n ','')
                    fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
                    scenario_name = f"{ac_name}, {fuel_name}"
                    data[f"{scenario_name} - {impact_cat_here} [{units[impact_cat_counter]}]"] = scenarios_LCIA_df_results[i][j][impact_cat_here].reset_index(drop=True)
            if a != 1: ax_here.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax_here.set_xlim(2024,2070)
            ax_here.set_ylim(bottom=0)
            impact_cat_counter += 1 
    # make legend
    handles, labels = axs_5[0][0].get_legend_handles_labels()
    colour = '#000000'
    for i in range(len(ac_scenario_names)):
        label = ac_scenario_names[i].replace('\n ','')
        labels.append(label)
        handles.append(Patch(facecolor=colours[i], alpha = alpha))
    for j in range(len(fuel_scenario_names)):
        label = fuel_scenario_names[j].replace('\n ','')
        linestyle = linestyles[j]
        labels.append(label)
        handles.append(Line2D([0,1],[0,1],linestyle=linestyle, color=colour, lw=1))
    by_label = dict(zip(labels, handles))    
    axs_5[1][1].legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(-0.1, -0.1), ncol=2, frameon=False)
    
    # save figure & data
    fig_name = folder+fig5_name+'.pdf'
    fig_5.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig5_name+'.png'
    fig_5.savefig(fig_name,bbox_inches='tight',dpi=600)
    df = pd.DataFrame(data)
    csv_file_name = folder_csvs+fig5_name+'.csv'
    df.to_csv(csv_file_name, index=False) 
        
    # sixth figure: 
    # (a) CO2 emissions per MJ
    # (b) MJ per RPK
    # (c) CO2 emissions per RPK
    fig6_name = 'Xfig_relative_performance'
    # (creating data for figure at the same time)
    data = {
        "Year": years_graph
    }
    fig_6, axs_6 = plt.subplots(3,1,gridspec_kw={'height_ratios':[1,1,1]},figsize =(6,10))
    let_labels = ['(a)','(b)','(c)']
    y_labels = ['Fuel CO$\mathregular{_2}$ intensity [g/MJ]', 'RPK fuel intensity [MJ/RPK]', 'RPK CO$\mathregular{_2}$ intensity \n [g/RPK]']
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            list_of_dic = scenarios_list_of_dic_results[i][j]
            total_fuel = scenarios_fuels_results[i][j]
            first_plot_data = calculate_impact_per_mj(total_fuel, list_of_dic, 'Carbon dioxide')*1e3
            second_plot_data = scenarios_fuels_results[i][j].sum(axis = 1)/scenarios_rpk_per_aircraft[i][j].sum(axis=1)
            third_plot_data = scenarios_co2_per_year_results[i][j]*1e9*1e3/scenarios_rpk_per_aircraft[i][j].sum(axis=1)
            axs_6[0].plot(years_graph, first_plot_data, linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
            axs_6[1].plot(years_graph, second_plot_data, linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
            axs_6[2].plot(years_graph, third_plot_data, linestyle=linestyles[j], color=colours[i], lw = 1, alpha = alpha)
            # add to figure data
            ac_name = ac_scenario_names[i].replace('\n ','')
            fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
            scenario_name = f"{ac_name}, {fuel_name}"
            data[f"Fuel CO2 intensity [g/MJ] - {scenario_name}"] = first_plot_data.reset_index(drop=True)
            data[f"RPK fuel intensity [MJ/RPK] - {scenario_name}"] = second_plot_data.reset_index(drop=True)
            data[f"RPK CO2 intensity [g/RPK] - {scenario_name}"] = third_plot_data.reset_index(drop=True)
    for i in range(3):
        ax_here = axs_6[i]
        ax_here.text(-0.12, 0.5, y_labels[i], ha='center', va='center', transform=ax_here.transAxes, fontweight="bold", fontsize=14, rotation = 90)        
        ax_here.text(.03, .05, let_labels[i], ha='left', va='bottom', transform=ax_here.transAxes, fontweight="bold", fontsize=14)
        ax_here.set_ylim(bottom=0)  
        ax_here.set_xlim(2024,2070)
        if i != 2: ax_here.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # make legend
    handles, labels = axs_6[0].get_legend_handles_labels()
    colour = '#000000'
    for i in range(len(ac_scenario_names)):
        label = ac_scenario_names[i].replace('\n ','')
        labels.append(label)
        handles.append(Patch(facecolor=colours[i], alpha = alpha))
    for j in range(len(fuel_scenario_names)):
        label = fuel_scenario_names[j].replace('\n ','')
        linestyle = linestyles[j]
        labels.append(label)
        handles.append(Line2D([0,1],[0,1],linestyle=linestyle, color=colour, lw=1))
    by_label = dict(zip(labels, handles))
    axs_6[2].legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, -0.05), ncol=2, frameon=False)
    
    # sort figure data
    keys_keep = list(data.keys())[:1]
    keys_sorted = sorted(list(data.keys())[1:])
    keys_new = keys_keep + keys_sorted
    data_new = {key: data[key] for key in keys_new}
    df = pd.DataFrame(data_new)

    # save figure image & data
    plt.subplots_adjust(hspace=0.05)
    fig_name = folder+fig6_name+'.pdf'
    fig_6.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig6_name+'.png'
    fig_6.savefig(fig_name,bbox_inches='tight',dpi=600)
    csv_file_name = folder_csvs+fig6_name+'.csv'
    df.to_csv(csv_file_name, index=False) 
    
    # seventh figure: LWE contributions
    lwe_colours = [LU_orange, LU_violet, LU_light_blue, LU_red, LU_turqoise] #LU_green
    fig7_name = 'Sfig6_LWE_substance_contribution_analysis'
    fig_7, axs_7 = plt.subplots(3,3,figsize = (9,9))
    label_counter = 0
    letter_labels = ['(a)', '(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    y_labels = ['High growth','Low growth','Degrowth']
    x_labels = ['No ReFuelEU Aviation, \n no H$\mathregular{_2}$ aircraft','ReFuelEU Aviation extended, \n no H$\mathregular{_2}$ aircraft', 'ReFuelEU Aviation extended, \n with H$\mathregular{_2}$ aircraft']
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            ax_here = axs_7[i][j]
            RF = scenarios_RF_results[i][j]
            RF_low = scenarios_RF_low_results[i][j]
            RF_high = scenarios_RF_high_results[i][j]
            
            labels = list(RF.columns)
            for k in range(len(labels)):
                labels[k] = labels[k].replace('CO2', 'CO$\mathregular{_2}$')
                labels[k] = labels[k].replace('NOx', 'NO$\mathregular{_x}$')
            
            ax_here.stackplot(years_graph, RF.T, labels=labels, colors=lwe_colours)
            err_years = []
            lower = []
            upper = []
            mid = []
            for year in years_graph:
                if year%5 == 0 and year%10 != 0:
                    err_years.append(year)
                    RF_here = RF.loc[year].sum()
                    mid.append(RF_here)
                    lower.append(RF_here - RF_low.loc[year].sum())
                    upper.append(RF_high.loc[year].sum() - RF_here)
            
            ax_here.errorbar(err_years, mid, yerr=(lower, upper), fmt='_', color='#000000', capsize=3, lw = 1, alpha = alpha)
            
            ax_here.set_xlim(2024,2070)
            ax_here.set_ylim(0,35)
            ax_here.text(.03, .96, letter_labels[label_counter], ha='left', va='top', transform=ax_here.transAxes, fontweight="bold", fontsize=14)
            label_counter += 1
            if i != 2: ax_here.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            if i == 0: 
                ax_here.set_xlabel(x_labels[j], fontweight="bold", fontsize=12, labelpad=4)
                ax_here.xaxis.set_label_position('top')
            if j != 0: 
                ax_here.tick_params(axis='y', which='both', left=False, labelleft=False)
            else:
                ax_here.set_ylabel(y_labels[i], fontweight="bold", fontsize=12)
    
    fig_7.suptitle('Radiative forcing [mW/m$\mathregular{^2}$]', fontsize=14, fontweight="bold",y=0.96)
    axs_7[2][1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
    
    fig_7.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.88)
    fig_name = folder+fig7_name+'.pdf'
    fig_7.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig7_name+'.png'
    fig_7.savefig(fig_name,bbox_inches='tight',dpi=600)

    # create and save figure data
    data = {
        "Year": years_graph
    }
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            RF = scenarios_RF_results[i][j]
            RF_low = scenarios_RF_low_results[i][j]
            RF_high = scenarios_RF_high_results[i][j]
            ac_name = ac_scenario_names[i].replace('\n ','')
            fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
            scenario_name = f"{ac_name}, {fuel_name}"

            labels = list(RF.columns)
            data[f"{scenario_name} - RF with middle efficacy, total [mW/m2]"] = RF.sum(axis = 1).reset_index(drop=True)
            data[f"{scenario_name} - RF with low-end efficacy, total [mW/m2]"] = RF_low.sum(axis = 1).reset_index(drop=True)
            data[f"{scenario_name} - RF with high-end efficacy, total [mW/m2]"] = RF_high.sum(axis = 1).reset_index(drop=True)
            for label in labels:
                data[f"{scenario_name} - RF with middle efficacy, contribution {label} [mW/m2]"] = RF[label].reset_index(drop=True)
                data[f"{scenario_name} - RF with low-end efficacy, contribution {label} [mW/m2]"] = RF_low[label].reset_index(drop=True)
                data[f"{scenario_name} - RF with high-end efficacy, contribution {label} [mW/m2]"] = RF_high[label].reset_index(drop=True)
    
    keys_keep = list(data.keys())[:1]
    keys_sorted = sorted(list(data.keys())[1:])
    keys_new = keys_keep + keys_sorted
    data_new = {key: data[key] for key in keys_new}
    df = pd.DataFrame(data_new)

    csv_file_name = folder_csvs+fig7_name+'.csv'
    df.to_csv(csv_file_name, index=False) 
    
    # eighth figure: contribution analysis CO2 emissions
    fig8_name = 'Sfig4_CO2_stage_contribution_analysis'
    # (creating data for figure at the same time)
    data = {
        "Year": years_graph
    }
    contr_colours = [LU_orange, LU_violet, LU_light_blue, LU_green, LU_red, LU_turqoise]
    fig_8, axs_8 = plt.subplots(3,3,figsize = (9,9))
    label_counter = 0
    letter_labels = ['(a)', '(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    y_labels = ['High growth','Low growth','Degrowth']
    x_labels = ['No ReFuelEU Aviation, \n no H$\mathregular{_2}$ aircraft','ReFuelEU Aviation extended, \n no H$\mathregular{_2}$ aircraft', 'ReFuelEU Aviation extended, \n with H$\mathregular{_2}$ aircraft']
    for i in range(len(ac_scenarios)):
        for j in range(len(fuel_scenarios)):
            ax_here = axs_8[i][j]
            list_of_dic = scenarios_list_of_dic_results[i][j]
            impact_cat = 'Carbon dioxide'
            plot_inflow = impact_cat_result(list_of_dic[0], impact_cat)
            plot_outflow = impact_cat_result(list_of_dic[1], impact_cat)
            plot_fossil_wtt = impact_cat_result(list_of_dic[2], impact_cat)
            plot_fossil_LTO_ttw = impact_cat_result(list_of_dic[3], impact_cat)
            plot_fossil_CCD_ttw = impact_cat_result(list_of_dic[4], impact_cat)
            plot_saf_wtt_infra = impact_cat_result(list_of_dic[5], impact_cat)
            plot_saf_wtt_ops = impact_cat_result(list_of_dic[6], impact_cat)
            plot_saf_LTO_ttw = impact_cat_result(list_of_dic[7], impact_cat)
            plot_saf_CCD_ttw = impact_cat_result(list_of_dic[8], impact_cat)
            plot_h2_wtt_infra = impact_cat_result(list_of_dic[9], impact_cat)
            plot_h2_wtt_ops = impact_cat_result(list_of_dic[10], impact_cat)
            plot_h2_LTO_ttw = impact_cat_result(list_of_dic[11], impact_cat)
            plot_h2_CCD_ttw = impact_cat_result(list_of_dic[12], impact_cat)
            
            combine_aircraft = plot_inflow + plot_outflow
            combine_wtt_infra = plot_saf_wtt_infra + plot_h2_wtt_infra
            combine_wtt_ops = plot_saf_wtt_ops + plot_h2_wtt_ops
            combine_ttw_fossil = plot_fossil_LTO_ttw + plot_fossil_CCD_ttw
            combine_ttw_aaf =  plot_saf_LTO_ttw + plot_h2_LTO_ttw + plot_saf_CCD_ttw + plot_h2_CCD_ttw
            
            contr_labels = ['year','aircraft system','fossil kerosene well-to-tank','fossil kerosene tank-to-wake','AAF well-to-tank infrastructure','AAF well-to-tank operations','AAF tank-to-wake']
            contr_stack = np.vstack([years_graph, combine_aircraft, plot_fossil_wtt, combine_ttw_fossil, combine_wtt_infra, combine_wtt_ops, combine_ttw_aaf])
            contr_df = pd.DataFrame(contr_stack.T/1e9, 
                              columns = contr_labels)  
            total_df = pd.DataFrame(np.vstack([years_graph,scenarios_co2_per_year_results[i][j]]).T, 
                                    columns= ['year','total'])
            total_df['year'] = total_df['year'].astype('string')
            contr_df.plot(x='year', kind="bar", stacked=True, ax=ax_here, legend=False, color=contr_colours, width=1)
            total_df.plot(x='year', kind="line", ax=ax_here, color='#000000', linestyle='--', lw=1, legend=False)
            
            # add to figure data
            ac_name = ac_scenario_names[i].replace('\n ','')
            fuel_name = fuel_scenario_names[j].replace('\n ','').replace('$\mathregular{_2}$','2')
            scenario_name = f"{ac_name}, {fuel_name}"
            data[f"{scenario_name} - CO2 emissions, total [kg]"] = total_df['total'].reset_index(drop=True)*1e9 # convert from Mton back to kg
            for label in list(contr_df.columns):
                if not label=='year': data[f"{scenario_name} - CO2 emissions, contribution {label} [kg]"] = contr_df[label].reset_index(drop=True)*1e9 # convert from Mton back to kg
            
            # configure image lay-out
            ax_here.set_xlim(0,46)
            ax_here.set_ylim(-120,180)
            
            pos = [6,16,26,36,46]
            l = [2030,2040,2050,2060,2070]
            ax_here.set(xticks=pos, xticklabels=l)
            
            ax_here.text(.03, .03, letter_labels[label_counter], ha='left', va='bottom', transform=ax_here.transAxes, fontweight="bold", fontsize=14)
            label_counter += 1
            if i != 2: ax_here.tick_params(axis='x', which='both', bottom=False, labelbottom=False)            
            if i == 0: 
                ax_here.set_xlabel(x_labels[j], fontweight="bold", fontsize=12, labelpad=4)
                ax_here.xaxis.set_label_position('top')
            else:
                ax_here.xaxis.label.set_visible(False)
            if j != 0: 
                ax_here.tick_params(axis='y', which='both', left=False, labelleft=False)
            else:
                ax_here.set_ylabel(y_labels[i], fontweight="bold", fontsize=12)
    
    fig_8.suptitle('CO$\mathregular{_2}$ emissions [Mton]', fontsize=14, fontweight="bold",y=0.956)
    axs_8[2][1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    
    # save figure image & data
    fig_8.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.88)
    fig_name = folder+fig8_name+'.pdf'
    fig_8.savefig(fig_name,bbox_inches='tight')
    fig_name = folder+fig8_name+'.png'
    fig_8.savefig(fig_name,bbox_inches='tight',dpi=600)
    df = pd.DataFrame(data)
    csv_file_name = folder_csvs+fig8_name+'.csv'
    df.to_csv(csv_file_name, index=False) 

#%% function used in visualising AAF scenario variables
def aaf_share_for_plt(aaf_advanced, y_start, y_stop):
    # variables for alternative fuel share
    aaf_0 = 0.0005 # share of saf in total fuel at start time (from the ICCT)
    if aaf_advanced == 'ReFuelEU base':
        aaf_milestones = pd.DataFrame([[2025, 2030, 2035, 2040, 2045, 2050], [0.02, 0.06, 0.2, 0.34, 0.42, 0.7]], ['year', 'AAF share']).T # milestones based on ReFuelEU Aviation (except for 2060, that one I added myself)
    if aaf_advanced == 'ReFuelEU extended':
        aaf_milestones = pd.DataFrame([[2025, 2030, 2035, 2040, 2045, 2050, 2060], [0.02, 0.06, 0.2, 0.34, 0.42, 0.7, 1]], ['year', 'AAF share']).T # milestones based on ReFuelEU Aviation (except for 2060, that one I added myself)
    if aaf_advanced == 'no ReFuelEU':
        aaf_milestones = pd.DataFrame([[2060], [aaf_0]], ['year', 'AAF share']).T # do not change AAF share
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
    
    return [i*100 for i in aaf_goal_timeline]

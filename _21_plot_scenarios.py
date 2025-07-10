import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase
from copy import deepcopy
import seaborn as sns

from _10_functions import *
from _11_define_scenarios import * 
from _12_LWE_function import *

# class to include multiple lines at once in the legend
class MultiStyleLineHandler(HandlerBase):
    def __init__(self, colors, linestyles, **kwargs):
        self.colors = colors
        self.linestyles = linestyles
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        y0 = height / 2
        dy = 3  # vertical spacing between lines
        lines = []
        # zip colors and linestyles, align with vertical offsets (e.g. -1,0,1)
        offsets = [-1, 0, 1]
        for i, color, ls in zip(offsets, self.colors, self.linestyles):
            y = y0 + i * dy
            line = Line2D([xdescent, xdescent + width],
                          [y, y],
                          linestyle=ls,
                          color=color,
                          linewidth=orig_handle.get_linewidth(),
                          transform=trans)
            lines.append(line)
        return lines
    
def make_legend_timeline(scenarios):
    colour = '#000000'
    handles = []
    labels = []
    scenario_colours = []
    scenario_linestyles = []
    for scenario in scenarios:
        if scenario['colour'] not in scenario_colours: scenario_colours.append(scenario['colour'])
        if scenario['line style'] not in scenario_linestyles: scenario_linestyles.append(scenario['line style'])
    for scenario in scenarios:
        line = Patch(facecolor=scenario['colour'], alpha = 1)
        label = scenario['name base'].replace('\n ','')
        handles.append(line)
        labels.append(label)
    for scenario in scenarios:
        line = Line2D([0,1],[0,1],linestyle=scenario['line style'], color=colour, lw=1)
        label = scenario['name AAF'].replace('\n ','')
        handles.append(line)
        labels.append(label)
    return labels, handles, None

# alternative function for creating standard legend labels & handles for timeline plots
def make_legend_timeline_alt(scenarios):
    handles = []
    labels = []
    handler_map = {}
    scenario_colours = []
    scenario_linestyles = []
    for scenario in scenarios:
        if scenario['colour'] not in scenario_colours: scenario_colours.append(scenario['colour'])
        if scenario['line style'] not in scenario_linestyles: scenario_linestyles.append(scenario['line style'])
    for scenario in scenarios:
        multi_line = Line2D([0], [0], linewidth=1)
        multi_label = scenario['name base'].replace('\n ','')
        handles.append(multi_line)
        labels.append(multi_label)
        handler_map[multi_line] = MultiStyleLineHandler([scenario['colour']]*len(scenario_colours), scenario_linestyles)
    for scenario in scenarios:
        multi_line = Line2D([0], [0], linewidth=1)
        multi_label = scenario['name AAF'].replace('\n ','')
        handles.append(multi_line)
        labels.append(multi_label)
        handler_map[multi_line] = MultiStyleLineHandler(scenario_colours, [scenario['line style']]*len(scenario_linestyles))
    return labels, handles, handler_map

#%% function used to process scenario results by determining LCIA results per MJ fuel
def calculate_impact_per_mj(total_fuel, list_of_dic, impact_cat):
    fuel_impacts = extract_processes(list_of_dic, impact_cat, list(range(2,len(list_of_dic)))) # 0 and 1 are excluded, since that are aircraft impacts
    sum_of_impacts = fuel_impacts.sum(axis = 1)
    sum_of_fuel = total_fuel.sum(axis = 1)
    
    total_impact_mj = sum_of_impacts/sum_of_fuel
    
    return total_impact_mj

#%% functions for plotting the various heatmaps
# make dataframes with values for heatmaps
def unpack_results_for_heatmap(scenario_result_here, target, time, flight_start_year, mode='LWE'):
    if mode == 'LWE':
        impacts = scenario_result_here[2].copy()
        impacts_sens = scenario_result_here[20].copy()

        impact_co2 = list(impacts.iloc[:,0])
        impact_all = list(impacts.sum(axis = 1))
        impact_sens = list(impacts_sens.sum(axis = 1))

    if mode == 'GWPstar':
        impacts = scenario_result_here[24].copy().drop(flight_start_year-1)
        impacts_sens = scenario_result_here[25].copy().drop(flight_start_year-1)

        impact_co2 = list(impacts['temperature_increase_from_co2_from_aviation'])
        impact_all = list(impacts['temperature_increase_from_aviation'])
        impact_sens = list(impacts_sens['temperature_increase_from_aviation'])

    budget_result = impact_co2[time]/target[time]*100
    warming_just_co2_result = impact_co2[time]/impact_co2[2050 - flight_start_year]*100
    warming_incl_non_co2_result = impact_all[time]/impact_all[2050 - flight_start_year]*100
    warming_incl_non_co2_sens_result = impact_sens[time]/impact_sens[2050 - flight_start_year]*100

    return budget_result, warming_just_co2_result, warming_incl_non_co2_result, warming_incl_non_co2_sens_result

subplot_fontweight = 'normal'
def make_target_heat_map(year_of_interest, flight_start_year, all_scenario_names, all_scenario_results, target, mode='LWE'):
    heat_map_columns = ['no ReFuelEU with \n no H$_2$ AC', 'ReFuelEU as-is with \n no H$_2$ AC', 'ReFuelEU as-is with \n low-performance H$_2$ AC', 'ReFuelEU as-is with \n mid-performance H$_2$ AC', 'ReFuelEU as-is with \n high-performance H$_2$ AC', 'ReFuelEU extended with \n no H$_2$ AC', 'ReFuelEU extended with \n low-performance H$_2$ AC', 'ReFuelEU extended with \n mid-performance H$_2$ AC', 'ReFuelEU extended with \n high-performance H$_2$ AC'] # variables 
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
    heat_map_warming_incl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_incl_non_co2_sens = np.empty([len(heat_map_rows), len(heat_map_columns)])
    
    # global variables across heat map
    background = '1.7C'
    hydrogen_source = 'grid'

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
                budget_result_here = warming_just_co2_result_here = warming_incl_non_co2_result_here = warming_incl_non_co2_sens_result_here = np.nan 
            else:
                scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
                budget_result_here, warming_just_co2_result_here, warming_incl_non_co2_result_here, warming_incl_non_co2_sens_result_here = unpack_results_for_heatmap(scenario_result_here, target, time, flight_start_year, mode=mode)
                
            heat_map_budget[i,j]                = budget_result_here
            heat_map_warming_just_co2[i,j]      = warming_just_co2_result_here
            heat_map_warming_incl_non_co2[i,j]  = warming_incl_non_co2_result_here
            heat_map_warming_incl_non_co2_sens[i,j]  = warming_incl_non_co2_sens_result_here
            
    heat_map_budget                 = pd.DataFrame(heat_map_budget, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_just_co2       = pd.DataFrame(heat_map_warming_just_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2   = pd.DataFrame(heat_map_warming_incl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2_sens = pd.DataFrame(heat_map_warming_incl_non_co2_sens, index = heat_map_rows, columns = heat_map_columns)
    
    return heat_map_budget, heat_map_warming_just_co2, heat_map_warming_incl_non_co2, heat_map_warming_incl_non_co2_sens

# make dataframes with values for heatmaps of foreground sensitivity analyses
def make_foreground_heat_map(year_of_interest, flight_start_year, all_scenario_names, all_scenario_results, target, mode='LWE'):
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
                heat_map_columns.append(aaf_impl_names[i]+' with '+'no '*(not ops_impr_state)+'op. impr. \n and no H$_2$ AC')
                heat_map_column_characteristics.append([aaf_impl_state, h2ac_impl_state, h2ac_tech_state, ops_impr_state])
        else:
            for h2ac_impl_state in h2ac_impl:
                if h2ac_impl_state == False:
                    for ops_impr_state in ops_impr:
                        h2ac_tech_state = 'low'
                        heat_map_columns.append(aaf_impl_names[i]+' with '+'no '*(not ops_impr_state)+'op. impr. \n and no H$_2$ AC')
                        heat_map_column_characteristics.append([aaf_impl_state, h2ac_impl_state, h2ac_tech_state, ops_impr_state])
                else:
                    for h2ac_tech_state in h2ac_tech:
                        for ops_impr_state in ops_impr:
                            heat_map_columns.append(aaf_impl_names[i]+' with '+'no '*(not ops_impr_state)+'op. impr. \n and '+h2ac_tech_state+'-performance H$_2$ AC')
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
    heat_map_warming_incl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_incl_non_co2_sens = np.empty([len(heat_map_rows), len(heat_map_columns)])
    
    # global variables across heat map
    background = '1.7C'
    hydrogen_source = 'grid'

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
                budget_result_here = warming_just_co2_result_here = warming_incl_non_co2_result_here = warming_incl_non_co2_sens_result_here = np.nan 
            else:
                scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
                budget_result_here, warming_just_co2_result_here, warming_incl_non_co2_result_here, warming_incl_non_co2_sens_result_here = unpack_results_for_heatmap(scenario_result_here, target, time, flight_start_year, mode=mode)
                
            heat_map_budget[i,j]                = budget_result_here
            heat_map_warming_just_co2[i,j]      = warming_just_co2_result_here
            heat_map_warming_incl_non_co2[i,j]  = warming_incl_non_co2_result_here
            heat_map_warming_incl_non_co2_sens[i,j]  = warming_incl_non_co2_sens_result_here
            
    heat_map_budget                 = pd.DataFrame(heat_map_budget, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_just_co2       = pd.DataFrame(heat_map_warming_just_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2   = pd.DataFrame(heat_map_warming_incl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2_sens = pd.DataFrame(heat_map_warming_incl_non_co2_sens, index = heat_map_rows, columns = heat_map_columns)
    
    return heat_map_budget, heat_map_warming_just_co2, heat_map_warming_incl_non_co2, heat_map_warming_incl_non_co2_sens

# make dataframes with values for heatmaps of background sensitivity analyses
def make_background_heat_map(year_of_interest, flight_start_year, all_scenario_names, all_scenario_results, background, hydrogen_source, target, mode='LWE'):
    heat_map_columns = ['no ReFuelEU with \n no H$_2$ AC', 'ReFuelEU as-is with \n no H$_2$ AC', 'ReFuelEU as-is with \n low-performance H$_2$ AC', 'ReFuelEU as-is with \n mid-performance H$_2$ AC', 'ReFuelEU as-is with \n high-performance H$_2$ AC', 'ReFuelEU extended with \n no H$_2$ AC', 'ReFuelEU extended with \n low-performance H$_2$ AC', 'ReFuelEU extended with \n mid-performance H$_2$ AC', 'ReFuelEU extended with \n high-performance H$_2$ AC'] # variables 
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
    heat_map_warming_incl_non_co2 = np.empty([len(heat_map_rows), len(heat_map_columns)])
    heat_map_warming_incl_non_co2_sens = np.empty([len(heat_map_rows), len(heat_map_columns)])
    
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
                budget_result_here = warming_just_co2_result_here = warming_incl_non_co2_result_here = warming_incl_non_co2_sens_result_here = np.nan 
            else:
                scenario_result_here = retrieve_scenario_results(scenario_name_here, all_scenario_names, all_scenario_results)
                budget_result_here, warming_just_co2_result_here, warming_incl_non_co2_result_here, warming_incl_non_co2_sens_result_here = unpack_results_for_heatmap(scenario_result_here, target, time, flight_start_year, mode=mode)
                
            heat_map_budget[i,j]                = budget_result_here
            heat_map_warming_just_co2[i,j]      = warming_just_co2_result_here
            heat_map_warming_incl_non_co2[i,j]  = warming_incl_non_co2_result_here
            heat_map_warming_incl_non_co2_sens[i,j]  = warming_incl_non_co2_sens_result_here
            
    heat_map_budget                 = pd.DataFrame(heat_map_budget, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_just_co2       = pd.DataFrame(heat_map_warming_just_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2   = pd.DataFrame(heat_map_warming_incl_non_co2, index = heat_map_rows, columns = heat_map_columns)
    heat_map_warming_incl_non_co2_sens = pd.DataFrame(heat_map_warming_incl_non_co2_sens, index = heat_map_rows, columns = heat_map_columns)
    
    return heat_map_budget, heat_map_warming_just_co2, heat_map_warming_incl_non_co2, heat_map_warming_incl_non_co2_sens
# plot dataframes to heatmaps
def plot_target_heat_map(heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4, folder, mode):
    fig, ((ax1, ax2, axcb), (ax3, ax4, axcb2)) = plt.subplots(2,3,gridspec_kw={'width_ratios':[1,1,0.08]},figsize = (11,13))
    
    fig1 = sns.heatmap(heat_map_df_1, 
                       annot = heat_map_df_1.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = False, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax1, cbar=False, cmap = 'seismic')
    fig2 = sns.heatmap(heat_map_df_2, 
                       annot = heat_map_df_2.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = False, yticklabels = False, center = 100, vmin=0, vmax=200, ax = ax2, cbar=False, cmap = 'seismic')
    fig3 = sns.heatmap(heat_map_df_3, 
                       annot = heat_map_df_3.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax3, cbar=False, cmap = 'seismic')
    fig4 = sns.heatmap(heat_map_df_4, 
                       annot = heat_map_df_4.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = True, yticklabels = False, center = 100, vmin=0, vmax=200, ax = ax4, cbar_ax=axcb, cmap = 'seismic')
    
    axcb2.axis('off')
    
    if mode=='LWE':
        ax1.set_title('(a) Share of CO$_2$ budget used by 2070 \n based on ICAO/IATA limit [\%]', fontweight=subplot_fontweight)
        ax2.set_title('(b) Relative change in RF from 2050 to 2070, \n CO$_2$ only [\%]', fontweight=subplot_fontweight)
        ax3.set_title('(c) Relative change in RF from 2050 to 2070 [\%]', fontweight=subplot_fontweight)
        ax4.set_title('(d) Relative change in RF from 2050 to 2070, \n if AAF does not affect cirrus impacts [\%]', fontweight=subplot_fontweight)
    if mode=='GWPstar':
        ax1.set_title('(a) Share of CO$_2$ budget used by 2070 \n based on ICAO/IATA limit [\%]', fontweight=subplot_fontweight)
        ax2.set_title('(b) Relative change in warming from 2050 to 2070, \n CO$_2$ only [\%]', fontweight=subplot_fontweight)
        ax3.set_title('(c) Relative change in warming from 2050 to 2070 [\%]', fontweight=subplot_fontweight)
        ax4.set_title('(d) Relative change in warming from 2050 to 2070, \n if AAF does not affect cirrus impacts [\%]', fontweight=subplot_fontweight)
    
    fig.tight_layout()
    fig_name = folder+'Xfig_heatmap_'+mode+'.pdf'
    fig.savefig(fig_name,bbox_inches='tight')#, dpi=600)
    fig_name = folder+'Xfig_heatmap_'+mode+'.png'
    fig.savefig(fig_name,bbox_inches='tight', dpi=600)   
# plot dataframes for foreground sensitivity heatmaps
def plot_foreground_heat_map(heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4, folder, mode):
    fig1, (ax1, axcb1) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    fig2, (ax2, axcb2) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    fig3, (ax3, axcb3) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    fig4, (ax4, axcb4) = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.04]},figsize = (10,15))
    figs = [fig1, fig2, fig3, fig4]
    
    sns.heatmap(heat_map_df_1, 
                annot = heat_map_df_1.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax1, cbar_ax=axcb1, cmap = 'seismic')
    sns.heatmap(heat_map_df_2, 
                annot = heat_map_df_2.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax2, cbar_ax=axcb2, cmap = 'seismic')
    sns.heatmap(heat_map_df_3, 
                annot = heat_map_df_3.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax3, cbar_ax=axcb3, cmap = 'seismic')
    sns.heatmap(heat_map_df_4, 
                annot = heat_map_df_4.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax4, cbar_ax=axcb4, cmap = 'seismic')
    
    fig_names = [folder+'Xfig_heatmap_sensitivity_budget_'+mode, folder+'Xfig_heatmap_sensitivity_trend_CO2_'+mode, folder+'Xfig_heatmap_sensitivity_trend_all_'+mode, folder+'Xfig_heatmap_sensitivity_trend_sens_'+mode]
    for i in range(len(fig_names)):
        figs[i].tight_layout()
        figs[i].savefig(fig_names[i]+'.pdf',bbox_inches='tight')
    for i in range(len(fig_names)):
        figs[i].tight_layout()
        figs[i].savefig(fig_names[i]+'.png',bbox_inches='tight', dpi=600)  
# plot dataframes for background sensitivity heatmaps
def plot_background_heat_map(heat_map_df_1, heat_map_df_2, heat_map_df_3, heat_map_df_4, folder, background, hydrogen_source, mode):
    #fig, (ax1, ax2, ax3, ax4, axcb) = plt.subplots(1,5,gridspec_kw={'width_ratios':[1,1,1,1, 0.08]},figsize = (20,8))
    fig, ((ax1, ax2, axcb), (ax3, ax4, axcb2)) = plt.subplots(2,3,gridspec_kw={'width_ratios':[1,1,0.08]},figsize = (11,13))
    
    fig1 = sns.heatmap(heat_map_df_1, 
                       annot = heat_map_df_1.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = False, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax1, cbar=False, cmap = 'seismic')
    fig2 = sns.heatmap(heat_map_df_2, 
                       annot = heat_map_df_2.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = False, yticklabels = False, center = 100, vmin=0, vmax=200, ax = ax2, cbar=False, cmap = 'seismic')
    fig3 = sns.heatmap(heat_map_df_3, 
                       annot = heat_map_df_3.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = True, yticklabels = True, center = 100, vmin=0, vmax=200, ax = ax3, cbar=False, cmap = 'seismic')
    fig4 = sns.heatmap(heat_map_df_4, 
                       annot = heat_map_df_4.applymap(lambda x: r"\textbf{{{:.0f}}}".format(x) if x < 100 else "{:.0f}".format(x)), fmt="", 
                       xticklabels = True, yticklabels = False, center = 100, vmin=0, vmax=200, ax = ax4, cbar_ax=axcb, cmap = 'seismic')
    
    axcb2.axis('off')
    
    if mode=='LWE':
        ax1.set_title('(a) Share of CO$_2$ budget used by 2070 \n based on ICAO/IATA limit [\%]', fontweight=subplot_fontweight)
        ax2.set_title('(b) Relative change in RF from 2050 to 2070, \n CO$_2$ only [\%]', fontweight=subplot_fontweight)
        ax3.set_title('(c) Relative change in RF from 2050 to 2070 [\%]', fontweight=subplot_fontweight)
        ax4.set_title('(d) Relative change in RF from 2050 to 2070, \n if AAF does not affect cirrus impacts [\%]', fontweight=subplot_fontweight)
    if mode=='GWPstar':
        ax1.set_title('(a) Share of CO$_2$ budget used by 2070 \n based on ICAO/IATA limit [\%]', fontweight=subplot_fontweight)
        ax2.set_title('(b) Relative change in warming from 2050 to 2070, \n CO$_2$ only [\%]', fontweight=subplot_fontweight)
        ax3.set_title('(c) Relative change in warming from 2050 to 2070 [\%]', fontweight=subplot_fontweight)
        ax4.set_title('(d) Relative change in warming from 2050 to 2070, \n if AAF does not affect cirrus impacts [\%]', fontweight=subplot_fontweight)
    
    fig.tight_layout()
    heatmap_name = folder+'Xfig_heatmap_'+background+hydrogen_source+'_'+mode
    fig_name = heatmap_name+'.pdf'
    fig.savefig(fig_name,bbox_inches='tight')
    fig_name = heatmap_name+'.png'
    fig.savefig(fig_name,bbox_inches='tight', dpi=600)  

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

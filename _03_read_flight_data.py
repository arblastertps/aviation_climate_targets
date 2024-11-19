import pandas as pd
import numpy as np
import os

# files downloaded from https://www.eurocontrol.int/dashboard/rnd-data-archive
file_march = 'Flights_20190301_20190331.csv'
file_june = 'Flights_20190601_20190630.csv'
file_sept = 'Flights_20190901_20190930.csv'

files = [file_march, file_june, file_sept]

nb_types = ['B738', # Boeing 737-800
    'A320', # Airbus A320
    'A321', # Airbus A321
    'A319', # Airbus A319
    'B737', # Boeing 737-700 / -700LR
    'B739', # Boeing 737-900 / -900ER
    'B752', # Boeing 757-200
    'A20N' # Airbus A320neo
    ]

nb_seats = [174, 169, 196, 137, 140, 187, 188, 180] # also from the ICCT

wb_types = ['B773', # Boeing 777-300 / -300ER
    'A333', # Airbus A330-300 
    'A388', # Airbus A380-800 
    'A332', # Airbus A330-200 
    'B77L', # Boeing 777-200 / -200ER / -200LR 
    'B789', # Boeing 787-9 
    'B788', # Boeing 787-8 
    'A359', # Airbus A350-900
    'B763', # Boeing 767-300 / -300ER
    'B744' # Boeing 747-400
    ]

wb_seats = [353, 298, 500, 270, 315, 284, 257, 302, 246, 366] # also from the ICCT

occupation = 0.8

flight_types = ['Traditional Scheduled', 'Lowcost'] # no All-Cargo flights, which are quite a different segment from passenger transport / also no business aviation or chartered flights

#airport_code = ['E', 'L', 'B'] # Countries covered by Destination 2050 have airport codes starting with these letters. However, this method also selects some additional countries, such as Greenland and Montenegro. 
airport_code =['E',
               'BI', # iceland
               'LB',
               'LC',
               'LD',
               'LE',
               'LF',
               'LG',
               'LH',
               'LI',
               'LJ',
               'LK', # skip LL, Israel
               'LM', # skip LN, Monaco
               'LO',
               'LP',
               'LR',
               'LS', # switzerland & liechtenstein
               'LU', # skip LT, Turkey; LU, Moldova; LV, Palestinian territories; LW, North Macedonia; LX, Gibraltar; LY, Serbia and Montenegro
               'LZ']

def cut_down(df):
    nm_to_km = 1.852 # 1 nm = 1.852 km
    df['Actual Distance Flown (km)'] = df['Actual Distance Flown (nm)']*nm_to_km
    df = df[['ADEP', 'ADES', 'AC Type', 'STATFOR Market Segment', 'Actual Distance Flown (km)']]
    return df

def select_place_type(df, airport_code, ac_type, flight_types):
    df = df[df['ADEP'].map(lambda x: x.startswith(tuple(airport_code)))] # only consider flights departing from Europe
    df = df[df.isin(ac_type).values]
    return df[df.isin(flight_types).values]

def read_file(file_name):
    df = pd.read_csv(file_name)
    df = cut_down(df)
    return df

def process_files(files, airport_code, ac_type, flight_types, seats, occupation, bins):
    frames = []
    for file in files:
        df = read_file(file)
        frames.append(select_place_type(df, airport_code, ac_type, flight_types))
      
    result = pd.concat(frames)
    result.index = np.arange(len(result))
  
    for i in range(len(ac_type)):
        mask = (result['AC Type'] == ac_type[i])
        ac_here = result[mask]
        result.loc[mask, 'Flight RPK'] = ac_here['Actual Distance Flown (km)']*seats[i]*occupation
        
    result['Bins'] = pd.cut(result['Actual Distance Flown (km)'], bins=bins)

    return result

def split_intra_leaving(df, airport_code):
    intra = df[df['ADES'].map(lambda x: x.startswith(tuple(airport_code)))]
    leaving = df[df['ADES'].map(lambda x: not x.startswith(tuple(airport_code)))] 
    return intra, leaving

#%% make a plot of the bins as they are implemented in the prospective LCA
bins_nb = [0, 1000, 2000, 3000, 20000]
bins_wb = [0, 4000, 6000, 8000, 20000]

df_nb = process_files(files, airport_code, nb_types, flight_types, nb_seats, occupation, bins_nb)
df_wb = process_files(files, airport_code, wb_types, flight_types, wb_seats, occupation, bins_wb)

df_nb_intra, df_nb_extra = split_intra_leaving(df_nb, airport_code)
df_wb_intra, df_wb_extra = split_intra_leaving(df_wb, airport_code)

nb_intra_total_rpk = df_nb_intra.groupby('Bins')['Flight RPK'].sum()*4 # multiply by 4 to represent full year
nb_extra_total_rpk = df_nb_extra.groupby('Bins')['Flight RPK'].sum()*4 # multiply by 4 to represent full year
wb_intra_total_rpk = df_wb_intra.groupby('Bins')['Flight RPK'].sum()*4 # multiply by 4 to represent full year
wb_extra_total_rpk = df_wb_extra.groupby('Bins')['Flight RPK'].sum()*4 # multiply by 4 to represent full year
nb_total_rpk = pd.DataFrame(np.array([nb_intra_total_rpk, nb_extra_total_rpk]).T, columns = ['Intra-Europe narrow-body aircraft', 'Extra-Europe narrow-body aircraft'], index = nb_intra_total_rpk.index)
wb_total_rpk = pd.DataFrame(np.array([wb_intra_total_rpk, wb_extra_total_rpk]).T, columns = ['Intra-Europe wide-body aircraft', 'Extra-Europe wide-body aircraft'], index = wb_intra_total_rpk.index)
ax1 = nb_total_rpk.plot.bar(stacked = True, color = ['#be1908', '#f46e32'])
ax2 = wb_total_rpk.plot.bar(stacked = True, color = ['#34a3a9', '#5cb1eb'])
ax1.set_ylabel('RPK in 2019')
ax1.legend(loc='upper left', frameon=False)
ax2.set_ylabel('RPK in 2019')
ax2.legend(loc='upper left', frameon=False)
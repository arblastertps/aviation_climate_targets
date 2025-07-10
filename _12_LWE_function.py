import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
import warnings
warnings.filterwarnings("ignore")
import itertools

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

# returns linear operator to convert emissions to forcing
# from Lynch et al. 2021
def EFmod(nyr,a):
    Fcal = np.zeros((nyr)) # create linear operator to convert emissions to forcing
    time = np.arange(nyr+1)    # extend time array to compute derivatives
    F_0 = a[4]*a[13]*a[0]*time # compute constant term (if there is one, otherwise a[0]=0)
    for j in [1,2,3]:          # loop over gas decay terms to calculate AGWP using AR5 formula
        F_0=F_0+a[j]*a[4]*a[13]*a[j+5]*(1-np.exp(-time/a[j+5]))
    for i in range(0,nyr):     # first-difference AGWP to obtain AGFP
        Fcal[i]=F_0[i+1]-F_0[i]

    return toeplitz(Fcal, np.zeros_like(Fcal))

def emissions_to_LWE(df_emissions, start, end):
    # Radiative forcing for SLCPs
    # mW/m^2/Mt

    # we rename the columns for simplicity
    df = df_emissions.copy()
    df.columns = ['Tg/year', 'HFC-152a', 'H',
                   'HCFC-140', 'HCFC-22', 'CH4', 'HFC-134a', 'R-10',
                   'HFC-125', 'CFC-11', 'HFC-143a', 'CFC-113',
                   'CO2', 'CO2 (flight)', 'NOx',
                   'BC', 'SOx', 'H2O', 'Cirrus']
    df['H2O'] = df['H2O']*1000 # unit conversion from m3 to kg
    
    df = df.fillna(0)
    
    RFI = {
        "low": {
            "HFC-152a":8.71e-12 * 1e9 * 1e3,
            "H": 3.64e-13 * 1e9 * 1e3,
            "HCFC-140": 2.75e-12 * 1e9 * 1e3,
            "HCFC-22": 1.4e-11 * 1e9 * 1e3,
            "CH4": 1.36e-13 * 1e9 * 1e3,
            "HFC-134a": 9.23e-12 * 1e9 * 1e3,
            "R-10": 6.09e-12 * 1e9 * 1e3,
            "HFC-125": 1.1e-11 * 1e9 * 1e3,
            "CFC-11": 1.06e-11 * 1e9 * 1e3,
            "HFC-143a": 1.13e-11 * 1e9 * 1e3,
            "CFC-113": 9.06e-12 * 1e9 * 1e3,
            "CO2": 1.70E-15 * 1e9 * 1e3,
            "NOx": -7.87 * (14/46), # RF coeff. for NOx
            "BC": 7.95, # RF coeff. for BC
            "SOx": -49.78, # RF coeff. for SO4
            "H2O": 0.0021, # RF coeff. for H2O
            "Cirrus": 6.3e-10# RF coeff. for Cirrus
        },
        "medium": {
            "HFC-152a": 8.71e-12 * 1e9 * 1e3,
            "H": 3.64e-13 * 1e9 * 1e3,
            "HCFC-140": 2.75e-12 * 1e9 * 1e3,
            "HCFC-22": 1.4e-11 * 1e9 * 1e3,
            "CH4": 1.36e-13 * 1e9 * 1e3,
            "HFC-134a": 9.23e-12 * 1e9 * 1e3,
            "R-10": 6.09e-12 * 1e9 * 1e3,
            "HFC-125": 1.1e-11 * 1e9 * 1e3,
            "CFC-11": 1.06e-11 * 1e9 * 1e3,
            "HFC-143a": 1.13e-11 * 1e9 * 1e3,
            "CFC-113": 9.06e-12 * 1e9 * 1e3,
            "CO2": 1.70E-15 * 1e9 * 1e3,
            "NOx": 5.5 * (14/46), # RF coeff. for NOx
            "BC": 100.67, # RF coeff. for BC
            "SOx": -19.91, # RF coeff. for SO4
            "H2O": 0.0052, # RF coeff. for H2O
            "Cirrus": 9.36e-10 # RF coeff. for Cirrus
        },
        "high": {
            "HFC-152a":8.71e-12 * 1e9 * 1e3,
            "H": 3.64e-13 * 1e9 * 1e3,
            "HCFC-140": 2.75e-12 * 1e9 * 1e3,
            "HCFC-22": 1.4e-11 * 1e9 * 1e3,
            "CH4": 1.36e-13 * 1e9 * 1e3,
            "HFC-134a": 9.23e-12 * 1e9 * 1e3,
            "R-10": 6.09e-12 * 1e9 * 1e3,
            "HFC-125": 1.1e-11 * 1e9 * 1e3,
            "CFC-11": 1.06e-11 * 1e9 * 1e3,
            "HFC-143a": 1.13e-11 * 1e9 * 1e3,
            "CFC-113": 9.06e-12 * 1e9 * 1e3,
            "CO2": 1.70E-15 * 1e9 * 1e3,
            "NOx": 12.57 * (14/46), # RF coeff. for NOx
            "BC": 428.65, # RF coeff. for BC
            "SOx": -6.87, # RF coeff. for SO4
            "H2O": 0.0083, # RF coeff. for H2O
            "Cirrus": 1.39e-9 # RF coeff. for Cirrus
        }
    }
    
    # molecular mass for SLCPs
    # kg/mol
    mol_mass = {
        "HFC-152a":66.05 / 1e3,
        "H": 1.01 / 1e3,
        "HCFC-140": 133.4 / 1e3,
        "HCFC-22": 86.47 / 1e3,
        "CH4": 16.04 / 1e3,
        "HFC-134a": 102.03 / 1e3,
        "R-10": 153.823 / 1e3,
        "HFC-125": 120.02 / 1e3,
        "CFC-11": 137.37 / 1e3,
        "HFC-143a": 84.04 / 1e3,
        "CFC-113": 187.375 / 1e3,
        'NOx': 14/1e3,
        'BC': 12/1e3, 
        'SOx': 64/1e3, 
        'H2O': 18/1e3,
        "Cirrus": 0
    }
    
    # lifetimes for SLCPs
    # years
    RF_lifetime = {
        "HFC-152a":1.6,
        "H": 2.5,
        "HCFC-140": 5,
        "HCFC-22": 11.9,
        "CH4": 11.8,
        "HFC-134a": 14,
        "R-10": 32,
        "HFC-125": 30,
        "CFC-11": 52,
        "HFC-143a": 51,
        "CFC-113": 93,
        'NOx': 0.267, # changed from 11.8, based on Fuglestvedt et al. (2010)
        'BC': 0.02, 
        'SOx': 0.011, 
        'H2O': 0.8,
        "Cirrus": 0.00057
    }
    
    # first set up AR5 model parameters, using syntax of FaIRv1.3 but units of GtCO2, not GtC
    ny2=len(range(start, end + 1))
    
    m_atm=5.1352*10**18 # AR5 official mass of atmosphere in kg
    m_air=28.97*10**-3  # AR5 official molar mass of air
    m_co2=44.01*10**-3  # AR5 official molar mass of CO2
    
    a_ar5=np.zeros(20)
    
    # Set to AR5 Values for CO2
    a_ar5[0:4] = [0.21787,0.22896,0.28454,0.26863]
    a_ar5[4] = 1.e12*1.e6/m_co2/(m_atm/m_air)# old value = 0.471 ppm/GtC # convert GtCO2 to ppm
    a_ar5[5:9] = [1.e8,381.330,34.7850,4.12370]
    a_ar5[10:12] = [0.631*0.7,0.429*0.7] #AR5 sensitivity coeffs multiplied by 0.7 to give ECS of 2.75K
    a_ar5[13] = 1.37e-2 # rad efficiency in W/m2/ppm
    a_ar5[14] = 0
    a_ar5[15:17] = [8.400,409.5]
    a_ar5[18:21] = 0
    
    FCO2 = EFmod(ny2,a_ar5)
    FCO2_inv = np.linalg.inv(FCO2)
    
    RF = pd.DataFrame(
        columns=[
            "net CO2",
            "surface - Others",
            "flight - Cirrus",
            "flight - NOx",
            "flight - Others"
        ],
        index = range(start, end + 1)
    )

    
    RF.loc[:, :] = 0
    RF_low = RF.copy()
    RF_high = RF.copy()
    RF_low.loc[:, :] = 0
    RF_high.loc[:, :] = 0
    
    d_map = {
        "surface - CO2": "CO2",
        "surface - Others": ['HFC-152a', 'H', 'HCFC-140', 'HCFC-22', 'CH4', 'HFC-134a', 'R-10', 'HFC-125', 'CFC-11', 'HFC-143a', 'CFC-113',],
        "flight - CO2": "CO2 (flight)",
        "flight - Cirrus": "Cirrus",
        "flight - NOx": "NOx",
        "flight - Others": ['BC', 'SOx', 'H2O']
    }
    d_map_rev = {
        "CO2": "net CO2",
        'HFC-152a': "surface - Others", 
        'H': "surface - Others", 
        'HCFC-140': "surface - Others", 
        'HCFC-22': "surface - Others", 
        'CH4': "surface - Others", 
        'HFC-134a': "surface - Others", 
        'R-10': "surface - Others", 
        'HFC-125': "surface - Others", 
        'CFC-11': "surface - Others", 
        'HFC-143a': "surface - Others", 
        'CFC-113': "surface - Others",
        "CO2 (flight)": "net CO2",
        "Cirrus": "flight - Cirrus",
        "NOx": "flight - NOx",
        'BC': "flight - Others", 
        'SOx': "flight - Others", 
        'H2O': "flight - Others"
    }
    
    # slice data of interest
    data = df.loc[:, :'Cirrus']
    data = data.loc[data["Tg/year"].isin(range(start, end + 1))]
    
    for r in data.loc[:, 'HFC-152a': 'Cirrus'].columns:
        if r not in ["CO2", "CO2 (flight)"]:
            # Set to AR6 Values for substance
            a_sub=a_ar5.copy()
            a_sub[0:4]=[0,1.0,0,0]
            a_sub[4]= 1 # Mt
            a_sub[5:9]= [1, RF_lifetime[r], 1, 1]
            a_sub[13]= RFI["medium"][r] / 1e3 # Radiative efficiency in W/m2/Mton -- division by 1e3 to convry mW to W
            
            a_sub_low = a_sub.copy()
            a_sub_low[13] = RFI["low"][r] / 1e3 # Radiative efficiency in W/m2/Mton -- division by 1e3 to convry mW to W
            
            a_sub_high = a_sub.copy()
            a_sub_high[13] = RFI["high"][r] / 1e3 # Radiative efficiency in W/m2/Mton -- division by 1e3 to convry mW to W
    
            Fsub = EFmod(ny2, a_sub)
            Fsub_low = EFmod(ny2, a_sub_low)
            Fsub_high = EFmod(ny2, a_sub_high)
            
            if r != "Cirrus":
                # LWE = FCO2^-1 * Fsub * Esub
                RF.loc[:, d_map_rev[r]] += Fsub@(data.loc[:, r] / 1e9 * 1e3) # <-- W to mW
                RF_low.loc[:, d_map_rev[r]] += Fsub_low@(data.loc[:, r] / 1e9 * 1e3) # <-- W to mW
                RF_high.loc[:, d_map_rev[r]] += Fsub_high@(data.loc[:, r] / 1e9 * 1e3) # <-- W to mW
                
            else:
                RF.loc[:, "flight - Cirrus"] = (data.loc[:, r]).values * RFI["medium"][r]
                RF_low.loc[:, "flight - Cirrus"] = (data.loc[:, r]).values * RFI["low"][r]
                RF_high.loc[:, "flight - Cirrus"] = (data.loc[:, r]).values * RFI["high"][r]
                
        else:
            # LWE = FCO2^-1 * Fsub * Esub
            RF.loc[:, d_map_rev[r]] += FCO2@(data.loc[:, r] / 1e9).values # <-- W to mW
            RF_low.loc[:, d_map_rev[r]] += FCO2@(data.loc[:, r] / 1e9).values # <-- W to mW
            RF_high.loc[:, d_map_rev[r]] += FCO2@(data.loc[:, r] / 1e9).values # <-- W to mW

    return RF.astype(float), RF_low.astype(float), RF_high.astype(float)
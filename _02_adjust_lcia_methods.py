import bw2data
import bw2io

# code to create custom "LCIA" methods which simply aggregate LCI data across (sub)compartments

# there are minor changes between ecoinvent 3.8 and 3.9 (as well as, potentially, difference in reference method)
project_version = 'ecoinvent 3.8'
# project_version = 'ecoinvent 3.9'

if project_version == 'ecoinvent 3.8':
    bw2data.projects.set_current("[specify project name!]") # name of project with ecoinvent 3.8 database
    db = bw2data.Database("biosphere3") 
    original_method = bw2data.Method(("IPCC 2013", "climate change", "GWP 100a, incl. H and bio CO2"))
    
if project_version == 'ecoinvent 3.9':
    bw2data.projects.set_current("[specify project name!]") # name of project with ecoinvent 3.9 database
    db = bw2data.Database("biosphere3")
    original_method = bw2data.Method(("IPCC 2021", "climate change", "GWP 100a, incl. H and bio CO2"))

# to avoid selecting unwanted flows, a reference method is chosen which includes the majority of flows of interest
# note that in this case, a method is used which includes hydrogen and biogenic CO2
original_list = []
for a,b in original_method.load():
    original_list.append([a,b])

# prepare lists for flows to be collected
flows_co2 = []
flows_methane = []
flows_152a = []
flows_hydrogen = []
flows_140 = []
flows_22 = []
flows_134a = []
flows_R_10 = []
flows_125 = []
flows_11 = []
flows_143a = []
flows_113 = []
flows_nox = []
flows_pm = []
flows_sox = []
flows_water = []

# populate above lists by cycling through biosphere database
all_names = []
for flow in db:
    # collect flows also present in reference method
    for a,b in original_list: 
        if flow.key == a:
            all_names.append(flow['name'])
            if flow['name'][:14] == 'Carbon dioxide':
                flows_co2.append([a, b])
            if flow['name'][:10] == 'Methane, f' or flow['name'][:10] == 'Methane, n':
                flows_methane.append([a, 1])
            if flow['name'] == 'Ethane, 1,1-difluoro-, HFC-152a':
                flows_152a.append([a, 1])
            if flow['name'] == 'Hydrogen':
                flows_hydrogen.append([a, 1])
            if flow['name'] == 'Ethane, 1,1,1-trichloro-, HCFC-140':
                flows_140.append([a, 1])
            if flow['name'] == 'Methane, chlorodifluoro-, HCFC-22':
                flows_22.append([a, 1])      
            if flow['name'] == 'Ethane, 1,1,1,2-tetrafluoro-, HFC-134a':
                flows_134a.append([a, 1])    
            if flow['name'] == 'Methane, tetrachloro-, R-10':
                flows_R_10.append([a, 1])  
            if flow['name'] == 'Ethane, pentafluoro-, HFC-125':
                flows_125.append([a, 1])  
            if flow['name'] == 'Methane, trichlorofluoro-, CFC-11':
                flows_11.append([a, 1])  
            if flow['name'] == 'Ethane, 1,1,1-trifluoro-, HFC-143a':
                flows_143a.append([a, 1])  
            if flow['name'] == 'Ethane, 1,1,2-trichloro-1,2,2-trifluoro-, CFC-113':
                flows_113.append([a, 1])
    # collect flows not present in reference method
    if flow['name'] == 'Nitrogen oxides':
        flows_nox.append([flow.key,1])
    if flow['name'][:11] == 'Particulate':
        flows_pm.append([flow.key,1])
    if flow['name'] == 'Sulfur dioxide':
        flows_sox.append([flow.key,1])
    if flow['name'] == 'Water' and str(flow['categories'])[:7] == "('air',":
        flows_water.append([flow.key,1])

# function to write method with specified name to the active project under the new method group "selected LCI results, custom"
def val_and_write(name, flow_list):
    method = bw2data.Method(("selected LCI results, custom", name))
    method.validate(flow_list)
    method.register() 
    method.write(flow_list)

# execute above function for all flows of interest
val_and_write('Carbon dioxide', flows_co2)
val_and_write('Methane', flows_methane)
val_and_write('Ethane, 1,1-difluoro-, HFC-152a', flows_152a)
val_and_write('Hydrogen', flows_hydrogen)
val_and_write('Ethane, 1,1,1-trichloro-, HCFC-140', flows_140)
val_and_write('Methane, chlorodifluoro-, HCFC-22', flows_22)
val_and_write('Ethane, 1,1,1,2-tetrafluoro-, HFC-134a', flows_134a)
val_and_write('Methane, tetrachloro-, R-10', flows_R_10)
val_and_write('Ethane, pentafluoro-, HFC-125', flows_125)
val_and_write('Methane, trichlorofluoro-, CFC-11', flows_11)
val_and_write('Ethane, 1,1,1-trifluoro-, HFC-143a', flows_143a)
val_and_write('Ethane, 1,1,2-trichloro-1,2,2-trifluoro-, CFC-113', flows_113) 
val_and_write('Nitrogen oxides', flows_nox)
val_and_write('Particulate Matter', flows_pm)
val_and_write('Sulfur dioxide', flows_sox)
val_and_write('Water', flows_water)
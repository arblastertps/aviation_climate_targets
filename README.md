# General overview

This file elaborates on the files in this repository and their use. In addition to this file, there are several file types.
First, files containing python scripts, being:
- `_01_running_premise.py`
- `_02_adjust_lcia_methods.py`
- `_03_read_flight_data.py`
- `_10_functions.py`
- `_11_define_scenarios.py`
- `_12_LWE_function.py`
- `_20_run_scenarios.py`
- `_21_plot_scenarios.py`

Secondly, files describing the packages present in the environments used to originally execute the python scripts:
- `env-premise.yml`
- `env-bw.yml`
- `env-ab.yml`
- `env-gen.yml`

Third of all, Microsoft Excel Workbooks (`.xlsx` files, most in `LCIA_building_blocks`):
- `foreground-LCI-data.xlsx`
- `aircraft-SSP2-NDC.xlsx`
- `plants-SSP2-NDC-wind.xlsx`
- `plants-SSP2-NDC-grid.xlsx`
- `hydrogen-market-SSP2-NDC.xlsx`
- `aircraft-SSP2-PkBudg1150.xlsx`
- `plants-SSP2-PkBudg1150-wind.xlsx`
- `plants-SSP2-PkBudg1150-grid.xlsx`
- `hydrogen-market-SSP2-PkBudg1150.xlsx`
- `aircraft-SSP1-PkBudg500.xlsx`
- `plants-SSP1-PkBudg500-wind.xlsx`
- `plants-SSP1-PkBudg500-grid.xlsx`
- `hydrogen-market-SSP1-PkBudg500.xlsx`

# Overview of Microsoft Excel Workbook files

### `foreground-LCI-data.xlsx`
This file describes the economic and environmental flows of activities in the foreground system.
Note that not all activities of the system (as illustrated in product system figure) are included in this overview, as these activities are created later, with the `_1X` python scripts.
Economic flows which cannot be followed further in this overview connect to a background database.
Background databases are generated using premise (see `01_running_premise.py`).
Premise is used to transform the ecoinvent database, but also adds additional unit processes.

### All other `.xlsx` files
These files are named using the format `[products]-[scenario].xlsx`.
They are created as exports from the Activity Browser and as intermediate building block for the full LCA.
Executing scenarios requires specifying a combination of these files to use, as described in `20_run_scenarios.py`.

# Overview of python files with corresponding environments

### `_01_running_premise.py` (environment: see `env-premise.yml`)
Uses the premise library to create a database with superstructure file. 
The superstructure file allows dynamic use of the database for 2020-2080 with a 5-year time interval.
The script is primed for the SSP2-PkBudg1150 pathway as described by REMIND, but can also be used for other pathways.
Note that this script cannot be run without a premise key and access to the ecoinvent database.

### `_02_adjust_lcia_methods.py` (environment: see `env-bw.yml`)
Used to create custom LCIA methods which act to aggregate LCI flows across various (sub)compartments for ease of further handling.
Running this script requires that the premise impact categories have been added to the project, but it is easily adjusted to remove this dependency.

### `_03_read_flight_data.py` (environment: see `env-gen.yml`)
Used to process data from the EUROCONTROL aviation data for research database.
This generates the RPK of interest for 2019, i.e., limited to commercial passenger flights departing from the selected countries.
Note that to run this script, the EUROCONTROL data must be downloaded and the file locations specified.

### `_10_functions.py` (environment: see `env-gen.yml`)
All functions used to execute a scenario with defined scenario variables are contained in this script.

### `_11_define_scenarios.py` (environment: see `env-gen.yml`)
This script defines the definition of scenario variables.
Meaning that, when other scripts refer to e.g. "low" or "high" development for a technology, the meaning of this is given by this script.

### `_12_LWE_function.py` (environment: see `env-gen.yml`)
The functions used in applying the LWE method are contained in this script.

### `_20_run_scenarios.py` (environment: see `env-gen.yml`)
This is the central script for both executing scenarios and visualising them.
It can make use of the `.pkl` files included in this repository.

### `_21_plot_scenarios.py` (environment: see `env-gen.yml`)
This script contains a variety of functions used in plotting scenario results.
Note that adjustments must be made in both 

# Description of workflow
This workflow assumes that the `[products]-[scenario].xlsx` files in this repository are being used.
The scripts with `_0X` numbering are used in the generation of data that has been copied to subsequent scripts and in the creation of these `.xlsx` files, in combination with the LCI data described in `foreground-LCI-data.xlsx`. 
Therefore, the scripts with `_0X` numbering are not utilised in the workflow described below.

1. Create a python environment to use, analogue to the one described in `env-gen.yml`.
2. To a folder of choice, download the scripts with `_1X` and `_2X` numbering and the `LCIA_building_blocks` folder, containing the `[products]-[scenario].xlsx` files.
3. In `11_define_scenarios.py`, adjust the definition of scenario variables, if desired *(see note 1 below)*.
4. In `20_run_scenarios.py`, choose which scenario(s) to model.
5. In `20_run_scenarios.py` and `21_plot_scenarios.py`, choose which scenario(s) to visualise *(see note 2 below)*.
6. In `20_run_scenarios.py`, choose which visualisations to use.
7. Execute `20_run_scenarios.py`.

**Note 1**: Depending on the number of scenarios selected, this step can save a lot of time. To save time in repeated runs, the code is set up to save scenario results in `.pkl` files. Such files are available upon request, but are excluded in this repository due to the files' prohibitive size.

**Note 2**: In most functions of `21_plot_scenarios.py`, the figure labels are currently hard-coded to match the illustrative scenarios selected in the text.
If you want to use these functions for other scenarios, the labels must be individually adjusted.
from premise import *
import bw2data

bw2data.projects.set_current("[specify project name!]")

ndb = NewDatabase(
    scenarios = [
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2020},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2025},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2030},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2035},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2040},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2045},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2050},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2055},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2060},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2065},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2070},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2075},
        {"model":"remind", "pathway":"SSP2-PkBudg1150", "year":2080},
                ],
    source_db="cutoff391", # <-- name of the database in the BW2 project. Must be a string.
    source_version="3.9.1", # <-- version of ecoinvent. Must be a string.
    key='...', # <-- decryption key
    use_multiprocessing=False,
)

ndb.update_all()

ndb.write_superstructure_db_to_brightway(name="premise REMIND SSP2-PkBudg1150")
ndb.generate_scenario_report()

# to make premise-specific characterisation model (that includes climate impact of H2 emissions):
from premise_gwp import add_premise_gwp
add_premise_gwp()
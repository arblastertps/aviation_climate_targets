import pandas as pd
import numpy as np
from typing import Tuple

class ModelParameters:
    def __init__(self, flight_start_year, flight_end_year):
        self.climate_historic_start_year = flight_start_year
        self.historic_start_year = flight_start_year
        self.prospection_start_year = flight_start_year
        self.end_year = flight_end_year

# everything below is adapted from the AeroMAPS project, v0.8.3-beta
# https://github.com/AeroMAPS/AeroMAPS
# note: inclusion of impacts from CH4 and H2 are not in the original

# TCRE & ERF
tcre_coefficient = 0.00045
erf_coefficient_contrails = 1.058e-09
erf_coefficient_h2o = 0.0052
erf_coefficient_nox = 11.55
erf_coefficient_nox_short_term_o3_increase = 40.9
erf_coefficient_nox_long_term_o3_decrease = -8.7
erf_coefficient_nox_ch4_decrease = -17.6
erf_coefficient_nox_stratospheric_water_vapor_decrease = -2.6
erf_coefficient_soot = 100.7
erf_coefficient_sulfur = -19.9

# 'updated settings' from Arriolabengoa et al. (2024)
contrails_gwpstar_variation_duration = 1.0
contrails_gwpstar_s_coefficient = 0.0
nox_short_term_o3_increase_gwpstar_variation_duration = 1.0
nox_short_term_o3_increase_gwpstar_s_coefficient = 0.0
nox_long_term_o3_decrease_gwpstar_variation_duration = 20.0
nox_long_term_o3_decrease_gwpstar_s_coefficient = 0.25
nox_ch4_decrease_gwpstar_variation_duration = 20.0
nox_ch4_decrease_gwpstar_s_coefficient = 0.25
nox_stratospheric_water_vapor_decrease_gwpstar_variation_duration = 20.0
nox_stratospheric_water_vapor_decrease_gwpstar_s_coefficient = 0.25
soot_gwpstar_variation_duration = 1.0
soot_gwpstar_s_coefficient = 0.0
h2o_gwpstar_variation_duration = 1.0
h2o_gwpstar_s_coefficient = 0.0
sulfur_gwpstar_variation_duration = 1.0
sulfur_gwpstar_s_coefficient = 0.0

# additional impacts: CH4 & H2
# methane: follows Smith et al. (2021)
erf_coefficient_ch4 = 2.00e-13 * 1e9 * 1e3 # AR6: 5.7e-4 W m-2 ppb-1 -- converted to W m-2 kg-1 using molar mass (16.04) and 1.773e20 moles of atmosphere, then converted to mW m-2 Tg-1
ch4_gwpstar_variation_duration = 20
ch4_gwpstar_s_coefficient = 0.25

# hydrogen: follows Paulot et al. (2021)
erf_coefficient_h2 = 0.84 # already reported in mW m-2 Tg-1
h2_gwpstar_variation_duration = 20
h2_gwpstar_s_coefficient = 0.25

# # original settings
# contrails_gwpstar_variation_duration = 6.0
# contrails_gwpstar_s_coefficient = 0.0
# nox_short_term_o3_increase_gwpstar_variation_duration = 6.0
# nox_short_term_o3_increase_gwpstar_s_coefficient = 0.0
# nox_long_term_o3_decrease_gwpstar_variation_duration = 20.0
# nox_long_term_o3_decrease_gwpstar_s_coefficient = 0.25
# nox_ch4_decrease_gwpstar_variation_duration = 20.0
# nox_ch4_decrease_gwpstar_s_coefficient = 0.25
# nox_stratospheric_water_vapor_decrease_gwpstar_variation_duration = 20.0
# nox_stratospheric_water_vapor_decrease_gwpstar_s_coefficient = 0.25
# soot_gwpstar_variation_duration = 6.0
# soot_gwpstar_s_coefficient = 0.0
# h2o_gwpstar_variation_duration = 6.0
# h2o_gwpstar_s_coefficient = 0.0
# sulfur_gwpstar_variation_duration = 6.0
# sulfur_gwpstar_s_coefficient = 0.0

class AeroMAPSModel(object):
    def __init__(
        self,
        name,
        parameters=None,
    ):
        self.name = name
        self.parameters = parameters
        self.float_outputs = {}
        if self.parameters is not None:
            self._initialize_df()

    def _initialize_df(self):
        self.climate_historic_start_year = self.parameters.climate_historic_start_year
        self.historic_start_year = self.parameters.historic_start_year
        self.prospection_start_year = self.parameters.prospection_start_year
        self.end_year = self.parameters.end_year
        self.df: pd.DataFrame = pd.DataFrame(
            index=range(self.historic_start_year, self.end_year + 1)
        )
        self.df_climate: pd.DataFrame = pd.DataFrame(
            index=range(self.climate_historic_start_year, self.end_year + 1)
        )
        self.years = np.linspace(self.historic_start_year, self.end_year, len(self.df.index))

def AbsoluteGlobalWarmingPotentialCO2Function(climate_time_horizon):
    # Reference: IPCC AR5 - https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf

    # Parameter: climate time horizon
    h = climate_time_horizon

    co2_molar_mass = 44.01 * 1e-3  # [kg/mol]
    air_molar_mass = 28.97e-3  # [kg/mol]
    atmosphere_total_mass = 5.1352e18  # [kg]

    radiative_efficiency = 1.37e-2 * 1e9  # radiative efficiency [mW/m^2]

    # RF per unit mass increase in atmospheric abundance of CO2 [W/m^2/kg]
    A_CO2 = radiative_efficiency * air_molar_mass / (co2_molar_mass * atmosphere_total_mass) * 1e-3

    # Coefficients for the model
    a = [0.2173, 0.2240, 0.2824, 0.2763]
    tau = [0, 394.4, 36.54, 4.304]  # CO2 lifetime [yrs]

    co2_agwp_h = A_CO2 * a[0] * h
    for i in [1, 2, 3]:
        co2_agwp_h += A_CO2 * a[i] * tau[i] * (1 - np.exp(-h / tau[i]))

    # From W/m^2/kg.yr to mW/m^2/Mt.yr
    co2_agwp_h = co2_agwp_h * 1e3 * 1e9

    return co2_agwp_h

def GWPStarEquivalentEmissionsFunction(
    self, emissions_erf, gwpstar_variation_duration, gwpstar_s_coefficient
):
    # Reference: Smith et al. (2021), https://doi.org/10.1038/s41612-021-00169-8
    # Global
    climate_time_horizon = 100
    co2_agwp_h = AbsoluteGlobalWarmingPotentialCO2Function(climate_time_horizon)

    # g coefficient for GWP*
    if gwpstar_s_coefficient == 0:
        g_coefficient = 1
    else:
        g_coefficient = (
            1 - np.exp(-gwpstar_s_coefficient / (1 - gwpstar_s_coefficient))
        ) / gwpstar_s_coefficient

    # Main
    for k in range(self.climate_historic_start_year, self.end_year + 1):
        if k - self.climate_historic_start_year >= gwpstar_variation_duration:
            self.df_climate.loc[k, "emissions_erf_variation"] = (
                emissions_erf.loc[k] - emissions_erf.loc[k - gwpstar_variation_duration]
            ) / gwpstar_variation_duration
        else:
            self.df_climate.loc[k, "emissions_erf_variation"] = (
                emissions_erf.loc[k] / gwpstar_variation_duration
            )

    for k in range(self.climate_historic_start_year, self.end_year + 1):
        self.df_climate.loc[k, "emissions_equivalent_emissions"] = (
            g_coefficient
            * (1 - gwpstar_s_coefficient)
            * climate_time_horizon
            / co2_agwp_h
            * self.df_climate.loc[k, "emissions_erf_variation"]
        ) + g_coefficient * gwpstar_s_coefficient / co2_agwp_h * emissions_erf.loc[k]
    emissions_equivalent_emissions = self.df_climate.loc[:, "emissions_equivalent_emissions"]

    # Delete intermediate df column
    self.df_climate.pop("emissions_erf_variation")
    self.df_climate.pop("emissions_equivalent_emissions")

    return emissions_equivalent_emissions

class SimplifiedERFCo2(AeroMAPSModel):
    def __init__(self, name="simplified_effective_radiative_forcing_co2", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def compute(
        self,
        co2_emissions: pd.Series,
    ) -> Tuple[
        pd.Series,
        pd.Series,
    ]:
        """ERF calculation for CO2 emissions with a simplified method."""

        # CO2
        h = 100  # Climate time horizon
        for k in range(self.climate_historic_start_year, self.end_year + 1):
            self.df_climate.loc[k, "annual_co2_erf"] = (
                co2_emissions.loc[k] * AbsoluteGlobalWarmingPotentialCO2Function(h) / h
            )
        self.df_climate.loc[self.climate_historic_start_year, "co2_erf"] = self.df_climate.loc[
            self.climate_historic_start_year, "annual_co2_erf"
        ]
        for k in range(self.climate_historic_start_year + 1, self.end_year + 1):
            self.df_climate.loc[k, "co2_erf"] = (
                self.df_climate.loc[k - 1, "co2_erf"] + self.df_climate.loc[k, "annual_co2_erf"]
            )
        annual_co2_erf = self.df_climate["annual_co2_erf"]
        co2_erf = self.df_climate["co2_erf"]

        return (
            annual_co2_erf,
            co2_erf,
        )

class ERFNox(AeroMAPSModel):
    def __init__(self, name="effective_radiative_forcing_nox", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def compute(
        self,
        nox_emissions: pd.Series,
        erf_coefficient_nox_short_term_o3_increase: float,
        erf_coefficient_nox_long_term_o3_decrease: float,
        erf_coefficient_nox_ch4_decrease: float,
        erf_coefficient_nox_stratospheric_water_vapor_decrease: float,
    ) -> Tuple[
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """ERF calculation for NOx emissions."""

        # NOx
        n_emissions = nox_emissions * 14 / 46  # Molar masses of N and NOx
        self.df_climate["nox_short_term_o3_increase_erf"] = (
            n_emissions * erf_coefficient_nox_short_term_o3_increase
        )
        self.df_climate["nox_long_term_o3_decrease_erf"] = (
            n_emissions * erf_coefficient_nox_long_term_o3_decrease
        )
        self.df_climate["nox_ch4_decrease_erf"] = n_emissions * erf_coefficient_nox_ch4_decrease
        self.df_climate["nox_stratospheric_water_vapor_decrease_erf"] = (
            n_emissions * erf_coefficient_nox_stratospheric_water_vapor_decrease
        )
        nox_short_term_o3_increase_erf = self.df_climate["nox_short_term_o3_increase_erf"]
        nox_long_term_o3_decrease_erf = self.df_climate["nox_long_term_o3_decrease_erf"]
        nox_ch4_decrease_erf = self.df_climate["nox_ch4_decrease_erf"]
        nox_stratospheric_water_vapor_decrease_erf = self.df_climate[
            "nox_stratospheric_water_vapor_decrease_erf"
        ]
        nox_erf = (
            nox_short_term_o3_increase_erf
            + nox_long_term_o3_decrease_erf
            + nox_ch4_decrease_erf
            + nox_stratospheric_water_vapor_decrease_erf
        )
        self.df_climate["nox_erf"] = nox_erf

        return (
            nox_short_term_o3_increase_erf,
            nox_long_term_o3_decrease_erf,
            nox_ch4_decrease_erf,
            nox_stratospheric_water_vapor_decrease_erf,
            nox_erf,
        )


class ERFOthers(AeroMAPSModel):
    def __init__(self, name="effective_radiative_forcing_others", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def compute(
        self,
        soot_emissions: pd.Series,
        h2o_emissions: pd.Series,
        sulfur_emissions: pd.Series,
        erf_coefficient_contrails: float,
        erf_coefficient_soot: float,
        erf_coefficient_h2o: float,
        erf_coefficient_sulfur: float,
        total_aircraft_distance: pd.Series,
        operations_contrails_gain: pd.Series,
        fuel_effect_correction_contrails: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """ERF calculation for the other climate impacts of aviation."""

        # Contrails
        for k in range(self.climate_historic_start_year, self.end_year + 1):
            self.df_climate.loc[k, "contrails_erf"] = (
                total_aircraft_distance.loc[k] * erf_coefficient_contrails
            )
        for k in range(self.historic_start_year, self.end_year + 1):
            self.df_climate.loc[k, "contrails_erf"] = (
                total_aircraft_distance.loc[k]
                * erf_coefficient_contrails
                * (1 - operations_contrails_gain.loc[k] / 100)
                * fuel_effect_correction_contrails.loc[k]
            )
        contrails_erf = self.df_climate["contrails_erf"]

        # Others
        self.df_climate["soot_erf"] = soot_emissions * erf_coefficient_soot
        self.df_climate["h2o_erf"] = h2o_emissions * erf_coefficient_h2o
        self.df_climate["sulfur_erf"] = sulfur_emissions * erf_coefficient_sulfur
        soot_erf = self.df_climate["soot_erf"]
        h2o_erf = self.df_climate["h2o_erf"]
        sulfur_erf = self.df_climate["sulfur_erf"]
        self.df_climate["aerosol_erf"] = soot_erf + sulfur_erf
        aerosol_erf = self.df_climate["aerosol_erf"]

        return (
            contrails_erf,
            soot_erf,
            h2o_erf,
            sulfur_erf,
            aerosol_erf,
        )
    
class ERFExtra(AeroMAPSModel):
    def __init__(self, name="effective_radiative_forcing_extra", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def compute(
        self,
        ch4_emissions: pd.Series,
        h2_emissions: pd.Series,
        erf_coefficient_ch4: float,
        erf_coefficient_h2: float,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """ERF calculation for the other climate impacts of aviation, not included in the original."""

        # Others
        self.df_climate["ch4_erf"] = ch4_emissions * erf_coefficient_ch4
        self.df_climate["h2_erf"] = h2_emissions * erf_coefficient_h2
        ch4_erf = self.df_climate["ch4_erf"]
        h2_erf = self.df_climate["h2_erf"]

        return (
            ch4_erf,
            h2_erf,
        )
    
class TemperatureGWPStar(AeroMAPSModel):
    def __init__(self, name="temperature_gwpstar", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def compute(
        self,
        contrails_gwpstar_variation_duration: float,
        contrails_gwpstar_s_coefficient: float,
        nox_short_term_o3_increase_gwpstar_variation_duration: float,
        nox_short_term_o3_increase_gwpstar_s_coefficient: float,
        nox_long_term_o3_decrease_gwpstar_variation_duration: float,
        nox_long_term_o3_decrease_gwpstar_s_coefficient: float,
        nox_ch4_decrease_gwpstar_variation_duration: float,
        nox_ch4_decrease_gwpstar_s_coefficient: float,
        nox_stratospheric_water_vapor_decrease_gwpstar_variation_duration: float,
        nox_stratospheric_water_vapor_decrease_gwpstar_s_coefficient: float,
        soot_gwpstar_variation_duration: float,
        soot_gwpstar_s_coefficient: float,
        h2o_gwpstar_variation_duration: float,
        h2o_gwpstar_s_coefficient: float,
        sulfur_gwpstar_variation_duration: float,
        sulfur_gwpstar_s_coefficient: float,
        contrails_erf: pd.Series,
        nox_short_term_o3_increase_erf: pd.Series,
        nox_long_term_o3_decrease_erf: pd.Series,
        nox_ch4_decrease_erf: pd.Series,
        nox_stratospheric_water_vapor_decrease_erf: pd.Series,
        soot_erf: pd.Series,
        h2o_erf: pd.Series,
        sulfur_erf: pd.Series,
        co2_erf: pd.Series,
        total_erf: pd.Series,
        co2_emissions: pd.Series,
        tcre_coefficient: float,
        ch4_gwpstar_variation_duration: float,
        ch4_gwpstar_s_coefficient: float,
        ch4_erf: pd.Series,
        h2_gwpstar_variation_duration: float,
        h2_gwpstar_s_coefficient: float,
        h2_erf: pd.Series,
    ) -> Tuple[
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """Temperature calculation using equivalent emissions (with GWP* method) and TCRE."""

        # EQUIVALENT EMISSIONS

        ## Contrails
        contrails_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=contrails_erf,
            gwpstar_variation_duration=contrails_gwpstar_variation_duration,
            gwpstar_s_coefficient=contrails_gwpstar_s_coefficient,
        )
        self.df_climate["contrails_equivalent_emissions"] = contrails_equivalent_emissions

        ## NOx short-term O3 increase
        nox_short_term_o3_increase_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=nox_short_term_o3_increase_erf,
            gwpstar_variation_duration=nox_short_term_o3_increase_gwpstar_variation_duration,
            gwpstar_s_coefficient=nox_short_term_o3_increase_gwpstar_s_coefficient,
        )
        self.df_climate["nox_short_term_o3_increase_equivalent_emissions"] = (
            nox_short_term_o3_increase_equivalent_emissions
        )

        ## NOx long-term O3 decrease
        nox_long_term_o3_decrease_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=nox_long_term_o3_decrease_erf,
            gwpstar_variation_duration=nox_long_term_o3_decrease_gwpstar_variation_duration,
            gwpstar_s_coefficient=nox_long_term_o3_decrease_gwpstar_s_coefficient,
        )
        self.df_climate["nox_long_term_o3_decrease_equivalent_emissions"] = (
            nox_long_term_o3_decrease_equivalent_emissions
        )

        ## NOx CH4 decrease
        nox_ch4_decrease_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=nox_ch4_decrease_erf,
            gwpstar_variation_duration=nox_ch4_decrease_gwpstar_variation_duration,
            gwpstar_s_coefficient=nox_ch4_decrease_gwpstar_s_coefficient,
        )
        self.df_climate["nox_ch4_decrease_equivalent_emissions"] = (
            nox_ch4_decrease_equivalent_emissions
        )

        ## NOx stratospheric water vapor decrease
        nox_stratospheric_water_vapor_decrease_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=nox_stratospheric_water_vapor_decrease_erf,
            gwpstar_variation_duration=nox_stratospheric_water_vapor_decrease_gwpstar_variation_duration,
            gwpstar_s_coefficient=nox_stratospheric_water_vapor_decrease_gwpstar_s_coefficient,
        )
        self.df_climate["nox_stratospheric_water_vapor_decrease_equivalent_emissions"] = (
            nox_stratospheric_water_vapor_decrease_equivalent_emissions
        )

        ## Soot
        soot_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=soot_erf,
            gwpstar_variation_duration=soot_gwpstar_variation_duration,
            gwpstar_s_coefficient=soot_gwpstar_s_coefficient,
        )
        self.df_climate["soot_equivalent_emissions"] = soot_equivalent_emissions

        ## H2O
        h2o_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=h2o_erf,
            gwpstar_variation_duration=h2o_gwpstar_variation_duration,
            gwpstar_s_coefficient=h2o_gwpstar_s_coefficient,
        )
        self.df_climate["h2o_equivalent_emissions"] = h2o_equivalent_emissions

        ## Sulfur
        sulfur_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=sulfur_erf,
            gwpstar_variation_duration=sulfur_gwpstar_variation_duration,
            gwpstar_s_coefficient=sulfur_gwpstar_s_coefficient,
        )
        self.df_climate["sulfur_equivalent_emissions"] = sulfur_equivalent_emissions

        ## CH4
        ch4_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=ch4_erf,
            gwpstar_variation_duration=ch4_gwpstar_variation_duration,
            gwpstar_s_coefficient=ch4_gwpstar_s_coefficient,
        )
        self.df_climate["ch4_equivalent_emissions"] = ch4_equivalent_emissions

        ## H2
        h2_equivalent_emissions = GWPStarEquivalentEmissionsFunction(
            self,
            emissions_erf=h2_erf,
            gwpstar_variation_duration=h2_gwpstar_variation_duration,
            gwpstar_s_coefficient=h2_gwpstar_s_coefficient,
        )
        self.df_climate["h2_equivalent_emissions"] = h2_equivalent_emissions

        ## Total
        non_co2_equivalent_emissions = (
            contrails_equivalent_emissions
            + nox_short_term_o3_increase_equivalent_emissions
            + nox_long_term_o3_decrease_equivalent_emissions
            + nox_ch4_decrease_equivalent_emissions
            + nox_stratospheric_water_vapor_decrease_equivalent_emissions
            + soot_equivalent_emissions
            + h2o_equivalent_emissions
            + sulfur_equivalent_emissions
            + ch4_equivalent_emissions
            + h2_equivalent_emissions
        )
        total_equivalent_emissions = co2_emissions + non_co2_equivalent_emissions
        self.df_climate["non_co2_equivalent_emissions"] = non_co2_equivalent_emissions
        self.df_climate["total_equivalent_emissions"] = total_equivalent_emissions

        ## Cumulative CO2, non-CO2 and total equivalent emissions (Gtwe)

        ### From 1940
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_co2_emissions"
        ] = co2_emissions.loc[self.climate_historic_start_year] / 1000
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_contrails_equivalent_emissions"
        ] = contrails_equivalent_emissions.loc[self.climate_historic_start_year] / 1000
        self.df_climate.loc[
            self.climate_historic_start_year,
            "historical_cumulative_nox_short_term_o3_increase_equivalent_emissions",
        ] = (
            nox_short_term_o3_increase_equivalent_emissions.loc[self.climate_historic_start_year]
            / 1000
        )
        self.df_climate.loc[
            self.climate_historic_start_year,
            "historical_cumulative_nox_long_term_o3_decrease_equivalent_emissions",
        ] = (
            nox_long_term_o3_decrease_equivalent_emissions.loc[self.climate_historic_start_year]
            / 1000
        )
        self.df_climate.loc[
            self.climate_historic_start_year,
            "historical_cumulative_nox_ch4_decrease_equivalent_emissions",
        ] = nox_ch4_decrease_equivalent_emissions.loc[self.climate_historic_start_year] / 1000
        self.df_climate.loc[
            self.climate_historic_start_year,
            "historical_cumulative_nox_stratospheric_water_vapor_decrease_equivalent_emissions",
        ] = (
            nox_stratospheric_water_vapor_decrease_equivalent_emissions.loc[
                self.climate_historic_start_year
            ]
            / 1000
        )
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_soot_equivalent_emissions"
        ] = soot_equivalent_emissions.loc[self.climate_historic_start_year] / 1000
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_h2o_equivalent_emissions"
        ] = h2o_equivalent_emissions.loc[self.climate_historic_start_year] / 1000
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_sulfur_equivalent_emissions"
        ] = sulfur_equivalent_emissions.loc[self.climate_historic_start_year] / 1000
        
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_ch4_equivalent_emissions"
        ] = ch4_equivalent_emissions.loc[self.climate_historic_start_year] / 1000
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_h2_equivalent_emissions"
        ] = h2_equivalent_emissions.loc[self.climate_historic_start_year] / 1000
        
        self.df_climate.loc[
            self.climate_historic_start_year, "historical_cumulative_non_co2_equivalent_emissions"
        ] = non_co2_equivalent_emissions[self.climate_historic_start_year] / 1000
        for k in range(self.climate_historic_start_year + 1, self.end_year + 1):
            self.df_climate.loc[k, "historical_cumulative_co2_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_co2_emissions"]
                + co2_emissions.loc[k] / 1000
            )
            self.df_climate.loc[k, "historical_cumulative_contrails_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_contrails_equivalent_emissions"]
                + contrails_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[
                k, "historical_cumulative_nox_short_term_o3_increase_equivalent_emissions"
            ] = (
                self.df_climate.loc[
                    k - 1, "historical_cumulative_nox_short_term_o3_increase_equivalent_emissions"
                ]
                + nox_short_term_o3_increase_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[
                k, "historical_cumulative_nox_long_term_o3_decrease_equivalent_emissions"
            ] = (
                self.df_climate.loc[
                    k - 1, "historical_cumulative_nox_long_term_o3_decrease_equivalent_emissions"
                ]
                + nox_long_term_o3_decrease_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[
                k, "historical_cumulative_nox_ch4_decrease_equivalent_emissions"
            ] = (
                self.df_climate.loc[
                    k - 1, "historical_cumulative_nox_ch4_decrease_equivalent_emissions"
                ]
                + nox_ch4_decrease_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[
                k,
                "historical_cumulative_nox_stratospheric_water_vapor_decrease_equivalent_emissions",
            ] = (
                self.df_climate.loc[
                    k - 1,
                    "historical_cumulative_nox_stratospheric_water_vapor_decrease_equivalent_emissions",
                ]
                + nox_stratospheric_water_vapor_decrease_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[k, "historical_cumulative_soot_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_soot_equivalent_emissions"]
                + soot_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[k, "historical_cumulative_h2o_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_h2o_equivalent_emissions"]
                + h2o_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[k, "historical_cumulative_sulfur_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_sulfur_equivalent_emissions"]
                + sulfur_equivalent_emissions.loc[k] / 1000
            )

            self.df_climate.loc[k, "historical_cumulative_ch4_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_ch4_equivalent_emissions"]
                + ch4_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[k, "historical_cumulative_h2_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_h2_equivalent_emissions"]
                + h2_equivalent_emissions.loc[k] / 1000
            )

            self.df_climate.loc[k, "historical_cumulative_non_co2_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "historical_cumulative_non_co2_equivalent_emissions"]
                + non_co2_equivalent_emissions.loc[k] / 1000
            )

        historical_cumulative_co2_emissions = self.df_climate["historical_cumulative_co2_emissions"]
        historical_cumulative_contrails_equivalent_emissions = self.df_climate[
            "historical_cumulative_contrails_equivalent_emissions"
        ]
        historical_cumulative_nox_short_term_o3_increase_equivalent_emissions = self.df_climate[
            "historical_cumulative_nox_short_term_o3_increase_equivalent_emissions"
        ]
        historical_cumulative_nox_long_term_o3_decrease_equivalent_emissions = self.df_climate[
            "historical_cumulative_nox_long_term_o3_decrease_equivalent_emissions"
        ]
        historical_cumulative_nox_ch4_decrease_equivalent_emissions = self.df_climate[
            "historical_cumulative_nox_ch4_decrease_equivalent_emissions"
        ]
        historical_cumulative_nox_stratospheric_water_vapor_decrease_equivalent_emissions = (
            self.df_climate[
                "historical_cumulative_nox_stratospheric_water_vapor_decrease_equivalent_emissions"
            ]
        )
        historical_cumulative_soot_equivalent_emissions = self.df_climate[
            "historical_cumulative_soot_equivalent_emissions"
        ]
        historical_cumulative_h2o_equivalent_emissions = self.df_climate[
            "historical_cumulative_h2o_equivalent_emissions"
        ]
        historical_cumulative_sulfur_equivalent_emissions = self.df_climate[
            "historical_cumulative_sulfur_equivalent_emissions"
        ]
        historical_cumulative_ch4_equivalent_emissions = self.df_climate[
            "historical_cumulative_ch4_equivalent_emissions"
        ]
        historical_cumulative_h2_equivalent_emissions = self.df_climate[
            "historical_cumulative_h2_equivalent_emissions"
        ]
        historical_cumulative_non_co2_equivalent_emissions = self.df_climate[
            "historical_cumulative_non_co2_equivalent_emissions"
        ]
        historical_cumulative_total_equivalent_emissions = (
            historical_cumulative_co2_emissions + historical_cumulative_non_co2_equivalent_emissions
        )
        self.df_climate["cumulative_total_equivalent_emissions"] = (
            historical_cumulative_total_equivalent_emissions
        )

        ### From 2020
        self.df_climate.loc[
            self.prospection_start_year - 1, "cumulative_non_co2_equivalent_emissions"
        ] = 0.0
        self.df_climate.loc[
            self.prospection_start_year - 1, "cumulative_total_equivalent_emissions"
        ] = 0.0
        for k in range(self.prospection_start_year, self.end_year + 1):
            self.df_climate.loc[k, "cumulative_non_co2_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "cumulative_non_co2_equivalent_emissions"]
                + non_co2_equivalent_emissions.loc[k] / 1000
            )
            self.df_climate.loc[k, "cumulative_total_equivalent_emissions"] = (
                self.df_climate.loc[k - 1, "cumulative_total_equivalent_emissions"]
                + total_equivalent_emissions.loc[k] / 1000
            )
        cumulative_non_co2_equivalent_emissions = self.df_climate[
            "cumulative_non_co2_equivalent_emissions"
        ]
        cumulative_total_equivalent_emissions = self.df_climate[
            "cumulative_total_equivalent_emissions"
        ]

        ## Share CO2/non-CO2
        for k in range(self.climate_historic_start_year, self.end_year + 1):
            self.df_climate.loc[k, "total_co2_equivalent_emissions_ratio"] = (
                total_equivalent_emissions.loc[k] / co2_emissions.loc[k]
            )
        total_co2_equivalent_emissions_ratio = self.df_climate[
            "total_co2_equivalent_emissions_ratio"
        ]

        co2_total_erf_ratio = co2_erf / total_erf * 100
        self.df_climate.loc[:, "co2_total_erf_ratio"] = co2_total_erf_ratio

        # TEMPERATURE

        for k in range(self.climate_historic_start_year, self.end_year + 1):
            self.df_climate.loc[k, "temperature_increase_from_co2_from_aviation"] = (
                tcre_coefficient * historical_cumulative_co2_emissions.loc[k]
            )
            self.df_climate.loc[k, "temperature_increase_from_contrails_from_aviation"] = (
                tcre_coefficient * historical_cumulative_contrails_equivalent_emissions.loc[k]
            )
            self.df_climate.loc[
                k, "temperature_increase_from_nox_short_term_o3_increase_from_aviation"
            ] = (
                tcre_coefficient
                * historical_cumulative_nox_short_term_o3_increase_equivalent_emissions.loc[k]
            )
            self.df_climate.loc[
                k, "temperature_increase_from_nox_long_term_o3_decrease_from_aviation"
            ] = (
                tcre_coefficient
                * historical_cumulative_nox_long_term_o3_decrease_equivalent_emissions.loc[k]
            )
            self.df_climate.loc[k, "temperature_increase_from_nox_ch4_decrease_from_aviation"] = (
                tcre_coefficient
                * historical_cumulative_nox_ch4_decrease_equivalent_emissions.loc[k]
            )
            self.df_climate.loc[
                k, "temperature_increase_from_nox_stratospheric_water_vapor_decrease_from_aviation"
            ] = (
                tcre_coefficient
                * historical_cumulative_nox_stratospheric_water_vapor_decrease_equivalent_emissions.loc[
                    k
                ]
            )
            self.df_climate.loc[k, "temperature_increase_from_soot_from_aviation"] = (
                tcre_coefficient * historical_cumulative_soot_equivalent_emissions.loc[k]
            )
            self.df_climate.loc[k, "temperature_increase_from_h2o_from_aviation"] = (
                tcre_coefficient * historical_cumulative_h2o_equivalent_emissions.loc[k]
            )
            self.df_climate.loc[k, "temperature_increase_from_sulfur_from_aviation"] = (
                tcre_coefficient * historical_cumulative_sulfur_equivalent_emissions.loc[k]
            )

            self.df_climate.loc[k, "temperature_increase_from_ch4_from_aviation"] = (
                tcre_coefficient * historical_cumulative_ch4_equivalent_emissions.loc[k]
            )
            self.df_climate.loc[k, "temperature_increase_from_h2_from_aviation"] = (
                tcre_coefficient * historical_cumulative_h2_equivalent_emissions.loc[k]
            )

            self.df_climate.loc[k, "temperature_increase_from_non_co2_from_aviation"] = (
                tcre_coefficient * historical_cumulative_non_co2_equivalent_emissions.loc[k]
            )
        temperature_increase_from_co2_from_aviation = self.df_climate[
            "temperature_increase_from_co2_from_aviation"
        ]
        temperature_increase_from_contrails_from_aviation = self.df_climate[
            "temperature_increase_from_contrails_from_aviation"
        ]
        temperature_increase_from_nox_short_term_o3_increase_from_aviation = self.df_climate[
            "temperature_increase_from_nox_short_term_o3_increase_from_aviation"
        ]
        temperature_increase_from_nox_long_term_o3_decrease_from_aviation = self.df_climate[
            "temperature_increase_from_nox_long_term_o3_decrease_from_aviation"
        ]
        temperature_increase_from_nox_ch4_decrease_from_aviation = self.df_climate[
            "temperature_increase_from_nox_ch4_decrease_from_aviation"
        ]
        temperature_increase_from_nox_stratospheric_water_vapor_decrease_from_aviation = (
            self.df_climate[
                "temperature_increase_from_nox_stratospheric_water_vapor_decrease_from_aviation"
            ]
        )
        temperature_increase_from_h2o_from_aviation = self.df_climate[
            "temperature_increase_from_h2o_from_aviation"
        ]
        temperature_increase_from_soot_from_aviation = self.df_climate[
            "temperature_increase_from_soot_from_aviation"
        ]
        temperature_increase_from_sulfur_from_aviation = self.df_climate[
            "temperature_increase_from_sulfur_from_aviation"
        ]
        temperature_increase_from_non_co2_from_aviation = self.df_climate[
            "temperature_increase_from_non_co2_from_aviation"
        ]
        temperature_increase_from_aviation = (
            temperature_increase_from_co2_from_aviation
            + temperature_increase_from_non_co2_from_aviation
        )
        self.df_climate["temperature_increase_from_aviation"] = temperature_increase_from_aviation

        return (
            contrails_equivalent_emissions,
            nox_short_term_o3_increase_equivalent_emissions,
            nox_long_term_o3_decrease_equivalent_emissions,
            nox_ch4_decrease_equivalent_emissions,
            nox_stratospheric_water_vapor_decrease_equivalent_emissions,
            soot_equivalent_emissions,
            h2o_equivalent_emissions,
            sulfur_equivalent_emissions,
            non_co2_equivalent_emissions,
            cumulative_non_co2_equivalent_emissions,
            total_equivalent_emissions,
            cumulative_total_equivalent_emissions,
            total_co2_equivalent_emissions_ratio,
            co2_total_erf_ratio,
            temperature_increase_from_aviation,
            temperature_increase_from_co2_from_aviation,
            temperature_increase_from_non_co2_from_aviation,
            temperature_increase_from_contrails_from_aviation,
            temperature_increase_from_nox_short_term_o3_increase_from_aviation,
            temperature_increase_from_nox_long_term_o3_decrease_from_aviation,
            temperature_increase_from_nox_ch4_decrease_from_aviation,
            temperature_increase_from_nox_stratospheric_water_vapor_decrease_from_aviation,
            temperature_increase_from_h2o_from_aviation,
            temperature_increase_from_sulfur_from_aviation,
            temperature_increase_from_soot_from_aviation,
        )
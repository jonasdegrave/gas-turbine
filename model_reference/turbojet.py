from . import components as comp
import pandas as pd
from .thermal_process import u_from_mach
import numpy as np

def correct_mass_flow(mass_flow, ta, pa):
    return mass_flow * (288.15/101.325) * (pa/ta)

class TurboJet:
    """
    A class representative of a Turbo Jet Engine.

    Parameters
    ----------
    data: dict
        A dictionary with all the required input parameters for a TurboJet model.
        ta: Ambient Temperature;
        pa: Ambient Pressure;
        t04: Temperature in the combustion chamber exit;
        u_i or mach: speed in m/s or mach number repectively;
        gamma_d: cp/cv in the Diffuser;
        gamma_c: cp/cv in the Compressor;
        gamma_b: cp/cv in the Combustion Chamber;
        gamma_t: cp/cv in the Turbine;
        gamma_n: cp/cv in the Nozzle;
        n_d: efficiency of the Diffuser;
        n_c: efficiency of the Compressor;
        n_b: efficiency of the Combustion Chamber;
        n_t: efficiency of the Turbine;
        n_n: efficiency of the Nozzle;
        prc: Compression rate;
        pc_fuel: Heat of Combustion of the fuel;
        cp_fuel: Specific Heat in the combustion chamber;
        pressure_loss: Pressure loss in combustion chamber, in percentage;
        r: the air Gas Constant.
    """
    def __init__(self, data:dict):
        if "u_in" in data.keys():
            speed = data.get("u_in")
        elif "mach" in data.keys():
            speed = u_from_mach(
                data.get("mach"), data.get("ta"), data.get("gamma_d"), data.get("r")
                )
        else:
            speed = 0

        if "pressure_loss" in data.keys():
            pressure_loss = data.get("pressure_loss")
        else:
            pressure_loss = 0

        if data.get("mass_flow") is None:
            self._has_mass_flow = False
        else:
            self._has_mass_flow = True

        if self._has_mass_flow:
            self._mass_flow_sea_level = data.get('mass_flow')
            self._mass_flow0 = correct_mass_flow(self._mass_flow_sea_level, data.get('pa'), data.get('ta'))
            self._mass_flow = self._mass_flow0
        else:
            self._mass_flow = np.nan

        self._n2 = 1

        self.air_entrance = comp.Diffuser_Adiab(
            data.get('ta'), data.get('pa'),
            data.get('gamma_d'), data.get('r'),
            speed, data.get('n_d')
        )

        self.compressor = comp.Compressor(
            self.air_entrance.t0f, self.air_entrance.p0f,
            data.get('gamma_c'), data.get('r'),
            data.get('n_c'), data.get('prc')
        )

        self.combustion_chamber = comp.CombustionChamber(self.compressor.t0f, self.compressor.p0f,
            data.get('gamma_b'), data.get('r'),
            data.get('t04'), data.get('cp_fuel'), data.get('pc_fuel'), data.get('n_b'),
            pressure_loss=pressure_loss
        )

        self.turbine = comp.Turbine(self.combustion_chamber.t0f, self.combustion_chamber.p0f,
            data.get('gamma_t'), data.get('r'),
            data.get('n_t'), self.compressor
        )

        self.nozzle = comp.Nozzle_Adiab(self.turbine.t0f, self.turbine.p0f,
            data.get('gamma_n'), data.get('r'),
            data.get('pa'), data.get('n_n')
        )
        self.components = [self.air_entrance, self.compressor, self.combustion_chamber, self.turbine, self.nozzle]

    def _set_n2_mass_flow(self, n2):
        coef = [-6.6970E+00, 1.7001E+01, -1.2170E+01, 2.8717E+00]
        self._mass_flow = self._mass_flow0 * np.polyval(coef, n2)

    def set_n2(self, n2):
        self.compressor.set_n2(n2)
        self.combustion_chamber.set_n2(n2)
        self.turbine.set_n2(n2)

        if self._has_mass_flow:
            self._set_n2_mass_flow(n2)

        self._n2 = n2

        self.update_model()
    
    def update_model(self):
        self.compressor.t0i, self.compressor.p0i = self.air_entrance.t0f, self.air_entrance.p0f

        self.combustion_chamber.t0i, self.combustion_chamber.p0i = self.compressor.t0f, self.compressor.p0f

        self.turbine.t0i, self.turbine.p0i = self.combustion_chamber.t0f, self.combustion_chamber.p0f

        self.nozzle.t0i, self.nozzle.p0i = self.turbine.t0f, self.turbine.p0f

    def sumarise(self):
        """
        Summary of engine components parameters.
        
        Return
        ------
        A column DataFrame of all the components inlet and outlet properties for the current N2 rotation.
        """
        data = pd.Series(dtype='float64')

        for i in self.components:
            comp_dict = i.sumarise()
            for k,v in comp_dict.items():
                data.loc[k] = v

        return data.sort_index().to_frame(name=self._n2)

    def sumarise_results(self):
        """
        Summary of engine components parameters.
        
        Return
        ------
        A column DataFrame with the performance results for the motor in the given N2 rotation.
        """
        data = pd.Series(dtype='float64')

        
        list_of_parameters = ['specific_thrust', 'TSFC', 'thrust_total','mass_flow']

        for parameter in list_of_parameters:
            result = getattr(self, parameter)
            data.loc[parameter] = result


        return data.sort_index().to_frame(name=self._n2)

    @property
    def specific_thrust(self):
        exit_speed = self.nozzle.u_s
        if np.isnan(exit_speed):
            return -self.air_entrance._ui
        else:
            return (1+self.combustion_chamber.f)*self.nozzle.u_s - self.air_entrance._ui

    @property
    def thrust_total(self):
        return self.specific_thrust * self.mass_flow

    @property
    def TSFC(self):
        return self.combustion_chamber.f / self.specific_thrust

    @property
    def mass_flow(self):
        return self._mass_flow
from . import components as comp
import pandas as pd
from .thermal_process import u_from_mach
import numpy as np

def correct_mass_flow(mass_flow, ta, pa):
    if mass_flow is not None:
        return mass_flow * (288.15/101.325) * (pa/ta)
    return None

class TurboProp:
    """
    A class representative of a TurboProp Engine.

    Parameters
    ----------
    data: dict
        A dictionary with all the required input parameters for a TurboJet model.
        mass_flow: mass flow at sea level
        ta: Ambient Temperature;
        pa: Ambient Pressure;
        t04: Temperature in the combustion chamber exit;
        u_i or mach: speed in m/s or mach number repectively;
        gamma_d: cp/cv in the Diffuser;
        gamma_f: cp/cv in the Fan;
        gamma_c: cp/cv in the Compressor;
        gamma_b: cp/cv in the Combustion Chamber;
        gamma_t: cp/cv in the Turbine;
        gamma_tl: cp/cv in the Free Turbine;
        gamma_n: cp/cv in the Nozzle;
        n_d: efficiency of the Diffuser;
        n_c: efficiency of the Compressor;
        n_b: efficiency of the Combustion Chamber;
        n_t: efficiency of the Turbine;
        n_tl: efficiency of the Free Turbine;
        n_n: efficiency of the Nozzle;
        prc: Compressor compression rate;
        pr_tl: Free Turbine expansion rate;
        pc_fuel: Heat of Combustion of the fuel;
        cp_tl: Specific Heat in the combustion chamber;
        cp_fuel: Specific Heat in the combustion chamber;
        pressure_loss: Pressure loss in combustion chamber, in percentage;
        r: the air Gas Constant.
        gearbox_power_ratio: power ratio between gearbox and turbine.
        propeller_efficiency: efficiency of the Fan;
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

        self._n2 = 1
        self._mass_flow_sea_level = data.get('mass_flow')
        self._mass_flow0 = correct_mass_flow(self._mass_flow_sea_level, data.get('ta'), data.get('pa'))
        self._aircraft_speed = speed
        self._mass_flow = self._mass_flow0
        self.propeller_efficiency = data.get('propeller_efficiency')
        self.gearbox_power_ratio = data.get('gearbox_power_ratio')


        self.air_entrance = comp.Diffuser_Adiab(
            data.get('ta'), data.get('pa'),
            data.get('gamma_d'), data.get('r'),
            self._aircraft_speed, data.get('n_d')
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

        self.turbine_compressor = comp.Turbine(self.combustion_chamber.t0f, self.combustion_chamber.p0f,
            data.get('gamma_t'), data.get('r'),
            data.get('n_t'), self.compressor
        )

        self.turbine_free = comp.FreeTurbine(self.turbine_compressor.t0f, self.turbine_compressor.p0f,
            data.get('gamma_tl'), data.get('r'),
            data.get('n_tl'), data.get('pr_tl'), data.get('cp_tl')
        )

        self.nozzle = comp.Nozzle_Adiab(self.turbine_free.t0f, self.turbine_free.p0f,
            data.get('gamma_n'), data.get('r'),
            data.get('pa'), data.get('n_n')
        )
        
        self.components = [
            self.air_entrance, 
            self.compressor, 
            self.combustion_chamber, 
            self.turbine_compressor, 
            self.turbine_free, 
            self.nozzle
        ]

    def set_n2(self, n2):
        self.compressor.set_n2(n2)
        self.combustion_chamber.set_n2(n2)
        self.turbine_compressor.set_n2(n2)
        self.turbine_free.set_n2(n2)
        self._set_n2_mass_flow(n2)

        self._n2 = n2

        self.update_model()


    def _set_n2_mass_flow(self, n2):
        if self._mass_flow is not None:
            coef = [-6.6970E+00, 1.7001E+01, -1.2170E+01, 2.8717E+00]
            self._mass_flow = self._mass_flow0 * np.polyval(coef, n2)
    
    def update_model(self):
        self.compressor.t0i, self.compressor.p0i = self.air_entrance.t0f, self.air_entrance.p0f

        self.combustion_chamber.t0i, self.combustion_chamber.p0i = self.compressor.t0f, self.compressor.p0f

        self.turbine_compressor.t0i, self.turbine_compressor.p0i = self.combustion_chamber.t0f, self.combustion_chamber.p0f
        
        self.turbine_free.t0i, self.turbine_free.p0i = self.turbine_compressor.t0f, self.turbine_compressor.p0f

        self.nozzle.t0i, self.nozzle.p0i = self.turbine_free.t0f, self.turbine_free.p0f
    
    def sumarise(self):
        data = pd.Series(dtype='float64')

        for i in self.components:
            comp_dict = i.sumarise()
            for k,v in comp_dict.items():
                data.loc[k] = v

        return data.sort_index().to_frame(name=self._n2)

    def sumarise_results(self):
        data = pd.Series(dtype='float64')

        
        list_of_parameters = [
            'specific_power_turbine', 
            'turbine_power', 
            'gearbox_power', 
            'specific_thrust', 
            'BSFC', 
            'EBSFC', 
            'TSFC', 
            'thrust_hot_air', 
            'thrust_propeller', 
            'fuel_consumption', 
            'thrust_total',
            'mass_flow',
            'aircraft_speed'
            ]

        for parameter in list_of_parameters:
            result = getattr(self, parameter)
            data.loc[parameter] = result

        return data.sort_index().to_frame(name=self._n2)
    
    @property
    def specific_power_turbine(self):
        specific_turbine_power = self.turbine_free.specific_work * (1 + self.combustion_chamber.f)
        return specific_turbine_power

    @property
    def turbine_power(self):
        if self._mass_flow is not None:
            turbine_power = self.specific_power_turbine * self._mass_flow
            return turbine_power
        return None

    @property
    def gearbox_power(self):
        if self._mass_flow is not None:
            gearbox_power = self.turbine_power * self.gearbox_power_ratio
            return gearbox_power
        return None

    @property
    def specific_thrust(self):
        outlet_speed = self.nozzle.u_s
        specific_thrust = (1 + self.combustion_chamber.f)*outlet_speed - self._aircraft_speed
        return specific_thrust/1000

    @property
    def BSFC(self):
        specific_power_turbine = self.specific_power_turbine
        BSFC = self.combustion_chamber.f/specific_power_turbine
        return BSFC

    @property
    def EBSFC(self):
        specific_power_thrust = self.specific_thrust * self._aircraft_speed
        specific_power_turbine = self.specific_power_turbine
        EBSFC = self.combustion_chamber.f/(specific_power_thrust + specific_power_turbine)
        return EBSFC
    
    @property
    def thrust_hot_air(self):
        if self._mass_flow is not None:
            thrust = self.specific_thrust * self._mass_flow
            return thrust
        return None
    
    @property
    def thrust_propeller(self):
        if self._mass_flow is not None:
            if self._aircraft_speed > 0:
                thrust_propeller = self.gearbox_power * self.propeller_efficiency / self._aircraft_speed
                return thrust_propeller
            return 0
        return None

    @property
    def thrust_total(self):
        if self._mass_flow is not None:
            return self.thrust_hot_air + self.thrust_propeller
        return None
        

    @property
    def fuel_consumption(self):
        if self._mass_flow is not None:
            fuel_consumption = self.turbine_power * self.EBSFC
            return fuel_consumption 
        return None

    @property
    def TSFC(self):
        if self._mass_flow is not None:
            TSFC = self.combustion_chamber.f * self._mass_flow/(self.thrust_hot_air+ self.thrust_propeller)
            return TSFC
        return None
    
    @property
    def mass_flow(self):
        return self._mass_flow

    @property
    def aircraft_speed(self):
        return self._aircraft_speed


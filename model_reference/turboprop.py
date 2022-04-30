from . import components as comp
import pandas as pd
import numpy as np

def correct_mass_flow(mass_flow, ta, pa):
    return mass_flow * (288.15/101.325) * (pa/ta)

class TurboProp:
    def __init__(self, data:dict):
        self._n2 = 1
        self._mass_flow0 = data.get('mass_flow')
        self._aircraft_speed = data.get('u_in')
        self._mass_flow = correct_mass_flow(self._mass_flow0, data.get('ta'), data.get('pa'))
        self.propeller_efficiency = data.get('propeller_efficiency')
        self.gearbox_power_ratio = data.get('gearbox_power_ratio')


        self.air_entrance = comp.Diffuser_Adiab(
            data.get('ta'), data.get('pa'),
            data.get('gamma_d'), data.get('r'),
            data.get('u_in'), data.get('n_d')
        )

        self.compressor = comp.Compressor(
            self.air_entrance.t0f, self.air_entrance.p0f,
            data.get('gamma_c'), data.get('r'),
            data.get('n_c'), data.get('prc')
        )

        self.combustion_chamber = comp.CombustionChamber(self.compressor.t0f, self.compressor.p0f,
            data.get('gamma_b'), data.get('r'),
            data.get('t04'), data.get('cp_fuel'), data.get('pc_fuel'), data.get('n_b')
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
        coef = [-6.6970E+00, 1.7001E+01, -1.2170E+01, 2.8717E+00]
        self._mass_flow = self._mass_flow * np.polyval(coef, n2)
    
    def update_model(self):
        self.compressor.t0i, self.compressor.p0i = self.air_entrance.t0f, self.air_entrance.p0f

        self.combustion_chamber.t0i, self.combustion_chamber.p0i = self.compressor.t0f, self.compressor.p0f

        self.turbine_compressor.t0i, self.turbine_compressor.p0i = self.combustion_chamber.t0f, self.combustion_chamber.p0f
        
        self.turbine_free.t0i, self.turbine_free.p0i = self.turbine_compressor.t0f, self.turbine_compressor.p0f

        self.nozzle.t0i, self.nozzle.p0i = self.turbine_free.t0f, self.turbine_free.p0f

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
        gearbox_power = self.turbine_power * self.gearbox_power_ratio
        return gearbox_power

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
        if self._aircraft_speed > 0:
            thrust_propeller = self.gearbox_power * self.propeller_efficiency / self._aircraft_speed
        else:
            thrust_propeller = 0

        return thrust_propeller

    @property
    def thrust_total(self):
        return self.thrust_hot_air + self.thrust_propeller
        

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

    def sumarise(self):
        data = pd.Series(dtype='float64')

        for i in self.components:
            comp_dict = i.sumarise()
            for k,v in comp_dict.items():
                data.loc[k] = v

        return data.sort_index().to_frame(name=self._n2)

    def sumarise_results(self):
        data = pd.Series(dtype='float64')

        
        list_of_parameters = ['specific_power_turbine', 'turbine_power', 'gearbox_power', 
        'specific_thrust', 'BSFC', 'EBSFC', 'TSFC', 'thrust_hot_air', 
        'thrust_propeller', 'fuel_consumption', 'thrust_total']

        for parameter in list_of_parameters:
            result = getattr(self, parameter)
            data.loc[parameter] = result


        return data.sort_index().to_frame(name=self._n2)



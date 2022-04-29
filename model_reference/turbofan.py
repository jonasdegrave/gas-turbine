import components as comp
import pandas as pd

class TurboFan:
    def __init__(self, data:dict):
        self._n2 = 1

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

        self.turbine = comp.Turbine(self.combustion_chamber.t0f, self.combustion_chamber.p0f,
            data.get('gamma_t'), data.get('r'),
            data.get('n_t'), self.compressor
        )

        self.nozzle = comp.Nozzle_Adiab(self.turbine.t0f, self.turbine.p0f,
            data.get('gamma_n'), data.get('r'),
            data.get('pa'), data.get('n_n')
        )
        self.components = [self.air_entrance, self.compressor, self.combustion_chamber, self.turbine, self.nozzle]

    def set_n2(self, n2):
        self.compressor.set_n2(n2)
        self.combustion_chamber.set_n2(n2)
        self.turbine.set_n2(n2)

        self._n2 = n2

        self.update_model()
    
    def update_model(self):
        self.compressor.t0i, self.compressor.p0i = self.air_entrance.t0f, self.air_entrance.p0f

        self.combustion_chamber.t0i, self.combustion_chamber.p0i = self.compressor.t0f, self.compressor.p0f

        self.turbine.t0i, self.turbine.p0i = self.combustion_chamber.t0f, self.combustion_chamber.p0f

        self.nozzle.t0i, self.nozzle.p0i = self.turbine.t0f, self.turbine.p0f

    def sumarise(self):
        data = pd.Series(dtype='float64')

        for i in self.components:
            comp_dict = i.sumarise()
            for k,v in comp_dict.items():
                data.loc[k] = v

        return data.sort_index().to_frame(name=self._n2)



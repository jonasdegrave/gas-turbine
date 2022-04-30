from . import thermal_process as tp
import numpy as np


class Diffuser_Adiab(tp.SpeedInThermalProcess):
    """
    A class to represent an Adiabatic Diffuser.

    Parameters
    ----------
        ti: Initial static temperature.
        pi: Initial static pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        ui: Initial velocity.
        n_d: The diffuser efficiency.
    """
    def __init__(self, ti, pi, gamma, r, ui, n_d):
        super().__init__(ti, pi, gamma, r, ui)
        self.n_d = n_d

    @classmethod
    def mach_constructor(cls, ti, pi, gamma, r, mach, n_d):
        """
        A constructor which uses the mach number instead of the speed.

        Parameters
        ----------
            ti: Initial static temperature.
            pi: Initial static pressure.
            gamma: Process cp/cv.
            r: Gas constant.
            mach: Mach number.
            n_d: The diffuser efficiency.
        """
        ui = tp.u_from_mach(mach, ti, gamma, r)
        return cls(ti, pi, gamma, r, ui, n_d)
        
    def get_t0f(self):
        """
        A method to determine the final total (stagnation) temperature for the diffuser.

        Returns
        ----------
            The diffuser final total (stagnation) temperature.
        """
        right = 1 + (self._gamma - 1) * self.mach_in() ** 2 / 2
        return self.ti * right

    def get_p0f(self):
        """
        A method to determine the final total (stagnation) pressure for the diffuser.

        Returns
        ----------
            The diffuser final total (stagnation) pressure.
        """
        t_ratio = self.get_t0f() / self.ti
        exp = self._gamma / (self._gamma - 1)
        return self.pi * (1 + self.n_d * (t_ratio - 1)) ** exp

    def sumarise(self):
        index = ['ta', 'pa', 't02', 'p02', 'gamma_d', 'n_d', 'u_i']
        values = [self.ti, self.pi, self.t0f, self.p0f, self._gamma, self.n_d, self._ui]
        return dict(zip(index, values))

    @property
    def t0f(self):
        return self.get_t0f()

    @property
    def p0f(self):
        return self.get_p0f()


class Nozzle_Adiab(tp.SpeedOutThermalProcess):
    """
    A class to represent a Adiabatic Nozzle.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature.
        p0i: Initial total (stagnation) pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        pf: Final static pressure.
        n_n: The nozzle efficiency.
    """
    def __init__(self, t0i, p0i, gamma, r, pf, n_n, fan_nozzle=False):
        super().__init__(t0i, p0i, gamma, r, pf)
        self.fan = fan_nozzle
        self.n_n = n_n

    def get_tf(self):
        """
        A method to determine the final static temperature for the nozzle.

        Returns
        ----------
            The nozzle final static temperature.
        """
        ts_ratio = self.get_tfs() / self.t0i
        return self.t0i * (1 - self.n_n * (1 - ts_ratio))

    def sumarise(self):
        
        if self.fan:
            index = ['t08', 'p08', 'tff', 'pa', 'gamma_nf', 'n_nf', 'u_sf']
        else:
            index = ['t06', 'p06', 'tf', 'pa', 'gamma_n', 'n_n', 'u_s']
        values = [self.t0i, self.p0i, self.get_tfs(), self._pf, self._gamma, self.n_n, self.u_s]
        return dict(zip(index, values))

    @property
    def u_s(self):
        gamma_mult = self._gamma / (self._gamma - 1)
        return np.sqrt(2 * self.n_n * gamma_mult * self._r * (self.t0i  - self.get_tfs()))


class Compressor(tp.StaticThermalProcess):
    """
    A class to represent a Compressor.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature.
        p0i: Initial total (stagnation) pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        n_c: The compressor efficiency.
        prc: The pressure ratio of the compressor.
    """
    def __init__(self, t0i, p0i, gamma, r, n_c, prc, turbo_fan=False):
        super().__init__(t0i, p0i, gamma, r)
        self.prc = prc
        self._prc0 = prc
        self.n_c = n_c
        self._n_c0 = n_c
        self.is_turbo_fan = turbo_fan

    def get_p0f(self):
        """
        A method to determine the final total (stagnation) pressure for the compressor.

        Returns
        ----------
            The compressor final total (stagnation) pressure.
        """
        return self.p0i * self.prc

    def get_t0f(self):
        """
        A method to determine the final total (stagnation) temperature for the compressor.

        Returns
        ----------
            The compressor final total (stagnation) temperature.
        """
        delta_p = self.prc ** ((self._gamma - 1) / self._gamma)
        return self.t0i * (1 + 1 / self.n_c * (delta_p - 1))

    def set_n2(self, n2):
        prc_constants = np.array([-6.073, 1.4821E1, -1.0042E1, 2.2915])
        self.prc = np.polyval(prc_constants, n2) * self._prc0

        n_c_constants = [ -1.1234, 2.1097, 1.8617E-2]
        self.n_c = np.polyval(n_c_constants, n2) * self._n_c0


    def sumarise(self):
        if self.is_turbo_fan:
            index = ['t08', 'p08', 't03', 'p03', 'gamma_c', 'n_c', 'prc']
        else:
            index = ['t02', 'p02', 't03', 'p03', 'gamma_c', 'n_c', 'prc']
        values = [self.t0i, self.p0i, self.t0f, self.p0f, self._gamma, self.n_c, self.prc]
        return dict(zip(index, values))


    @property
    def t0f(self):
        return self.get_t0f()

    @property
    def p0f(self):
        return self.get_p0f()


class Fan(Compressor):
    def __init__(self, t0i, p0i, gamma, r, n_c, prc, bypass_ratio):
        super().__init__(t0i, p0i, gamma, r, n_c, prc)
        self._bypass_ratio0 = bypass_ratio
        self.bypass_ratio = self._bypass_ratio0

    def set_n2(self, n2):
        n1_constants = np.array([1.4166, -4.0478E-01])
        n1 = np.polyval(n1_constants, n2)
        self.set_n1(n1)

    def set_n1(self, n1):
        
        bypass_ratio_constants = [-8.3241E-1, 3.8824E-1, 1.4263]
        self.bypass_ratio = np.polyval(bypass_ratio_constants, self._bypass_ratio0) * self._bypass_ratio0

        a_coef_constants = [-0.00179, 0.00687, 0.5]
        a_coef = np.polyval(a_coef_constants, self.bypass_ratio)

        b_coef = -4.3317E-2

        c_coef_constants = [ 0.011, 0.53782]
        c_coef = np.polyval(c_coef_constants, self.bypass_ratio)
        
        prc_constants = np.array([a_coef, b_coef, c_coef])
        self.prc = self._prc0 * np.polyval(prc_constants, n1)

        n_c_constants = [-6.6663, 17.752, - 17.469, 7.7181, - 0.32985]
        self.n_c = self._n_c0 * np.polyval(n_c_constants, n1)

    def sumarise(self):
        index = ['t02', 'p02', 't08', 'p08', 'gamma_f', 'n_f', 'prf']
        values = [self.t0i, self.p0i, self.t0f, self.p0f, self._gamma, self.n_c, self.prc]
        return dict(zip(index, values))


class Turbine(tp.StaticThermalProcess):
    """
    A class to represent a Turbine.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature.
        p0i: Initial total (stagnation) pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        n_t: The turbine efficiency.
        compressor: The compressor in the same axis.
        bypass_ratio: The bypass_ratio for Turbofans.
    """
    def __init__(self, t0i, p0i, gamma, r, n_t, compressor: Compressor, turbo_fan=False):
        super().__init__(t0i, p0i, gamma, r)
        self.n_t = n_t
        self._n_t0 = n_t
        self.comp = compressor
        self.is_turbo_fan = turbo_fan

    def get_t0f(self):
        """
        A method to determine the final total (stagnation) temperature for the turbine.

        Returns
        ----------
            The turbine final total (stagnation) temperature.
        """
        return self.t0i - (self.comp.t0f - self.comp.t0i)

    def get_p0f(self):
        """
        A method to determine the final total (stagnation) pressure for the turbine.

        Returns
        ----------
            The turbine final total (stagnation) pressure.
        """
        t_ratio = self.get_t0f() / self.t0i
        exp = self._gamma / (self._gamma - 1)
        return self.p0i * (1 - 1/self.n_t * (1 - t_ratio)) ** exp

    def set_n2(self, n2):
        n_t_constants = np.array([-6.7490E-2, 2.5640E-1, 8.1153E-1])
        self.n_t = self._n_t0 * np.polyval(n_t_constants, n2)

    def sumarise(self):
        if self.is_turbo_fan:
            index = ['t04', 'p04', 'tet', 'pet', 'gamma_t', 'n_t']
        else:
            index = ['t04', 'p04', 't05', 'p05', 'gamma_t', 'n_t']

        values = [self.t0i, self.p0i, self.t0f, self.p0f, self._gamma, self.n_t]
        return dict(zip(index, values))


    @property
    def t0f(self):
        return self.get_t0f()

    @property
    def p0f(self):
        return self.get_p0f()


class FanTurbine(Turbine):
    """
    A class to represent a Turbine.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature.
        p0i: Initial total (stagnation) pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        n_t: The turbine efficiency.
        compressor: The compressor in the same axis.
        bypass_ratio: The bypass_ratio for Turbofans.
    """
    def __init__(self, t0i, p0i, gamma, r, n_t, compressor: Fan):
        super().__init__(t0i, p0i, gamma, r, n_t, compressor)
        self.n_t = n_t
        self._n_t0 = n_t
        self.comp = compressor

    def get_t0f(self):
        """
        A method to determine the final total (stagnation) temperature for the turbine.

        Returns
        ----------
            The turbine final total (stagnation) temperature.
        """
        return self.t0i - (self.comp.bypass_ratio + 1) * (self.comp.t0f - self.comp.t0i)

    def get_p0f(self):
        """
        A method to determine the final total (stagnation) pressure for the turbine.

        Returns
        ----------
            The turbine final total (stagnation) pressure.
        """
        t_ratio = self.get_t0f() / self.t0i
        exp = self._gamma / (self._gamma - 1)
        return self.p0i * (1 - 1/self.n_t * (1 - t_ratio)) ** exp

    def set_n2(self, n2):
        n1_constants = np.array([1.4166, -4.0478E-01])
        n1 = np.polyval(n1_constants, n2)
        self.set_n1(n1)

    def set_n1(self, n1):
        n_t_constants = np.array([-6.7490E-2, 2.5640E-1, 8.1153E-1])
        self.n_t = self._n_t0 * np.polyval(n_t_constants, n1)

    def sumarise(self):
        index = ['tet', 'pet', 't05', 'p05', 'gamma_tf', 'n_tf']
        values = [self.t0i, self.p0i, self.t0f, self.p0f, self._gamma, self.n_t]
        return dict(zip(index, values))


class FreeTurbine(tp.StaticThermalProcess):
    """
    A class to represent a Free Turbine.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature.
        p0i: Initial total (stagnation) pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        n_t: The turbine efficiency.
        bypass_ratio: The bypass_ratio for Turbofans.
    """
    def __init__(self, t0i, p0i, gamma, r, n_t, prt, cp):
        super().__init__(t0i, p0i, gamma, r)
        self.n_t = n_t
        self._n_t0 = n_t
        self._prt = prt
        self.prt = prt
        self.cp = cp

    def get_t0f(self):
        """
        A method to determine the final total (stagnation) temperature for the turbine.

        Returns
        ----------
            The turbine final total (stagnation) temperature.
        """
        exp = (self._gamma - 1) / self._gamma
        t0f = self.t0i * (1 - self.n_t * (1 - (1/self.prt)**exp))
        return t0f

    def get_p0f(self):
        """
        A method to determine the final total (stagnation) pressure for the turbine.

        Returns
        ----------
            The turbine final total (stagnation) pressure.
        """
        p0f = self.p0i/self.prt

        return p0f
    
    def set_n2(self, n2):
        prt_constants = [-1.8063E+01, 4.2469E+01, -3.1480E+01, 8.0681E+00]
        self.prt = self._prt * np.polyval(prt_constants, n2)
        
        n_t_constants = [1.9062E+01, -5.2456E+01, 4.7887E+01, -1.3489E+01]
        self.n_t = self._n_t0 * np.polyval(n_t_constants, n2)

    def set_n1(self, n1):
        pass

    def sumarise(self):
        index = ['t06', 'p06', 'gamma_tf', 'n_tf']
        values = [self.t0f, self.p0f, self._gamma, self.n_t]
        return dict(zip(index, values))
    
    @property
    def specific_work(self):
        return (self.t0i - self.t0f)*(self.cp)
    
    
    @property
    def t0f(self):
        return self.get_t0f()

    @property
    def p0f(self):
        return self.get_p0f()


class CombustionChamber(tp.StaticThermalProcess):
    """
    A class to represent a Combustion Chamber.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature.
        p0i: Initial total (stagnation) pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        t0f: The final total (stagnation) temperature.
        cp: The Specific Heat.
        pc: The Heat of Combustion.
        n_b: The combustion chamber efficiency.
    """
    def __init__(self, ti, pi, gamma, r, t0f, cp, pc, n_b):
        super().__init__(ti, pi, gamma, r)
        self.t0f = t0f
        self._t0f0 = t0f
        self._cp = cp
        self._pc = pc
        self.n_b = n_b
        self._n_b0 = n_b


    def get_f(self):
        """
        A method to determine fuel/gas mass ratio.

        Returns
        ----------
            The fuel/gas mass ratio.
        """
        t_ratio = self.t0f/self.t0i
        den = self.n_b * self._pc / (self._cp * self.t0i) - t_ratio
        return (t_ratio - 1) / den

    def set_n2(self, n2):
        n_b_constants = np.array([-6.7490E-2, 2.5640E-1, 8.1153E-1])
        self.n_b = self._n_b0 * np.polyval(n_b_constants, n2)

        t0f_constants = np.array([8.1821E-1, -2.2401E-1, 4.1842E-1])
        self.t0f = self._t0f0 * np.polyval(t0f_constants, n2)

    def sumarise(self):
        index = ['t03', 'p03', 't04', 'p04', 'gamma_b', 'n_b', 'pc_comb', 'cp_comb', 'f']
        values = [self.t0i, self.p0i, self.t0f, self.p0f, self._gamma, self.n_b, self._pc, self._cp, self.f]
        return dict(zip(index, values))

    @property
    def f(self):
        return self.get_f()
    

    @property
    def p0f(self):
        return self.p0i

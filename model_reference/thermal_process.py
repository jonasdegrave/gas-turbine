import numpy

def sound_speed(gamma, r, t):
    return numpy.sqrt(gamma * r * t)

def mach(u, a):
    return u / a

def u_from_mach(mach, t, gamma, r):
    a = sound_speed(t, gamma, r)
    return a * mach

def t_total_from_static(t_static, u, cp):
    return t_static + u ** 2 / (2 * cp)

def t_static_from_total(t_total, u, cp):
    return t_total - u ** 2 / (2 * cp)

class ThermalProcess:
    """
    Base class of a Gas Thermal Process. Includes formulas and definitions standard for all processes.

    Parameters
    ----------
        t_in: Inlet Temperature (can be total or static).
        pi_in: Inlet Pressure (can be total or static).
        gamma: Process cp/cv.
        r: Gas constant.
    """

    def __init__(self, t_in, pi_in, gamma, r):
        self._t_in = t_in
        self._p_in = pi_in
        self._gamma = gamma
        self._r = r

    def sound_speed(self):
        """
        Get sound speed in current temperature.

        Returns
        -------
        The sound speed for the inlet conditions.

        """
        return sound_speed(self._gamma, self._r, self._t_in)

    def pfs_from_t(self, tfs):
        """
        Get the final pressure for an isotropic process.

        Returns
        -------
        The final pressure for an isotropic process. The result will be static or total upon the pressure attribute.
        """
        t_ratio = tfs/self._t_in
        exp = self._gamma/(self._gamma - 1)
        return self._p_in * t_ratio ** exp

    def tfs_from_p(self, pf):
        """
        Get the final temperature for an isotropic process.

        Returns
        -------
        The final temperature for an isotropic process. The result will be static or total upon the pressure attribute.
        """
        p_ratio = pf / self._p_in
        exp = (self._gamma - 1) / self._gamma
        return self._t_in * p_ratio ** exp


class SpeedInThermalProcess(ThermalProcess):
    """
    Gas Thermal Process in which the inlet speed is relevant.

    Parameters
    ----------
        ti: Initial static temperature.
        pi: Initial static pressure.
        gamma: Process cp/cv.
        r: Gas constant.
        ui: Initial velocity.
    """
    def __init__(self, ti, pi, gamma, r, ui):
        super().__init__(ti, pi, gamma, r)
        self._ui = ui

    def mach_in(self):
        """
        Get the mach number for initial speed and temperature, speed must be in m/s.

        Returns
        -------
        The mach number for initial speed and temperature.
        """
        return mach(self._ui, self.sound_speed())

    def get_t0i(self, cp):
        """
        Get the total initial temperature for the provided Specific Heat Value.
        Parameters
        ----------
            cp: The Specific Heat of the Gas.

        Returns
        -------
        The total initial temperature.
        """
        return t_total_from_static(self._t_in, self._ui, cp)

    @property
    def ti(self):
        return self._t_in

    @ti.setter
    def ti(self, ti):
        self._t_in = ti

    @property
    def pi(self):
        return self._p_in

    @pi.setter
    def pi(self, pi):
        self._p_in = pi
        


class SpeedOutThermalProcess(ThermalProcess):
    """
    Gas Thermal Process in which the outlet speed is relevant.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature
        p0i: Initial total (stagnation) pressure
        gamma: Process cp/cv
        r: Gas constant
        pf: Final static pressure
    """
    def __init__(self, t0i, p0i, gamma, r, pf):
        super().__init__(t0i, p0i, gamma, r)
        self._pf = pf

    @property
    def t0i(self):
        return self._t_in

    @t0i.setter
    def t0i(self, t0i):
        self._t_in = t0i

    @property
    def p0i(self):
        return self._p_in

    @p0i.setter
    def p0i(self, p0i):
        self._p_in = p0i

    def get_tfs(self):
        """
        Get the static final temperature.

        Returns
        -------
        Get the static final temperature.
        """
        p_ratio = self._pf / self.p0i
        exp = (self._gamma - 1) / self._gamma
        return self._t_in * p_ratio ** exp



class StaticThermalProcess(ThermalProcess):
    """
    Gas Thermal Process in which the speed is negligible.

    Parameters
    ----------
        t0i: Initial total (stagnation) temperature
        p0i: Initial total (stagnation) pressure
        gamma: Process cp/cv
        r: Gas constant
        pf: Final static pressure
    """
    def __init__(self, t0i, p0i, gamma, r):
        super().__init__(t0i, p0i, gamma, r)

    @property
    def t0i(self):
        return self._t_in

    @t0i.setter
    def t0i(self, t0i):
        self._t_in = t0i

    @property
    def p0i(self):
        return self._p_in

    @p0i.setter
    def p0i(self, p0i):
        self._p_in = p0i
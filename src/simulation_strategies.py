import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from typing import Callable, List
from functools import lru_cache

def keep_edges_shell_vol_func(self):
    # Calculate the volumes of spherical shells
    r_inner = np.zeros_like(self.r)
    r_inner[1:] = self.r[:-1]
    volumes = 4/3 * np.pi * (self.r**3 - r_inner**3)
    return volumes

# Abstract methods for making strategies and factories

class SimulationComponent(ABC):
    @abstractmethod
    def __call__(self, sim):
        pass


class StrategyFactory:
    @classmethod
    def create(cls, strategy_name):
        for strategy_cls in cls.strategy_type.__subclasses__():
            if getattr(strategy_cls, 'name', None) == strategy_name:
                return strategy_cls()
        raise ValueError(
            f"Unknown {cls.strategy_type.__name__}: {strategy_name}")


def name_strategy(name):
    def decorator(cls):
        cls.name = name
        return cls
    return decorator


class StepperStrategy(SimulationComponent):
    pass


class StepperFactory(StrategyFactory):
    strategy_type = StepperStrategy


class RTurnaroundStrategy(SimulationComponent):
    pass


class RTurnaroundFactory(StrategyFactory):
    strategy_type = RTurnaroundStrategy

    
class AccelerationStrategy(SimulationComponent):
    pass

class AccelerationFactory(StrategyFactory):
    strategy_type = AccelerationStrategy

class EnclosedMassStrategy(SimulationComponent):
    pass

class EnclosedMassFactory(StrategyFactory):
    strategy_type = EnclosedMassStrategy

class InitialVelocityStrategy(SimulationComponent):
    pass

class InitialVelocityFactory(StrategyFactory):
    strategy_type = InitialVelocityStrategy

class EnergyStrategy(SimulationComponent):
    pass

class EnergyFactory(StrategyFactory):
    strategy_type = EnergyStrategy

class TimeScaleComponent:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func

class TimeScaleStrategy(SimulationComponent):
    def calculate_min_time_scale(self, *args):
        return min(args)

class TimeScaleFactory(StrategyFactory):
    strategy_type = TimeScaleStrategy

    @classmethod
    def create(cls, strategy_name):
        if '_' in strategy_name:
            # This is a composite strategy
            component_names = strategy_name.split('_')
            return CompositeTimeScaleStrategy.create(*component_names)
        else:
            return CompositeTimeScaleStrategy.create(strategy_name)


class SaveComponent:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func

class SaveStrategy(SimulationComponent):
    pass

class SaveFactory(StrategyFactory):
    strategy_type = SaveStrategy

    @classmethod
    def create(cls, strategy_name):
        if '_' in strategy_name:
            # This is a composite strategy
            component_names = strategy_name.split('_')
            return CompositeSaveStrategy.create(*component_names)
        else:
            return CompositeSaveStrategy.create(strategy_name)

class TimeStepStrategy(SimulationComponent):
    pass

class TimeStepFactory(StrategyFactory):
    strategy_type = TimeStepStrategy

class ShellThicknessStrategy(SimulationComponent):
    pass

class ShellThicknessFactory(StrategyFactory):
    strategy_type = ShellThicknessStrategy

class DensityStrategy(SimulationComponent):
    pass

class DensityFactory(StrategyFactory):
    strategy_type = DensityStrategy

class AngularMomentumStrategy(SimulationComponent):
    pass

class AngularMomentumFactory(StrategyFactory):
    strategy_type = AngularMomentumStrategy

class SofteningStrategy(SimulationComponent):
    pass

class SofteningFactory(StrategyFactory):
    strategy_type = SofteningStrategy

class RMinStrategy(SimulationComponent):
    pass

class RMinFactory(StrategyFactory):
    strategy_type = RMinStrategy

# Implementations of strategies
@name_strategy("leapfrog_kdk")
class LeapfrogStepper(StepperStrategy):
    @staticmethod
    @njit
    def _leapfrog_numba(r, v, a, dt):
        v_half = v + 0.5 * a * dt
        r_new = r + v_half * dt
        return r_new, v_half

    @staticmethod
    @njit
    def _leapfrog_update_v_numba(v_half, a, dt):
        return v_half + 0.5 * a * dt

    def __call__(self, sim):
        sim.r, v_half = self._leapfrog_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        sim.a = sim.a_func()
        sim.v = self._leapfrog_update_v_numba(v_half, sim.a, sim.dt)
        sim.t += sim.dt

@name_strategy("velocity_verlet_alt_v_reflection")
class VelocityVerletAltVReflectionStepper(StepperStrategy):
    def __call__(self, sim):
        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        sim.v = self._velocity_verlet_update_v_numba(
            sim.v, a_old, sim.a, sim.dt, sim.which_reflected)
        sim.t += sim.dt

    @staticmethod
    @njit
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit
    def _velocity_verlet_update_v_numba(v, a_old, a_new, dt, which_reflected):
        return np.where(which_reflected, v, v + 0.5 * (a_old + a_new) * dt)

@name_strategy("velocity_verlet_a_old")
class VelocityVerletAOldStepper(StepperStrategy):
    def __call__(self, sim):
        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        sim.v = self._velocity_verlet_update_v_numba(
            sim.v, a_old, sim.a, sim.dt)
        sim.t += sim.dt

    @staticmethod
    @njit
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit
    def _velocity_verlet_update_v_numba(v, a_old, a_new, dt):
        return v + a_old * dt
    
@name_strategy("velocity_verlet_a_new")
class VelocityVerletANewStepper(StepperStrategy):
    def __call__(self, sim):
        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        sim.v = self._velocity_verlet_update_v_numba(
            sim.v, a_old, sim.a, sim.dt)
        sim.t += sim.dt

    @staticmethod
    @njit
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit
    def _velocity_verlet_update_v_numba(v, a_old, a_new, dt):
        return v + a_new * dt

@name_strategy("velocity_verlet_discontinuity")
class VelocityVerletDiscontinuityStepper(StepperStrategy):
    def __call__(self, sim):
        sim.r = self._velocity_verlet_discontinuity_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.r_min_func()
        sim.m_enc_old = sim.m_enc.copy()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        sim.v, sim.r = self._velocity_verlet_discontinuity_update_v_numba(
            sim.v, a_old, sim.a, sim.dt, sim.m_enc_old, sim.m_enc, sim.G, sim.r)
        sim.t += sim.dt

    @staticmethod
    @njit
    def _velocity_verlet_discontinuity_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit
    def _velocity_verlet_discontinuity_update_v_numba(v, a_old, a_new, dt, m_enc_old, m_enc_new, G, r):
        v_typical = 0.5 * (a_old + a_new) * dt
        v_delta = -G * (m_enc_new - m_enc_old) / r
        v_new = v + np.where(m_enc_old != m_enc_new, v_delta, v_typical)
        r_new = r + np.where(m_enc_old != m_enc_new, v_delta*dt, 0)
        return v_new, r_new

@name_strategy("velocity_verlet")
class VelocityVerletStepper(StepperStrategy):
    def __call__(self, sim):
        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        sim.v = self._velocity_verlet_update_v_numba(
            sim.v, a_old, sim.a, sim.dt)
        sim.t += sim.dt

    @staticmethod
    @njit
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit
    def _velocity_verlet_update_v_numba(v, a_old, a_new, dt):
        return v + 0.5 * (a_old + a_new) * dt


@name_strategy("beeman")
class BeemanStepper(StepperStrategy):
    def __call__(self, sim):
        if sim.prev_a is None:
            # Use Taylor expansion for the first step
            sim.r = sim.r + sim.v * sim.dt + 0.5 * sim.a * sim.dt**2
            sim.r_min_func()
            sim.m_enc = sim.m_enc_func()
            a_new = sim.a_func()
            v_new = sim.v + sim.a * sim.dt
        else:
            sim.r = self._beeman_r_numba(
                sim.r, sim.v, sim.a, sim.prev_a, sim.dt)
            sim.r_min_func()
            sim.m_enc = sim.m_enc_func()
            a_new = sim.a_func()
            v_new = self._beeman_v_numba(
                sim.v, sim.a, a_new, sim.prev_a, sim.dt)

        # Update for next step
        sim.prev_a = sim.a.copy()
        sim.prev_v = sim.v.copy()
        sim.prev_m_enc = sim.m_enc.copy()
        sim.a = a_new
        sim.v = v_new
        sim.t += sim.dt

    @staticmethod
    @njit
    def _beeman_r_numba(r, v, a, prev_a, dt):
        return r + v * dt + (4 * a - prev_a) * (dt**2) / 6

    @staticmethod
    @njit
    def _beeman_v_numba(v, a, a_new, prev_a, dt):
        return v + (2 * a_new + 5 * a - prev_a) * dt / 6



# TODO: this is a bit of a hack, but it works for now
@name_strategy("r_is_r_ta")
class RIsRTurnaroundStrategy(RTurnaroundStrategy):
    def __call__(self, sim):
        return sim.r


@name_strategy("soft_grav_delta_cross")
class SoftGravDeltaCrossAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _soft_grav_delta_cross_a_func_numba(G, m_enc, j, r, r_soft, prev_m_enc):
        return -G * m_enc / r_soft**2 + j**2 / r**3 + G*np.abs(m_enc-prev_m_enc)/r_soft + G*np.abs(m_enc-prev_m_enc)/(2*r_soft**2)

    def __call__(self, sim):
        r_soft = sim.soft_func()
        prev_m_enc = sim.prev_m_enc if sim.prev_m_enc is not None else sim.m_enc
        return self._soft_grav_delta_cross_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, r_soft, prev_m_enc)

@name_strategy("soft_grav")
class SoftGravAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _soft_grav_a_func_numba(G, m_enc, j, r, r_soft):
        return -G * m_enc / r_soft**2 + j**2 / r**3

    def __call__(self, sim):
        r_soft = sim.soft_func()
        return self._soft_grav_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, r_soft)


@name_strategy("soft_all")
class SoftAllAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _soft_all_a_func_numba(G, m_enc, j, r_soft):
        return -G * m_enc / r_soft**2 + j**2 / r_soft**3

    def __call__(self, sim):
        r_soft = sim.soft_func()
        return self._soft_all_a_func_numba(sim.G, sim.m_enc, sim.j, r_soft)
    
@name_strategy("soft_le_delliou")
class SoftLeDelliouAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _soft_le_delliou_a_func_numba(G, m_enc, j, r, r_soft):
        return -G * m_enc * r / r_soft**3 + j**2 * r / r_soft**4

    def __call__(self, sim):
        r_soft = sim.soft_func()
        return self._soft_le_delliou_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, r_soft)


@name_strategy("const_soft")
class ConstSoftStrategy(SofteningStrategy):
    @staticmethod
    @njit
    def _const_soft_func_numba(r, softlen):
        return np.sqrt(r**2 + softlen**2)

    def __call__(self, sim):
        return self._const_soft_func_numba(sim.r, sim.softlen)


@name_strategy("r_ta_soft")
class RTASoftStrategy(SofteningStrategy):
    @staticmethod
    @njit
    def _r_ta_soft_func_numba(r, softlen, r_ta):
        return np.sqrt(r**2 + (softlen * r_ta)**2)

    def __call__(self, sim):
        return self._r_ta_soft_func_numba(sim.r, sim.softlen, sim.r_ta)
    
@name_strategy("const_inclusive")
class ConstInclusiveEnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_const_inclusive_numba(r, m, point_mass):
        m_enc = np.cumsum(m) + point_mass
        return m_enc

    def __call__(self, sim):
        if sim.m_enc is not None:
            return sim.m_enc
        return self._m_enc_const_inclusive_numba(sim.r, sim.m, sim.point_mass)
    
@name_strategy("const_exclusive")
class ConstExclusiveEnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_const_exclusive_numba(r, m, point_mass):
        m_enc = np.cumsum(m) + point_mass - m
        return m_enc

    def __call__(self, sim):
        if sim.m_enc is not None:
            return sim.m_enc
        return self._m_enc_const_exclusive_numba(sim.r, sim.m, sim.point_mass)

@name_strategy("inclusive")
class InclusiveEnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_inclusive_numba(r, m, point_mass):
        sorted_indices = np.argsort(r)
        sorted_masses = m[sorted_indices]
        cumulative_mass = np.cumsum(sorted_masses)
        m_enc = np.empty_like(cumulative_mass)
        m_enc[sorted_indices] = cumulative_mass + point_mass
        return m_enc

    def __call__(self, sim):
        return self._m_enc_inclusive_numba(sim.r, sim.m, sim.point_mass)
    
@name_strategy("inclusive_const_at_small_r")
class InclusiveConstAtSmallREnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_inclusive_const_at_small_r_numba(r, m, point_mass):
        sorted_indices = np.argsort(r)
        sorted_masses = m[sorted_indices]
        cumulative_mass = np.cumsum(sorted_masses)
        m_enc = np.empty_like(cumulative_mass)
        m_enc[sorted_indices] = cumulative_mass + point_mass
        return m_enc

    def __call__(self, sim):
        r_small = 0.1
        return np.where(sim.r < r_small, sim.m_enc, self._m_enc_inclusive_const_at_small_r_numba(sim.r, sim.m, sim.point_mass))

@name_strategy("overlap_inclusive")
class OverlapInclusiveEnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_overlap_inclusive_numba(r, m, thicknesses, point_mass):
        n = len(r)
        m_enc = np.zeros_like(m)
        inner_radii = r - thicknesses
        outer_radii = r
        volumes = outer_radii**3 - inner_radii**3
        #--------------------r[j]--------------------
        #
        #
        #
        #--------------------r[i]--------------------
        #
        #
        #--------------------r[j]-thicknesses[i]-----
        #

        for i in range(n):
            m_enc[i] = m[i] + point_mass
            for j in range(n):
                if i == j:
                    continue
                if r[i] > r[j]:
                    m_enc[i] += m[j]
                elif r[i] > r[j] - thicknesses[j]:
                    # r[i] encloses some of the shell r[j]
                    overlap_volume = r[i]**3 - (r[j] - thicknesses[j])**3
                    volume_fraction = overlap_volume / volumes[j]
                    assert volume_fraction >= 0 and volume_fraction <= 1
                    m_enc[i] += m[j] * volume_fraction
        return m_enc

    def __call__(self, sim):
        sim.thickness_func()
        return self._m_enc_overlap_inclusive_numba(sim.r, sim.m, sim.thicknesses, sim.point_mass)
    
@name_strategy("overlap_avg")
class OverlapAvgEnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_overlap_avg_numba(r, m, thicknesses, point_mass):
        n = len(r)
        m_enc = np.zeros_like(m)
        inner_radii = r - thicknesses
        outer_radii = r
        volumes = outer_radii**3 - inner_radii**3

        for i in range(n):
            m_enc[i] = m[i] + point_mass
            for j in range(n):
                if i == j:
                    continue
                if r[i] > r[j]:
                    m_enc[i] += m[j]
                elif r[j] - thicknesses[j] < r[i]:
                    overlap_volume = min(
                        r[i]**3 - (r[j] - thicknesses[j])**3, volumes[j])
                    volume_fraction = overlap_volume / volumes[j]
                    m_enc[i] += 0.5 * m[j] * volume_fraction
        return m_enc

    def __call__(self, sim):
        sim.thickness_func()
        return self._m_enc_overlap_avg_numba(sim.r, sim.m, sim.thicknesses, sim.point_mass)

@name_strategy("kin_grav_rot")
class KinGravRotEnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _default_energy_func_numba(G, m, v, m_enc, r, j):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_tot = e_k + e_g + e_r
        return e_k, e_g, e_r, e_tot

    def __call__(self, sim):
        sim.e_k, sim.e_g, sim.e_r, sim.e_tot = self._default_energy_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j)

@name_strategy("energy_le_delliou")
class EnergyLeDelliouStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _energy_le_delliou_func_numba(G, m, v, m_enc, r, j, r_soft):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc * r**2 / r_soft**3
        e_r = 0.5 * m * j**2 / r_soft**2
        e_tot = e_k + e_g + e_r
        return e_k, e_g, e_r, e_tot

    def __call__(self, sim):
        r_soft = sim.soft_func()
        sim.e_k, sim.e_g, sim.e_r, sim.e_tot = self._energy_le_delliou_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j, r_soft)

@name_strategy("kin_softgrav_rot")
class KinSoftGravRotEnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _soft_energy_func_numba(G, m, v, m_enc, r, j, r_soft):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r_soft
        e_r = 0.5 * m * j**2 / r**2
        e_tot = e_k + e_g + e_r
        return e_k, e_g, e_r, e_tot

    def __call__(self, sim):
        r_soft = sim.soft_func()
        sim.e_k, sim.e_g, sim.e_r, sim.e_tot = self._soft_energy_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j, r_soft)
        
@name_strategy("kin_softgrav_softrot")
class KinSoftGravRotEnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _soft_energy_func_numba(G, m, v, m_enc, r, j, r_soft):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r_soft
        e_r = 0.5 * m * j**2 / r_soft**2
        e_tot = e_k + e_g + e_r
        return e_k, e_g, e_r, e_tot

    def __call__(self, sim):
        r_soft = sim.soft_func()
        sim.e_k, sim.e_g, sim.e_r, sim.e_tot = self._soft_energy_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j, r_soft)


@njit
def calculate_t_dyn(G, m_enc, r):
    return np.min(1/np.sqrt(G * m_enc / r**3))

@njit
def calculate_t_dynnext(G, m_enc, r, v, dt):
    r_next = r + v * dt
    return np.min(1/np.sqrt(G * m_enc / r_next**3))

@njit
def calculate_t_dynr(G, m_enc, r):
    return np.min(1/np.sqrt(G * m_enc / r**3) * r)

@njit
def calculate_t_vel(r, v, r_max, eps=1e-2):
    return np.min(r_max / (np.abs(v)+eps))

@njit
def calculate_t_acc(r, a, r_max, eps=1e-2):
    return np.min(np.sqrt(r_max / (np.abs(a)+eps)))

@njit
def calculate_t_zero(r, v):
    t_zero = np.inf
    for i in range(len(r)):
        if v[i] < 0:
            t = r[i] / abs(v[i])
            if t < t_zero:
                t_zero = t
    return t_zero

@njit
def calculate_t_rmin(r, v, r_min):
    t_rmin = np.inf
    for i in range(len(r)):
        if v[i] < 0:
            t = (r[i] - r_min) / abs(v[i])
            if t < t_rmin:
                t_rmin = t
    return t_rmin

@njit
def calculate_t_rmina(r, v, a, r_min):
    t_rmina = np.inf
    for i in range(len(r)):
        if v[i] < 0 and a[i] < 0:
            t = np.sqrt((r[i] - r_min) / np.abs(a[i]))
            if t < t_rmina:
                t_rmina = t
    return t_rmina

@njit
def calculate_t_cross(r, v):
    n = len(r)
    t_cross = np.inf
    for i in range(n):
        for j in range(i+1, n):
            dr = np.abs(r[i] - r[j])
            dv = np.abs(v[i] - v[j])
            if dv > 1e-9:
                t = dr / dv
                if t > 0 and t < t_cross:
                    t_cross = t
    return t_cross

@njit
def calculate_t_thickness(r, v, thicknesses):
    n = len(r)
    t_thickness = np.inf
    for i in range(n):
        for j in range(i+1, n):
            v_rel = v[i] - v[j]
            r_rel = r[i] - r[j]
            if np.abs(v_rel) > 1e-9 and ((r_rel / v_rel < 0 or (r_rel + thicknesses[i] + thicknesses[j]) / v_rel < 0) or (r_rel - thicknesses[i] - thicknesses[j])):
                dt = (np.abs(r_rel) + min(thicknesses[i], thicknesses[j])) / np.abs(v_rel)
                if dt < t_thickness:
                    t_thickness = dt
    return t_thickness


                

class CompositeTimeScaleStrategy(TimeScaleStrategy):
    def __init__(self, components: List[TimeScaleComponent]):
        self.components = components

    def __call__(self, sim):
        time_scales = {}
        for component in self.components:
            time_scales[component.name] = component.func(sim)
        
        for name, value in time_scales.items():
            setattr(sim, f"t_{name}", value)
        
        sim.min_time_scale = min(time_scales.values())

    @classmethod
    @lru_cache(maxsize=None)
    def create(cls, *component_names):
        component_map = {
            "dyn": lambda sim: calculate_t_dyn(sim.G, sim.m_enc, sim.r),
            "zero": lambda sim: calculate_t_zero(sim.r, sim.v),
            "rmin": lambda sim: calculate_t_rmin(sim.r, sim.v, sim.r_min),
            "rmina": lambda sim: calculate_t_rmina(sim.r, sim.v, sim.a, sim.r_min),
            "vel": lambda sim: calculate_t_vel(sim.r, sim.v, sim.r_max),
            "acc": lambda sim: calculate_t_acc(sim.r, sim.a, sim.r_max),
            "cross": lambda sim: calculate_t_cross(sim.r, sim.v),
            "dynnext": lambda sim: calculate_t_dynnext(sim.G, sim.m_enc, sim.r, sim.v, sim.dt),
            "dynr": lambda sim: calculate_t_dynr(sim.G, sim.m_enc, sim.r),
            "thickness": lambda sim: calculate_t_thickness(sim.r, sim.v, sim.thicknesses),
        }
        
        components = [
            TimeScaleComponent(name, component_map[name])
            for name in component_names if name in component_map
        ]
        
        if not components:
            raise ValueError("No valid time scale components specified")
        
        return cls(components)
    
def save_default():
    return False

@njit
def save_on_direction_change(v, prev_v):
    if prev_v is None:
        return False
    return np.any(v * prev_v < 0)

#@njit
def save_more_on_direction_change(v, prev_vs):
    return np.any(np.any(v * prev_v < 0) for prev_v in prev_vs if prev_v is not None)

class CompositeSaveStrategy(SaveStrategy):
    def __init__(self, components: List[SaveComponent]):
        self.components = components

    def __call__(self, sim):
        # Return True if any of the save conditions are met
        return any(component.func(sim) for component in self.components)

    @classmethod
    @lru_cache(maxsize=None)
    def create(cls, *component_names):
        component_map = {
            "default": lambda sim: save_default(),
            "vflip": lambda sim: save_on_direction_change(sim.v, sim.prev_v),
            "vflipmore": lambda sim: save_more_on_direction_change(sim.v, [sim.deque[i]['v'] for i in range(len(sim.deque))]),
            "all": lambda sim: True,
        }
        
        components = [
            SaveComponent(name, component_map[name])
            for name in component_names if name in component_map
        ]
        
        if not components:
            raise ValueError("No valid save components specified")
        
        return cls(components)
@name_strategy("const")
class ConstTimeStepStrategy(TimeStepStrategy):
    def __call__(self, sim):
        pass  # Constant timestep, so we don't need to do anything

@name_strategy("simple_adaptive")
class SimpleAdaptiveTimeStepStrategy(TimeStepStrategy):
    @staticmethod
    @njit
    def _simple_adaptive_timestep_numba(safety_factor, min_time_scale):
        return safety_factor * min_time_scale

    def __call__(self, sim):
        sim.dt = max(sim.dt_min, self._simple_adaptive_timestep_numba(
            sim.safety_factor, sim.min_time_scale))

@name_strategy("equal_mass")
class EqualMassDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _equal_mass_rho_func_numba(r, m_pert, N):
        dr = r[1] - r[0]
        return m_pert / N * 1/ (4/3 * np.pi * (r-dr)**3)

    def __call__(self, sim):
        return self._equal_mass_rho_func_numba(sim.r, sim.m_pert, sim.N)
    
@name_strategy("background_plus_tophat")
class BackgroundPlusTophatDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_tophat_rho_func_numba(r, rho_bar, m_pert, tophat_radius):
        delta = m_pert / (4/3 * np.pi * tophat_radius**3)
        print(rho_bar*6*np.pi, delta*6*np.pi)
        retval = np.where(r <= tophat_radius, rho_bar + delta, rho_bar)
        print('r', r)
        print('tophat_radius', tophat_radius)
        print('retval', retval*6*np.pi)
        return retval

    def __call__(self, sim):
        return self._background_plus_tophat_rho_func_numba(r=sim.r, rho_bar=sim.rho_bar, m_pert=sim.m_pert, tophat_radius=sim.tophat_radius)

@name_strategy("const")
class ConstDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _const_rho_func_numba(r_max, m_pert):
        return m_pert / (4/3 * np.pi * r_max**3)

    def __call__(self, sim):
        return self._const_rho_func_numba(sim.r_max, sim.m_pert)

@name_strategy("power_law")
class PowerLawDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _power_law_rho_func_numba(r, r_max, m_pert, gamma):
        norm_const = (3 + gamma) * m_pert / (4 * np.pi * r_max**(3 + gamma))
        return norm_const * r**gamma

    def __call__(self, sim):
        return self._power_law_rho_func_numba(sim.r, sim.r_max, sim.m_pert, sim.gamma)

@name_strategy("background_plus_power_law")
class BackgroundPlusPowerLawDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_power_law_rho_func_numba(r, rho_bar, m_pert, gamma, r_max):
        return rho_bar + (3 + gamma) * m_pert / (4 * np.pi * r_max**(3 + gamma)) * r**gamma

    def __call__(self, sim):
        return self._background_plus_power_law_rho_func_numba(sim.r, sim.rho_bar, sim.m_pert, sim.gamma, sim.r_max)


@name_strategy("const")
class ConstShellThicknessStrategy(ShellThicknessStrategy):
    @staticmethod
    @njit
    def _const_shell_thickness_numba(r, thickness_coef):
        return np.full(len(r), thickness_coef)

    def __call__(self, sim):
        sim.thicknesses = self._const_shell_thickness_numba(sim.r, sim.thickness_coef)

@name_strategy("const")
class ConstAngularMomentumStrategy(AngularMomentumStrategy):
    def __call__(self, sim):
        return sim.j_coef

@name_strategy("gmr")
class GMRAngularMomentumStrategy(AngularMomentumStrategy):
    def __call__(self, sim):
        return sim.j_coef * np.sqrt(sim.G * sim.m_enc * sim.r_ta)
    
@name_strategy("hubble")
class HubbleInitialVelocityStrategy(InitialVelocityStrategy):
    def __call__(self, sim):
        return sim.H * sim.r
    
@name_strategy("reflect")
class ReflectRMinStrategy(RMinStrategy):
    def __call__(self, sim):
        sim.r, sim.v, sim.which_reflected = self._r_min_func_numba(sim.r, sim.v, sim.r_min)

    @staticmethod
    @njit
    def _r_min_func_numba(r, v, r_min):
        which_reflected = np.zeros_like(r, dtype=np.bool_)
        for i in range(len(r)):
            if r[i] < r_min:
                r[i] = 2 * r_min - r[i]
                v[i] = -v[i]
                which_reflected[i] = True
        return r, v, which_reflected
    
@name_strategy("absorb")
class AbsorbRMinStrategy(RMinStrategy):
    def __call__(self, sim):
        sim.r, sim.v, sim.a, sim.absorbed = self._handle_absorbations_numba(sim.r, sim.v, sim.a, sim.r_min, sim.absorbed if sim.absorbed is not None else np.zeros_like(sim.r, dtype=np.bool_))

    @staticmethod
    @njit
    def _handle_absorbations_numba(r, v, a, r_min, absorbed):
        absorbed = np.logical_or(absorbed, r <= r_min)
        r = np.where(absorbed, r_min, r)
        v = np.where(absorbed, 0, v)
        a = np.where(absorbed, 0, a)
        return r, v, a, absorbed


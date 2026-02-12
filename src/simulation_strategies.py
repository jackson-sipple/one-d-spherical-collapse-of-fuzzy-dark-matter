import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from typing import Callable, List
from functools import lru_cache
from scipy import integrate, optimize, interpolate
from src.utils import derivative_nonuniform
from scipy.interpolate import CubicSpline, splrep, splev, UnivariateSpline
import scipy.special as sps
import math


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

class TTurnaroundStrategy(SimulationComponent):
    pass

class TTurnaroundFactory(StrategyFactory):
    strategy_type = TTurnaroundStrategy

class PointMassStrategy(SimulationComponent):
    pass

class PointMassFactory(StrategyFactory):
    strategy_type = PointMassStrategy
    
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
    @abstractmethod
    def __call__(self, sim):
        """Sets the shell masses based on current radii"""
        pass

    @abstractmethod
    def density_at_r(self, r):
        """Returns density at given radius/radii"""
        pass

class DensityFactory(StrategyFactory):
    strategy_type = DensityStrategy

class CurrentDensityStrategy(SimulationComponent):
    pass

class CurrentDensityFactory(StrategyFactory):
    strategy_type = CurrentDensityStrategy

class InitialMassStrategy(SimulationComponent):
    pass

class InitialMassFactory(StrategyFactory):
    strategy_type = InitialMassStrategy

class AngularMomentumStrategy(SimulationComponent):
    pass

class AngularMomentumFactory(StrategyFactory):
    strategy_type = AngularMomentumStrategy

class SofteningStrategy(SimulationComponent):
    pass

class SofteningFactory(StrategyFactory):
    strategy_type = SofteningStrategy

class ProblematicShellStrategy(SimulationComponent):
    pass

class ProblematicShellFactory(StrategyFactory):
    strategy_type = ProblematicShellStrategy

class RMinStrategy(SimulationComponent):
    pass

class RMinFactory(StrategyFactory):
    strategy_type = RMinStrategy

class InitialRadiusStrategy(SimulationComponent):
    pass

class InitialRadiusFactory(StrategyFactory):
    strategy_type = InitialRadiusStrategy

class TerminateEarlyStrategy(SimulationComponent):
    pass

class ShellVolumeStrategy(SimulationComponent):
    @abstractmethod
    def __call__(self, sim):
        pass

class ShellVolumeFactory(StrategyFactory):
    strategy_type = ShellVolumeStrategy

class TerminateEarlyFactory(StrategyFactory):
    strategy_type = TerminateEarlyStrategy

class PressureStrategy(SimulationComponent):
    pass

class PressureFactory(StrategyFactory):
    strategy_type = PressureStrategy

class ArtificialViscosityStrategy(SimulationComponent):
    pass

class ArtificialViscosityFactory(StrategyFactory):
    strategy_type = ArtificialViscosityStrategy

class DensityDerivativeStrategy(SimulationComponent):
    pass

class DensityDerivativeFactory(StrategyFactory):
    strategy_type = DensityDerivativeStrategy

@name_strategy("from_edge_rho")
class FromEdgeRhoStrategy(InitialMassStrategy):
    def __call__(self, sim):
        return sim.shell_vol_func() * sim.rho_func()
    
@name_strategy("integrated_mass")
class IntegratedMassStrategy(InitialMassStrategy):
    def __call__(self, sim):
        n_points = sim.N * 10_000
        eps = 1e-12 # avoid potential singularity at r=0
        r_grid = np.logspace(np.log10(sim.r_min if sim.r_min > 0 else eps), np.log10(sim.r_max), n_points)
        
        # Calculate density on the grid
        rho_grid = sim.rho_func.density_at_r(sim, r_grid)
        
        M_enc_grid = 4*np.pi * integrate.cumulative_trapezoid(
            rho_grid * r_grid**2, 
            r_grid,
            initial=0
        )
        # evaluate at shell boundaries
        m_cumu = np.interp([sim.r_min, *sim.r], r_grid, M_enc_grid)
        mvals = np.diff(m_cumu)
        return mvals


@name_strategy("manual")
class ManualPointMassStrategy(PointMassStrategy):
    def __call__(self, sim):
        return sim.point_mass
    
@name_strategy("rmin_rho_r")
class RMinRhoRStrategy(PointMassStrategy):
    def __call__(self, sim):
        n_points = 10_000
        eps = 1e-12 # avoid potential singularity at r=0
        if sim.r_min < eps:
            return 0
        r_grid = np.logspace(np.log10(eps), np.log10(sim.r_min), n_points)
        
        # Calculate density on the grid
        rho_grid = sim.rho_func.density_at_r(sim, r_grid)
        
        M_enc_grid = 4*np.pi * integrate.cumulative_trapezoid(
            rho_grid * r_grid**2, 
            r_grid,
            initial=0
        )

        return M_enc_grid[-1]

@name_strategy("finite_diff")
class FiniteDiffDensityDerivativeStrategy(DensityDerivativeStrategy):
    @staticmethod
    @njit
    def _density_derivative_numba(rho, r):
        # Get sorting indices and sort r and rho
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        rho_sorted = rho[sort_idx]
        
        # Calculate derivative on sorted arrays
        drho_dr = np.zeros_like(rho_sorted)
        drho_dr[1:-1] = (rho_sorted[2:] - rho_sorted[:-2]) / (r_sorted[2:] - r_sorted[:-2])
        drho_dr[0] = (rho_sorted[1] - rho_sorted[0]) / (r_sorted[1] - r_sorted[0])
        drho_dr[-1] = (rho_sorted[-1] - rho_sorted[-2]) / (r_sorted[-1] - r_sorted[-2])
        
        # Get indices that will restore original order and apply them
        unsort_idx = np.argsort(sort_idx)
        return drho_dr[unsort_idx]

    def __call__(self, sim):
        return self._density_derivative_numba(sim.rho_r, sim.r)
    
@name_strategy("vacuum_bc")
class VacuumBCStrategy(DensityDerivativeStrategy):
    @staticmethod
    @njit
    def _density_derivative_numba(rho, r):
        # Get sorting indices and sort r and rho
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        rho_sorted = rho[sort_idx]
        
        # Calculate derivative on sorted arrays
        drho_dr = np.zeros_like(rho_sorted)
        drho_dr[1:-1] = (rho_sorted[2:] - rho_sorted[:-2]) / (r_sorted[2:] - r_sorted[:-2])
        drho_dr[0] = (rho_sorted[1] - rho_sorted[0]) / (r_sorted[1] - r_sorted[0])
        drho_dr[-1] = rho_sorted[-2] / 2*(r_sorted[-1] - r_sorted[-2]) # assume a phantom additional shell equally above top shell w zero density?
        
        # Get indices that will restore original order and apply them
        unsort_idx = np.argsort(sort_idx)
        return drho_dr[unsort_idx]

    def __call__(self, sim):
        return self._density_derivative_numba(sim.rho_r, sim.r)

@name_strategy("polytrope")
class PolytropicPressureStrategy(PressureStrategy):
    def __call__(self, sim):
        sim.pressure = sim.polytropic_coef * sim.rho_r**(1+1/sim.polytropic_index)
        sim.dpressure_drho = sim.polytropic_coef * (1+1/sim.polytropic_index) * sim.rho_r**(1/sim.polytropic_index)
        return sim.pressure
    
@name_strategy("polytrope_from_internal_energy")
class PolytropicFromInternalEnergyStrategy(PressureStrategy):
    def __call__(self, sim):
        if sim.e_p is None:
            sim.e_p = np.zeros_like(sim.rho_r)
        sim.pressure = 1/sim.polytropic_index * sim.rho_r * sim.e_p / sim.m
        return sim.pressure
    
@name_strategy("zero")
class ZeroPressureStrategy(PressureStrategy):
    def __call__(self, sim):
        sim.pressure = 0
        sim.dpressure_drho = 0
        return sim.pressure
    
@name_strategy("quantum_pressure")
class QuantumPressureStrategy(PressureStrategy):
    def __call__(self, sim):
        dlogrho_dr = np.gradient(np.log(sim.rho_r), sim.r)
        d2logrho_dr2 = np.gradient(dlogrho_dr, sim.r)
        laplacian_log_rho = d2logrho_dr2 + 2/sim.r * dlogrho_dr
        sim.pressure = - sim.hbar2_over_m2/4 * sim.rho_r * laplacian_log_rho
        return sim.pressure
    
@name_strategy("quantum_pressure2")
class QuantumPressure2Strategy(PressureStrategy):
    def __call__(self, sim):
        dlogrho_dr = np.zeros_like(sim.rho_r)
        d2logrho_dr2 = np.zeros_like(sim.rho_r)
    
        if len(sim.rho_r) > 1:
            # Forward difference for first point
            dlogrho = np.log(sim.rho_r[1]) - np.log(sim.rho_r[0])
            dr = sim.r[1] - sim.r[0]
            dlogrho_dr[0] = dlogrho / dr
            
            # For remaining points
            for i in range(1, len(sim.rho_r)):
                dlogrho = np.log(sim.rho_r[i]) - np.log(sim.rho_r[i-1])
                dr = sim.r[i] - sim.r[i-1]
                dlogrho_dr[i] = dlogrho / dr
        
        # Second derivative calculation
        if len(sim.rho_r) > 2:
            # Forward difference for first point
            d2logrho_dr2[0] = (dlogrho_dr[1] - dlogrho_dr[0]) / (sim.r[1] - sim.r[0])
            
            # Central differences for middle points
            for i in range(1, len(sim.rho_r) - 1):
                d2logrho_dr2[i] = (dlogrho_dr[i+1] - dlogrho_dr[i-1]) / (sim.r[i+1] - sim.r[i-1])
            
            # Backward difference for last point
            d2logrho_dr2[-1] = (dlogrho_dr[-1] - dlogrho_dr[-2]) / (sim.r[-1] - sim.r[-2])

        laplacian_log_rho = d2logrho_dr2 + 2/sim.r * dlogrho_dr
        sim.pressure = - sim.hbar2_over_m2/4 * sim.rho_r * laplacian_log_rho
        return sim.pressure

@name_strategy("default")
class DefaultViscosityStrategy(ArtificialViscosityStrategy):
    @staticmethod
    @njit
    def _default_viscosity_numba(rho_r, v, viscosity_cq, r, rho_r_old):
        # Calculate velocity differences (equivalent to np.diff([*v, v[-1]]))
        v_diffs = np.empty_like(v)
        for i in range(1, len(v)):
            v_diffs[i] = v[i] - v[i-1]
        v_diffs[0] = v[0] # at the very center, the velocity is zero
        
        # Set positive velocity differences to zero
        for i in range(len(v_diffs)):
            if v_diffs[i] > 0:
                v_diffs[i] = 0
        
        # Calculate viscosity
        viscosity_q = viscosity_cq * 2/(1/rho_r_old + 1/rho_r) * v_diffs**2
        
        # Calculate gradient of viscosity_q with respect to r
        dq_dr = np.zeros_like(viscosity_q)
        
        # # Forward difference for first point
        # dq_dr[0] = (viscosity_q[1] - viscosity_q[0]) / (r[1] - r[0])
        
        # # Central difference for interior points
        # for i in range(1, len(viscosity_q)-1):
        #     dq_dr[i] = (viscosity_q[i+1] - viscosity_q[i-1]) / (r[i+1] - r[i-1])
        
        # # Backward difference for last point
        # dq_dr[-1] = (viscosity_q[-1] - viscosity_q[-2]) / (r[-1] - r[-2])
        
        return viscosity_q, dq_dr

    def __call__(self, sim):
        sim.viscosity_q, sim.dq_dr = self._default_viscosity_numba(
            sim.rho_r, sim.v, sim.viscosity_cq, sim.r, sim.rho_r_old)
        return sim.viscosity_q
    
@name_strategy("nothing")
class NothingProblematicShellStrategy(ProblematicShellStrategy):
    def __call__(self, sim):
        return
    
@name_strategy("energy")
class EnergyProblematicShellStrategy(ProblematicShellStrategy):

    def __call__(self, sim):
        if sim.problem_idx is None:
            sim.problem_idx = np.zeros(sim.N)
        e_tot0 = sim.snapshots[0]['e_tot'] if sim.snapshots is not None else None
        if e_tot0 is None:
            return
        del_e_on_e = (sim.e_tot - e_tot0) / e_tot0
        sim.problem_idx = np.where(del_e_on_e > 10, sim.problem_idx + 1, 0)
        sim.absorbed = np.where(sim.problem_idx > 1000, True, False)

@name_strategy("keep_edges")
class KeepEdgesShellVolumeStrategy(ShellVolumeStrategy):
    def __call__(self, sim):
        # Calculate the volumes of spherical shells
        # Get sorting indices and sort r
        sort_idx = np.argsort(sim.r)
        r_sorted = sim.r[sort_idx]
        
        # Calculate inner radii for sorted shells
        r_inner = np.zeros_like(r_sorted)
        r_inner[1:] = r_sorted[:-1]
        
        # Calculate volumes for sorted shells
        volumes_sorted = 4/3 * np.pi * (r_sorted**3 - r_inner**3)
        
        # Get indices that will restore original order and apply them
        unsort_idx = np.argsort(sort_idx)
        return volumes_sorted[unsort_idx]
    
@name_strategy("inner_not_zero")
class InnerNotZeroShellVolumeStrategy(ShellVolumeStrategy):
    @staticmethod
    @njit(cache=True)
    def _inner_not_zero_shell_volume_numba(r, r_min):
        # Check if already sorted
        is_sorted = np.all(r[:-1] <= r[1:])
        if is_sorted:
            return 4/3 * np.pi * (r**3 - r_min**3)

        # Get sorting indices and sort r
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        
        # Calculate inner radii for sorted shells
        r_inner = np.zeros_like(r_sorted)
        r_inner[1:] = r_sorted[:-1]
        r_inner[0] = r_min
        
        # Calculate volumes for sorted shells
        volumes_sorted = 4/3 * np.pi * (r_sorted**3 - r_inner**3)
        
        # Get indices that will restore original order and apply them
        unsort_idx = np.argsort(sort_idx)
        return volumes_sorted[unsort_idx]

    def __call__(self, sim):
        # Calculate the volumes of spherical shells
        return self._inner_not_zero_shell_volume_numba(sim.r, sim.r_min)

# @name_strategy("inner_not_zero")
# class InnerNotZeroShellVolumeStrategy(ShellVolumeStrategy):
#     def __call__(self, sim):
#         # Calculate the volumes of spherical shells
#         r_inner = np.zeros_like(sim.r)
#         r_inner[1:] = sim.r[:-1]
#         if len(sim.r) > 1:
#             r_inner[0] = sim.r[0] - (sim.r[1] - sim.r[0])
#         else:
#             r_inner[0] = sim.r_min  # Ensure the single shell extends down to r_min
#         volumes = 4/3 * np.pi * (sim.r**3 - r_inner**3)

#         return volumes


@name_strategy("default")
class DefaultTerminateEarlyStrategy(TerminateEarlyStrategy):
    pass

@name_strategy("shell_density")
class ShellDensityStrategy(CurrentDensityStrategy):
    def __call__(self, sim):
        """Calculate density at each shell's position based on shell volumes"""
        volumes = sim.shell_vol_func()
        return sim.m / volumes

@name_strategy("dr_start_equal")
class DrStartEqualInitialRadiusStrategy(InitialRadiusStrategy):
    def __call__(self, sim):
        return np.linspace(sim.r_max/sim.N, sim.r_max, sim.N)

@name_strategy("rmin_start_equal")
class RminStartEqualInitialRadiusStrategy(InitialRadiusStrategy):
    def __call__(self, sim):
        return np.linspace(sim.r_min, sim.r_max, sim.N)
    
@name_strategy("r0min_start_equal")
class R0minStartEqualInitialRadiusStrategy(InitialRadiusStrategy):
    def __call__(self, sim):
        return np.linspace(sim.r0_min, sim.r_max, sim.N)
    
@name_strategy("equal_mass")
class EqualMassInitialRadiusStrategy(InitialRadiusStrategy):
    def __call__(self, sim):
        """
        Calculate initial radii such that each shell contains equal mass.
        Works by creating a mass profile and inverting it to find radii.
        """
        # Create a fine grid of radii for accurate integration
        n_points = sim.N * 10_000
        eps = 1e-12 # avoid potential singularity at r=0
        r_grid = np.logspace(np.log10(sim.r_min if sim.r_min > 0 else eps), np.log10(sim.r_max), n_points)
        
        # Calculate density on the grid
        rho_grid = sim.rho_func.density_at_r(sim, r_grid)
        
        M_enc_grid = 4*np.pi * integrate.cumulative_trapezoid(
            rho_grid * r_grid**2, 
            r_grid,
            initial=0
        )
        
        # Total mass available for shells
        M_total = M_enc_grid[-1]
        
        fracs = np.linspace(0, 1, sim.N+1)[1:]
        radii = np.interp(fracs*M_total, M_enc_grid, r_grid)
        return radii
    
@name_strategy("equal_mass_neighbor")
class EqualMassNeighborInitialRadiusStrategy(InitialRadiusStrategy):
    def __call__(self, sim):
        """
        Calculate initial radii such that each shell contains equal mass.
        Works by creating a mass profile and inverting it to find radii.
        """
        # Create a fine grid of radii for accurate integration
        n_points = sim.N * 10_000
        eps = 1e-12 # avoid potential singularity at r=0
        r_grid = np.logspace(np.log10(sim.r_min if sim.r_min > 0 else eps), np.log10(sim.r_max), n_points)
        
        # Calculate density on the grid
        rho_grid = sim.rho_func.density_at_r(sim, r_grid)
        
        M_enc_grid = 4*np.pi * integrate.cumulative_trapezoid(
            rho_grid * r_grid**2, 
            r_grid,
            initial=0
        )
        
        # Total mass available for shells
        M_total = M_enc_grid[-1]
        
        fracs = np.linspace(0, 1, sim.N+1)[1:]
        radii = np.interp(fracs*M_total, M_enc_grid, r_grid)
        return radii
    
# @name_strategy("equal_mass")
# class EqualMassInitialRadiusStrategy(InitialRadiusStrategy):
#     def __call__(self, sim):
#         """
#         Calculate initial radii such that each shell contains equal mass.
#         Works by creating a mass profile and inverting it to find radii.
#         """
#         # Create a fine grid of radii for accurate integration
#         n_points = sim.N * 10_000
#         eps = 1e-6 # avoid potential singularity at r=0
#         r_grid = np.linspace(eps, sim.r_max, n_points)
        
#         # Calculate density on the grid
#         rho_grid = sim.rho_func.density_at_r(sim, r_grid)
        
#         # Calculate cumulative mass profile
#         # M(r) = ∫ 4πr²ρ(r)dr
#         mass_integrand = 4 * np.pi * r_grid**2 * rho_grid
#         cumulative_mass = integrate.cumulative_trapezoid(
#             mass_integrand, 
#             r_grid, 
#             initial=0
#         )
        
#         # Total mass available for shells
#         M_total = cumulative_mass[-1]
        
#         # Target masses for each shell
#         target_masses = np.linspace(M_total/sim.N, M_total, sim.N)
        
#         # Interpolate to find radii that give these target masses
#         radius_from_mass = interpolate.interp1d(
#             cumulative_mass, 
#             r_grid,
#             kind='cubic',
#             bounds_error=True
#         )
        
#         try:
#             radii = radius_from_mass(target_masses)
            
#             # Verify the result is monotonically increasing
#             if not np.all(np.diff(radii) > 0):
#                 raise ValueError("Generated radii are not strictly increasing")
                
#             return radii
            
#         except Exception as e:
#             raise RuntimeError(
#                 f"Failed to generate equal mass shells: {str(e)}\n"
#                 f"Check that density profile and radius range are compatible."
#             )

# Implementations of strategies
@name_strategy("thoul_weinberg")
class ThoulWeinbergStepper(StepperStrategy):
    # what they call v_i^(n-1/2) is identified with v_i^n
    def __call__(self, sim):
        v_i_nminushalf = sim.v.copy()
        r_i_n = sim.r.copy()
        p_outside = 0
        p_iminushalf_n = sim.p.copy()
        p_iplushalf_n = np.array([*p_iminushalf_n[1:], p_outside])
        dm_i = sim.m.copy()
        m_i_n = sim.m_enc.copy()
        dt_n = sim.dt

        a_i_n_pressure = -4*np.pi*r_i_n**2 * (p_iplushalf_n - p_iminushalf_n)/dm_i
        a_i_n_gravity = -sim.G*m_i_n/r_i_n**2
        a_i_n = a_i_n_pressure + a_i_n_gravity

        v_i_nplushalf = v_i_nminushalf + a_i_n * dt_n

        dt_nplushalf = dt_n # I don't understand how they do this part
        r_i_nplusone = r_i_n + v_i_nplushalf * dt_nplushalf
        
        rho_n

@name_strategy("adaptive_leapfrog")
class AdaptiveLeapfrogStepper(StepperStrategy):
    def __call__(self, sim):
        halfings = max(1, int(np.ceil(np.log2(sim.dt/sim.dt_calc))))
        dt_n = sim.dt/2**halfings
        for i in range(halfings):
            sim.v = self._kick_numba(sim.v, sim.a, dt_n)
            sim.r = self._drift_numba(sim.r, sim.v, dt_n)
            sim.r_min_func()
            sim.m_enc = sim.m_enc_func()
            sim.a = sim.a_func()
            sim.v = self._kick_numba(sim.v, sim.a, dt_n)
            sim.t += dt_n
        sim.t += sim.dt

    @staticmethod
    @njit    
    def _kick_numba(v, a, dt):
        return v + 0.5 * a * dt
    
    @staticmethod
    @njit
    def _drift_numba(r, v, dt):
        return r + v * dt
    
@name_strategy("leapfrog_quinn")
class LeapfrogQuinnStepper(StepperStrategy):
    def __call__(self, sim):
        self._timestep(sim, sim.dt)
        sim.t += sim.dt

    @staticmethod
    @njit    
    def _kick_numba(v, a, dt):
        return v + 0.5 * a * dt
    
    @staticmethod
    @njit
    def _drift_numba(r, v, dt):
        return r + v * dt
    
    def _timestep(self, sim, dt):
        sim.r = self._drift_numba(sim.r, sim.v, dt)
        if self._select(sim, dt):
            sim.v = self._kick_numba(sim.v, sim.a, dt)
        else:
            sim.r = self._drift_numba(sim.r, sim.v, -dt)
            self._timestep(sim, dt/2)
            sim.v = self._kick_numba(sim.v, sim.a, dt)
            self._timestep(sim, dt/2)

    def _select(self, sim, dt):
        pass

@name_strategy("levi_civita_leapfrog")
class LeviCivitaLeapfrogStepper(StepperStrategy):
    @staticmethod
    @njit
    def _kick_numba(v, dt, G, m_enc, j, r):
        return v + 0.5 * (-G*m_enc/r**2 + j**2/r**4) * dt
    
    @staticmethod
    @njit
    def _drift_numba(r, v, dt):
        return r + 0.5 * r * v * dt

    def __call__(self, sim):
        if sim.t == 0:
            sim.r = np.sqrt(sim.r)
        sim.t_true += 0.5 * sim.r**2 * sim.dt
        v_half = self._kick_numba(sim.v, sim.dt, sim.G, sim.m_enc, sim.j, sim.r) #sim.v + 0.5 * a_lc * sim.dt
        sim.r = self._drift_numba(sim.r, v_half, sim.dt) #sim.r + sim.r**2 * v_half * sim.dt
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        sim.a = sim.a_func()
        sim.v = self._kick_numba(v_half, sim.dt, sim.G, sim.m_enc, sim.j, sim.r) #v_half + 0.5 * a_lc_new * sim.dt
        sim.t += sim.dt
        sim.t_true += 0.5 * sim.r**2 * sim.dt


@name_strategy("leapfrog_hut")
class LeapfrogHutStepper(StepperStrategy):
    def __call__(self, sim):
        # save old values
        r_old = sim.r.copy()
        v_old = sim.v.copy()
        a_old = sim.a.copy()
        dt_old = sim.dt
        self._do_kdk_step(r_old, v_old, a_old, dt_old, sim)
        dt_new = self._get_new_timestep(sim)
        dt_true = 0.5 * (dt_old + dt_new)
        self._do_kdk_step(r_old, v_old, a_old, dt_true, sim)
        sim.dt = dt_true
        sim.t += dt_true

    def _do_kdk_step(self, r_old, v_old, a_old, dt_old, sim):
        v_half = self._kick_numba(v_old, a_old, dt_old)
        r_new = self._drift_numba(r_old, v_half, dt_old)
        
        # Only update non-absorbed shells
        if sim.absorbed is not None:
            sim.r = np.where(sim.absorbed, sim.r, r_new)
            sim.v = np.where(sim.absorbed, sim.v, v_half)
        else:
            sim.r = r_new
            sim.v = v_half
            
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        sim.a = sim.a_func()
        
        # Only update velocity for non-absorbed shells
        v_final = self._kick_numba(v_half, sim.a, dt_old)
        if sim.absorbed is not None:
            sim.v = np.where(sim.absorbed, sim.v, v_final)
        else:
            sim.v = v_final

    def _get_new_timestep(self, sim):
        sim.timescale_func()
        sim.timestep_func()
        return sim.dt

    #@staticmethod
    # def __call__(self, sim):
    #     dt_nplus1 = 0
    #     dt_n = sim.dt_calc
    #     dt_nplus1 = np.inf
    #     r_n = sim.r.copy()
    #     v_n = sim.v.copy()
    #     a_n = sim.a.copy()
    #     dt_min = sim.dt_min
    #     iters = 0
    #     while np.abs(dt_nplus1 - dt_n) > dt_min:
    #         iters += 1
    #         # kick
    #         v_nplushalf = self._kick_numba(v_n, a_n, dt_n)
    #         #drift
    #         r_nplus1 = self._drift_numba(r_n, v_nplushalf, dt_n)
    #         sim.r = r_nplus1
    #         # update a and kick
    #         sim.r_min_func()
    #         sim.m_enc = sim.m_enc_func()
    #         a_nplus1 = sim.a_func()
    #         sim.a = a_nplus1
    #         v_nplus1 = self._kick_numba(v_nplushalf, a_nplus1, dt_n)
    #         sim.v = v_nplus1
    #         # check if timestep changes too much
    #         sim.timescale_func()
    #         sim.timestep_func()
    #         dt_nplus1 = sim.dt_calc
    #         dt_n = 0.5 * (dt_n + dt_nplus1)
    #         # redo from the beginning with new timestep if not converged
    #         sim.r = r_n
    #         sim.v = v_n
    #         sim.a = a_n
    #         if iters > 100:
    #             print("Failed to converge!")
    #             break

    #     # actually update once converged
    #     # if iters > 1:
    #     #     print(iters)
    #     sim.r = r_nplus1
    #     sim.v = v_nplus1
    #     sim.a = a_nplus1
    #     sim.dt = dt_n
    #     sim.t += dt_n

    @staticmethod
    @njit
    def _kick_numba(v, a, dt):
        return v + 0.5 * a * dt
    
    @staticmethod
    @njit
    def _drift_numba(r, v, dt):
        return r + v * dt

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
    

    
@name_strategy("euler")
class EulerStepper(StepperStrategy):
    def __call__(self, sim):
        sim.r = self._euler_numba(sim.r, sim.v, sim.dt)
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        sim.a = sim.a_func()
        sim.v = self._euler_update_v_numba(
            sim.v, sim.a, sim.dt)
        sim.t += sim.dt

    @staticmethod
    @njit(cache=True)
    def _euler_numba(r, v, dt):
        return r + v * dt 

    @staticmethod
    @njit(cache=True)
    def _euler_update_v_numba(v, a, dt):
        return v + a * dt

@name_strategy("velocity_verlet_variable_dt")
class VelocityVerletVariableDtStepper(StepperStrategy):
    def __call__(self, sim):
        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, sim.dt)
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        old_dt = sim.dt
        sim.timestep_func()
        sim.v = self._velocity_verlet_update_v_numba(
            sim.v, a_old, sim.a, old_dt, sim.dt)
        sim.t += old_dt

    @staticmethod
    @njit(cache=True)
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit(cache=True)
    def _velocity_verlet_update_v_numba(v, a_old, a_new, old_dt, new_dt):
        return v + 0.5 * (a_old*old_dt + a_new*new_dt)

@name_strategy("velocity_verlet_with_reflection_reset")
class VelocityVerletWithReflectionResetStepper(StepperStrategy):
    def __call__(self, sim):
        # Update positions
        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, sim.dt)
        
        # Detect reflections
        which_reflected = sim.r < sim.r_min
        dt_half = sim.dt * 0.5
        
        if np.any(which_reflected):
            # Handle reflections
            sim.r[which_reflected] = 2*sim.r_min - sim.r[which_reflected]
            sim.v[which_reflected] = -sim.v[which_reflected]
            
            # Recalculate acceleration
            sim.r_min_func()
            sim.m_enc = sim.m_enc_func()
            sim.a = sim.a_func()
            
            # For reflected particles, start fresh integration
            # from reflection point with half the remaining timestep
            
            sim.r[which_reflected] += (sim.v[which_reflected] * dt_half + 
                                     0.5 * sim.a[which_reflected] * dt_half**2)
        
        # Regular updates for non-reflected particles
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        
        # Final velocity update
        sim.v = np.where(which_reflected,
                        sim.v + sim.a * dt_half,  # reflected particles
                        sim.v + 0.5*(a_old + sim.a)*sim.dt)  # normal particles
        
        sim.t += sim.dt

    @staticmethod
    @njit
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

@name_strategy("stormer_verlet")
class StormerVerletStepper(StepperStrategy):
    def __call__(self, sim):
        prev_r = sim.r.copy()
        if sim.t == 0:
            sim.r = sim.r + sim.v * sim.dt + 0.5 * sim.a * sim.dt**2
        else:
            sim.r = self._stormer_verlet_numba(sim.r, sim.prev_r, sim.a, sim.dt)
        sim.prev_r = prev_r
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        sim.a = sim.a_func()
        sim.t += sim.dt

    @staticmethod
    @njit
    def _stormer_verlet_numba(r, prev_r, a, dt):
        return 2*r - prev_r + a * dt**2
    
@name_strategy("stormer_verlet_variable_dt")
class StormerVerletVariableDtStepper(StepperStrategy):
    def __call__(self, sim):
        prev_r = sim.r.copy()
        prev_dt = sim.dt
        if sim.t == 0:
            sim.r = sim.r + sim.v * sim.dt + 0.5 * sim.a * sim.dt**2
        else:
            sim.r = self._stormer_verlet_variable_dt_numba(sim.r, sim.prev_r, sim.a, sim.dt, sim.prev_dt)
        sim.prev_r = prev_r
        sim.prev_dt = prev_dt
        sim.r_min_func()
        sim.m_enc = sim.m_enc_func()
        sim.a = sim.a_func()
        sim.t += sim.dt

    @staticmethod
    @njit
    def _stormer_verlet_variable_dt_numba(r, prev_r, a, dt, prev_dt):
        return r + (r - prev_r) * (dt/prev_dt) + a * dt * (dt + prev_dt)/2
  
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
    @njit(cache=True)
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit(cache=True)
    def _velocity_verlet_update_v_numba(v, a_old, a_new, dt):
        return v + 0.5 * (a_old + a_new) * dt
    
@name_strategy("velocity_verlet_event_driven")
class VelocityVerletEventDrivenStepper(StepperStrategy):
    def __call__(self, sim):
        dt_remaining = sim.dt
        max_events = 10  # Prevent infinite loops in pathological cases
        events_processed = 0

        while dt_remaining > 1e-12 and events_processed < max_events:
            # Predict positions and velocities for the remaining timestep
            predicted_r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, dt_remaining)
            predicted_v = self._velocity_verlet_update_v_numba(sim.v, sim.a, sim.a, dt_remaining)

            # Detect potential crossings within dt_remaining
            crossing_pairs, crossing_times = self._find_potential_crossings_numba(sim.r, sim.v, sim.a, dt_remaining)

            if crossing_pairs.size == 0:
                # No crossings detected; proceed with full timestep
                sim.r = predicted_r
                sim.v = predicted_v
                sim.t += dt_remaining
                dt_remaining = 0
            else:
                # Find the earliest crossing event
                first_event_idx = np.argmin(crossing_times)
                t_cross = crossing_times[first_event_idx]
                pair = crossing_pairs[first_event_idx]

                if t_cross < 0 or t_cross > dt_remaining:
                    # Invalid crossing time; skip
                    sim.r = predicted_r
                    sim.v = predicted_v
                    sim.t += dt_remaining
                    dt_remaining = 0
                else:
                    # Integrate up to the crossing time
                    if t_cross > 1e-12:
                        sim.r = self._velocity_verlet_numba(sim.r, sim.v, sim.a, t_cross)
                        sim.v = self._velocity_verlet_update_v_numba(sim.v, sim.a, sim.a, t_cross)
                        sim.t += t_cross
                        dt_remaining -= t_cross

                    # Handle the crossing event
                    i, j = pair
                    self._handle_crossing(sim, i, j)

                    events_processed += 1

        # Update other quantities after all events
        sim.m_enc = sim.m_enc_func()
        self._update_acceleration(sim)

    @staticmethod
    @njit
    def _velocity_verlet_numba(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt**2

    @staticmethod
    @njit
    def _velocity_verlet_update_v_numba(v, a_old, a_new, dt):
        return v + 0.5 * (a_old + a_new) * dt

    @staticmethod
    @njit
    def _find_potential_crossings_numba(r, v, a, dt):
        n = len(r)
        crossing_pairs = []
        crossing_times = []

        for i in range(n):
            for j in range(i + 1, n):
                dr = r[j] - r[i]
                dv = v[j] - v[i]
                da = a[j] - a[i]

                # Only consider shells moving towards each other
                if dv >= 0:
                    continue

                # Quadratic equation: 0.5 * da * t^2 + dv * t + dr = 0
                # Solve for t where dr + dv*t + 0.5*da*t^2 = 0
                A = 0.5 * da
                B = dv
                C = dr

                if A == 0:
                    if B == 0:
                        continue  # Parallel motion
                    t_cross = -C / B
                    if 0 < t_cross <= dt:
                        crossing_pairs.append((i, j))
                        crossing_times.append(t_cross)
                else:
                    discriminant = B**2 - 4*A*C
                    if discriminant < 0:
                        continue  # No real solution
                    sqrt_disc = np.sqrt(discriminant)
                    t1 = (-B - sqrt_disc) / (2*A)
                    t2 = (-B + sqrt_disc) / (2*A)
                    t_candidates = [t for t in [t1, t2] if 0 < t <= dt]
                    if t_candidates:
                        t_cross = min(t_candidates)
                        crossing_pairs.append((i, j))
                        crossing_times.append(t_cross)

        if crossing_pairs:
            return np.array(crossing_pairs), np.array(crossing_times)
        else:
            return np.empty((0, 2), dtype=np.int32), np.array([])

    @staticmethod
    @njit
    def _handle_crossing_numba(r, v, a, i, j):
        """
        Define the behavior when two shells cross.
        For example, swap their velocities to prevent further crossing.
        """
        # Swap velocities
        temp_v = v[i]
        v[i] = v[j]
        v[j] = temp_v
        # Optionally, handle other properties like mass or energy

    def _handle_crossing(self, sim, i, j):
        """
        Handle the crossing between shell i and shell j.
        """
        self._handle_crossing_numba(sim.r, sim.v, sim.a, i, j)
        sim.num_crossing += 1
        #logger.debug(f"Handled crossing between shells {i} and {j} at time {sim.t}")

    @staticmethod
    @njit
    def _handle_crossing_numba(r, v, a, i, j):
        """
        Numba-compatible method to swap velocities.
        """
        temp = v[i]
        v[i] = v[j]
        v[j] = temp
        # Add more handling if needed

    def _update_acceleration(self, sim):
        a_old = sim.a.copy()
        sim.a = sim.a_func()
        sim.v = self._velocity_verlet_update_v_numba(sim.v, a_old, sim.a, sim.dt)


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

@name_strategy("beeman_alt_reflect")
class BeemanAltReflectStepper(StepperStrategy):
    def __call__(self, sim):
        if sim.prev_a is None:
            # Use Taylor expansion for the first step
            sim.r = sim.r + sim.v * sim.dt + 0.5 * sim.a * sim.dt**2
            sim.r_min_func()
            sim.m_enc = sim.m_enc_func()
            a_new = sim.a_func()
            v_new = np.where(sim.which_reflected, sim.v, sim.v + sim.a * sim.dt)
        else:
            sim.r = self._beeman_r_numba(
                sim.r, sim.v, sim.a, sim.prev_a, sim.dt)
            sim.r_min_func()
            sim.m_enc = sim.m_enc_func()
            a_new = sim.a_func()
            v_new = np.where(sim.which_reflected, sim.v, self._beeman_v_numba(
                sim.v, sim.a, a_new, sim.prev_a, sim.dt))

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

@name_strategy("t_ta_cycloid")
class TTACycloidTTurnaroundStrategy(TTurnaroundStrategy):
    def __call__(self, sim):
        energy_per_mass_cycloid = -sim.G * sim.m_enc / sim.r + (1/2) * sim.v**2
        return np.pi * sim.G * sim.m_enc / np.abs(2 * energy_per_mass_cycloid)**(3/2)

@name_strategy("t_ta_gas")
class TTAGasTTurnaroundStrategy(TTurnaroundStrategy):
    def __call__(self, sim):
        energy_per_mass_gas = -sim.G * sim.m_enc / sim.r + (1/2) * sim.v**2 + sim.polytropic_index * sim.polytropic_coef * sim.rho_r**(1/sim.polytropic_index)
        return np.pi * sim.G * sim.m_enc / np.abs(2 * energy_per_mass_gas)**(3/2)         

#This is a bit of a hack
@name_strategy("r_is_r_ta")
class RIsRTurnaroundStrategy(RTurnaroundStrategy):
    def __call__(self, sim):
        return sim.r
    
@name_strategy("r_ta_cycloid")
class RTACycloidRTurnaroundStrategy(RTurnaroundStrategy):
    def __call__(self, sim):
        energy_per_mass_cycloid = -sim.G * sim.m_enc / sim.r + (1/2) * sim.v**2
        return sim.G * sim.m_enc / np.abs(energy_per_mass_cycloid)

@name_strategy("r_ta_gas")
class RTAGasRTurnaroundStrategy(RTurnaroundStrategy):
    def __call__(self, sim):
        energy_per_mass_gas = -sim.G * sim.m_enc / sim.r + (1/2) * sim.v**2 + sim.polytropic_index * sim.polytropic_coef * sim.rho_r**(1/sim.polytropic_index)
        return sim.G * sim.m_enc / np.abs(energy_per_mass_gas)
    
@name_strategy("tophat_pert_r_ta")
class TophatPertRTurnaroundStrategy(RTurnaroundStrategy):
    def __call__(self, sim):
        delta = sim.m_pert / (4/3 * np.pi * min(sim.tophat_radius, sim.r)**3)
        #delta = sim.m_pert / (4/3 * np.pi * sim.tophat_radius**3)
        return sim.r**4 / (3*sim.m_pert) * delta
    
@name_strategy("tophat_point_mass_r_ta")
class TophatPointMassRTurnaroundStrategy(RTurnaroundStrategy):
    def __call__(self, sim):
        return sim.r**4 / (6*sim.point_mass)

@name_strategy("grant_gmr_j_at_r_ta_soft_all")
class GrantGMRJAtRTASoftAllAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _grant_gmr_j_at_r_ta_soft_all_a_func_numba(G, m_enc, j, r_soft):
        return -G * m_enc / r_soft**2 + j**2 / r_soft**3

    def __call__(self, sim):
        r_soft = sim.soft_func()
        sim.granted_j = np.where((sim.v < 0) | sim.granted_j)
        j = np.where(sim.granted_j, sim.j, 0)
        return self._grant_gmr_j_at_r_ta_soft_all_a_func_numba(sim.G, sim.m_enc, j, r_soft)
    
@name_strategy("grant_gmr_j_at_r_ta_soft")
class GrantGMRJAtRTASoftAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _grant_gmr_j_at_r_ta_soft_a_func_numba(G, m_enc, j, r):
        return -G*m_enc/r**2 + j**2/r**3

    def __call__(self, sim):
        sim.granted_j = np.where((sim.v < 0) | sim.granted_j, True, False)
        j = np.where(sim.granted_j, sim.j, 0)
        #print(sim.j, j, sim.granted_j)
        return self._grant_gmr_j_at_r_ta_soft_a_func_numba(sim.G, sim.m_enc, j, sim.r)
    
@name_strategy("grant_gmr_j_at_r_ta_simple_fdm")
class GrantGMRJAtRTASoftAllAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _grant_gmr_j_at_r_ta_soft_a_func_numba(G, m_enc, j, r, hbar2_over_m2):
        return -G*m_enc/r**2 + j**2/r**3 + hbar2_over_m2/(2*r**3)

    def __call__(self, sim):
        sim.granted_j = np.where((sim.v < 0) | sim.granted_j, True, False)
        j = np.where(sim.granted_j, sim.j, 0)
        #print(sim.j, j, sim.granted_j)
        return self._grant_gmr_j_at_r_ta_soft_a_func_numba(sim.G, sim.m_enc, j, sim.r, sim.hbar2_over_m2)
    
# @name_strategy("only_quantum_pressure")
    
@name_strategy("grant_gmr_j_at_r_ta_fdm")
class GrantGMRJAtRTAFDMAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _grant_gmr_j_at_r_ta_fdm_a_func_numba(G, m_enc, j, r, d1, d2, d3, hbar2_over_m2):
        aq = hbar2_over_m2 * (1/4 * d3 + 1/4 * d1*d2 - 1/2 * 1/r**2 * d1 + 1/2 * 1/r * d2)
        return -G*m_enc/r**2 + j**2/r**3 + aq

    def __call__(self, sim):
        sorted_indices = np.argsort(sim.r)
        r_halfs = np.zeros_like(sim.r)
        r_halfs[0] = sim.r[sorted_indices[0]]/2
        r_halfs[1:] = (sim.r[sorted_indices[1:]] + sim.r[sorted_indices[:-1]])/2
        
        fs = splrep(r_halfs, np.log(sim.rho_r[sorted_indices]), k=3)
        d1 = splev(sim.r[sorted_indices], fs, der=1)
        d2 = splev(sim.r[sorted_indices], fs, der=2)
        d3 = splev(sim.r[sorted_indices], fs, der=3)
        d1 = d1[np.argsort(sorted_indices)]
        d2 = d2[np.argsort(sorted_indices)]
        d3 = d3[np.argsort(sorted_indices)]
        sim.granted_j = np.where((sim.v < 0) | sim.granted_j, True, False)
        j = np.where(sim.granted_j, sim.j, 0)
        return self._grant_gmr_j_at_r_ta_fdm_a_func_numba(sim.G, sim.m_enc, j, sim.r, d1, d2, d3, sim.hbar2_over_m2)
    
@name_strategy("grant_gmr_j_at_r_ta_fdm2")
class GrantGMRJAtRTAFDM2AccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _grant_gmr_j_at_r_ta_fdm_a_func_numba(G, m_enc, j, r, d1, d2, d3, hbar2_over_m2):
        aq = hbar2_over_m2 * (1/4 * d3 + 1/4 * d1*d2 - 1/2 * 1/r**2 * d1 + 1/2 * 1/r * d2)
        return -G*m_enc/r**2 + j**2/r**3 + aq

    def __call__(self, sim):
        sorted_indices = np.argsort(sim.r)
        r_sorted = sim.r[sorted_indices]
        rho_sorted = sim.rho_r[sorted_indices]
        y = np.log(rho_sorted)
        y_spline = UnivariateSpline(rho_sorted, y, k=3)
        d1 = y_spline.derivative()(rho_sorted)
        d2 = y_spline.derivative()(rho_sorted)
        d3 = y_spline.derivative()(rho_sorted)
        d1 = d1[np.argsort(sorted_indices)]
        d2 = d2[np.argsort(sorted_indices)]
        d3 = d3[np.argsort(sorted_indices)]

        sim.granted_j = np.where((sim.v < 0) | sim.granted_j, True, False)
        j = np.where(sim.granted_j, sim.j, 0)
        return self._grant_gmr_j_at_r_ta_fdm_a_func_numba(sim.G, sim.m_enc, j, sim.r, d1, d2, d3, sim.hbar2_over_m2)
    
@name_strategy("grant_gmr_j_at_r_ta_fdm3")
class GrantGMRJAtRTAFDM3AccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _grant_gmr_j_at_r_ta_fdm_a_func_numba(G, m_enc, j, r, aq):
        return -G*m_enc/r**2 + j**2/r**3 + aq
    
    def loglog_menc_spline_aq(self, rvals, menc, hbar2_over_m2, degree=5, s=None, sim=None, mvals=None):
        loglog_menc_spline = UnivariateSpline(np.log(rvals), np.log(menc), k=degree, s=s)
        b0 = loglog_menc_spline(np.log(rvals))
        b1 = loglog_menc_spline.derivative(1)(np.log(rvals))
        b2 = loglog_menc_spline.derivative(2)(np.log(rvals))
        b3 = loglog_menc_spline.derivative(3)(np.log(rvals))
        b4 = loglog_menc_spline.derivative(4)(np.log(rvals))
        y = b0 - 3*np.log(rvals) - np.log(4*np.pi) + np.log(b1)
        u1 = b1 + b2/b1 - 3
        u2 = b2 + b3/b1 - (b2/b1)**2
        u3 = b3 + b4/b1 - 3*b2*b3/b1**2 + 2*(b2/b1)**3
        
        d1 = u1/rvals
        d2 = (u2-u1)/rvals**2
        d3 = (u3-3*u2+2*u1)/rvals**3

        aq = hbar2_over_m2 * (1/4 * d3 + 1/4 * d1*d2 - 1/2 * 1/rvals**2 * d1 + 1/2 * 1/rvals * d2)
        sim.e_q = mvals * hbar2_over_m2/2 * (1/2 * d2 + 1/4 * d1**2 + 1/rvals * d1)
        return aq

    def __call__(self, sim):
        sorted_indices = np.argsort(sim.r)
        aq_sorted = self.loglog_menc_spline_aq(sim.r[sorted_indices], sim.m_enc[sorted_indices], sim.hbar2_over_m2, degree=5, sim=sim, mvals=sim.m)
        aq = np.zeros_like(sim.r)
        aq[sorted_indices] = aq_sorted
        sim.granted_j = np.where((sim.v < 0) | sim.granted_j, True, False)
        j = np.where(sim.granted_j, sim.j, 0)
        return self._grant_gmr_j_at_r_ta_fdm_a_func_numba(sim.G, sim.m_enc, j, sim.r, aq)
    
@name_strategy("fdm4")
class FDM4AccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _fdm4_a_func_numba(G, m_enc, j, r, aq):
        return -G*m_enc/r**2 + j**2/r**3 + aq
    
    def loglog_menc_spline_aq(self, rvals, menc, hbar2_over_m2, degree=5, s=None, sim=None, mvals=None):
        loglog_menc_spline = UnivariateSpline(np.log(rvals), np.log(menc), k=degree, s=s)
        b0 = loglog_menc_spline(np.log(rvals))
        b1 = loglog_menc_spline.derivative(1)(np.log(rvals))
        b2 = loglog_menc_spline.derivative(2)(np.log(rvals))
        b3 = loglog_menc_spline.derivative(3)(np.log(rvals))
        b4 = loglog_menc_spline.derivative(4)(np.log(rvals))
        y = b0 - 3*np.log(rvals) - np.log(4*np.pi) + np.log(b1)
        u1 = b1 + b2/b1 - 3
        u2 = b2 + b3/b1 - (b2/b1)**2
        u3 = b3 + b4/b1 - 3*b2*b3/b1**2 + 2*(b2/b1)**3
        
        d1 = u1/rvals
        d2 = (u2-u1)/rvals**2
        d3 = (u3-3*u2+2*u1)/rvals**3

        aq = hbar2_over_m2 * (1/4 * d3 + 1/4 * d1*d2 - 1/2 * 1/rvals**2 * d1 + 1/2 * 1/rvals * d2)
        sim.e_q = mvals * hbar2_over_m2/2 * (1/2 * d2 + 1/4 * d1**2 + 1/rvals * d1)
        return aq

    def __call__(self, sim):
        sorted_indices = np.argsort(sim.r)
        aq_sorted = self.loglog_menc_spline_aq(sim.r[sorted_indices], sim.m_enc[sorted_indices], sim.hbar2_over_m2, degree=5, sim=sim, mvals=sim.m)
        aq = np.zeros_like(sim.r)
        aq[sorted_indices] = aq_sorted
        return self._fdm4_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, aq)


@name_strategy("grant_fdm_at_r_ta")
class GrantFDMAtRTAAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _grant_fdm_at_r_ta_a_func_numba(G, m_enc, j, r, d1, d2, d3, hbar2_over_m2):
        aq = hbar2_over_m2 * (1/4 * d3 + 1/4 * d1*d2 - 1/2 * 1/r**2 * d1 + 1/2 * 1/r * d2)
        return -G*m_enc/r**2 + j**2/r**3 + aq

    def __call__(self, sim):
        r_halfs = np.zeros_like(sim.r)
        r_halfs[0] = sim.r[0]/2
        r_halfs[1:] = (sim.r[1:] + sim.r[:-1])/2
        fs = splrep(r_halfs, np.log(sim.rho_r), k=3)
        d1 = splev(sim.r, fs, der=1)
        d2 = splev(sim.r, fs, der=2)
        d3 = splev(sim.r, fs, der=3)
        sim.j = np.where((sim.v < 0) & (sim.j == 0), sim.j_coef * np.sqrt(sim.G * sim.m_enc * sim.r_ta), sim.j)
        hbar2_over_m2 = np.where((sim.v < 0) & (sim.hbar2_over_m2 == 0), 10_000, 0)
        return self._grant_fdm_at_r_ta_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, d1, d2, d3, sim.hbar2_over_m2)




@name_strategy("soft_new")
class SoftNewAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _soft_new_a_func_numba(G, m_enc, j, r, softlen):
        return -G * m_enc * r / (r**2 + softlen)**(3/2) + j**2/r**3 * r/ (r**2 + softlen**2)**2

    def __call__(self, sim):
        return self._soft_new_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.softlen)


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
    
@name_strategy("quantum_potential")
class QuantumPotentialAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _quantum_potential_a_func_numba(G, m_enc, j, r_soft, d1, d2, d3, hbar2_over_m2):
        aq = hbar2_over_m2 * (1/4 * d3 + 1/4 * d1*d2 - 1/2 * 1/r_soft**2 * d1 + 1/2 * 1/r_soft * d2)
        return -G * m_enc / r_soft**2 + j**2 / r_soft**3 + aq

    def __call__(self, sim):
        r_soft = sim.soft_func()
        r_halfs = np.zeros_like(sim.r)
        r_halfs[0] = sim.r[0]/2
        r_halfs[1:] = (sim.r[1:] + sim.r[:-1])/2
        fs = splrep(r_halfs, np.log(sim.rho_r), k=3)
        d1 = splev(sim.r, fs, der=1)
        d2 = splev(sim.r, fs, der=2)
        d3 = splev(sim.r, fs, der=3)
        return self._quantum_potential_a_func_numba(sim.G, sim.m_enc, sim.j, r_soft, d1, d2, d3, sim.hbar2_over_m2)
    
@name_strategy("soft_le_delliou")
class SoftLeDelliouAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _soft_le_delliou_a_func_numba(G, m_enc, j, r, r_soft):
        return -G * m_enc * r / r_soft**3 + j**2 * r / r_soft**4

    def __call__(self, sim):
        r_soft = sim.soft_func()
        return self._soft_le_delliou_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, r_soft)
    
@name_strategy("gas")
class GasAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _gas_a_func_numba(G, m_enc, j, r, rho, rho_prime, dpressure_drho):
        return -G * m_enc / r**2 + j**2 / r**3 + 1/rho * dpressure_drho * rho_prime

    def __call__(self, sim):
        sim.rho_prime = sim.drhodr_func()
        sim.pressure_func() # update pressure and dpressure_drho
        return self._gas_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.rho_r, sim.rho_prime, sim.dpressure_drho)
    
@name_strategy("gas_w_dissipation")
class GasWDissipationAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    #TODO: NJIT WORKING
    def _gas_w_dissipation_a_func_numba(G, m_enc, j, r, rho, rho_prime, dpressure_drho, v, relaxation_time, t):
        # NOTE: including the v in the acceleration equation goes against the assumptions of velocity verlet and may fail
        if t < (5 * relaxation_time):
            relax_term = v / relaxation_time
        else:
            relax_term = 0 # turn off the relaxation term as T+W do at a certain point.
        return -G * m_enc / r**2 + j**2 / r**3 - 1/rho * dpressure_drho * rho_prime - relax_term

    def __call__(self, sim):
        sim.rho_prime = sim.drhodr_func()
        sim.pressure_func() # update pressure and dpressure_drho
        return self._gas_w_dissipation_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.rho_r, sim.rho_prime, sim.dpressure_drho, sim.v, sim.relaxation_time, sim.t)
    
@name_strategy("gas_vis_w_dissipation")
class GasVisWDissipationAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    #TODO: NJIT WORKING
    def _gas_vis_w_dissipation_a_func_numba(G, m_enc, j, r, rho, rho_prime, dpressure_drho, v, relaxation_time, t, dq_dr):
        # NOTE: including the v in the acceleration equation goes against the assumptions of velocity verlet and may fail
        if t < (5 * relaxation_time):
            relax_term = v / relaxation_time
        else:
            relax_term = 0 # turn off the relaxation term as T+W do at a certain point.
        return -G * m_enc / r**2 + j**2 / r**3 - 1/rho * dpressure_drho * rho_prime - relax_term - 1/rho * dq_dr

    def __call__(self, sim):
        sim.rho_prime = sim.drhodr_func()
        sim.pressure_func() # update pressure and dpressure_drho
        sim.viscosity_func() # update viscosity and dq_dr
        return self._gas_vis_w_dissipation_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.rho_r, sim.rho_prime, sim.dpressure_drho, sim.v, sim.relaxation_time, sim.t, sim.dq_dr)
    
@name_strategy("gas_vis_w_dissipation2")
class GasVisWDissipationAccelerationStrategy2(AccelerationStrategy):
    @staticmethod
    #TODO: NJIT WORKING
    def _gas_vis_w_dissipation2_a_func_numba(G, m_enc, j, r, rho, rho_prime, dpressure_drho, v, relaxation_time, t, dq_dr, pressure):
        # NOTE: including the v in the acceleration equation goes against the assumptions of velocity verlet and may fail
        if t < (5 * relaxation_time):
            relax_term = v / relaxation_time
        else:
            relax_term = 0 # turn off the relaxation term as T+W do at a certain point.
        dpressure_dr = np.gradient([*pressure, 0], [*r, r[-1] + (r[-1] - r[-2])])[:-1]
        return -G * m_enc / r**2 + j**2 / r**3 - 1/rho * dpressure_dr - relax_term - 1/rho * dq_dr

    def __call__(self, sim):
        sim.rho_prime = sim.drhodr_func()
        sim.pressure_func() # update pressure and dpressure_drho
        sim.viscosity_func() # update viscosity and dq_dr
        return self._gas_vis_w_dissipation2_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.rho_r, sim.rho_prime, sim.dpressure_drho, sim.v, sim.relaxation_time, sim.t, sim.dq_dr, sim.pressure)
    
@name_strategy("gas_explicit")
class GasExplicitAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    @njit
    def _gas_explicit_a_func_numba(G, m_enc, j, r, m, rho, dpressure):
        return -G * m_enc / r**2 + j**2 / r**3 - 4*np.pi*r**2 * dpressure/m
    
    def __call__(self, sim):
        # assume no crossings so order is right
        rho = np.zeros_like(sim.r)
        pressure = np.zeros_like(sim.r)
        dpressure = np.zeros_like(sim.r)
        for i in range(len(sim.r)):
            if i == 0:
                volume = 4*np.pi*sim.r[i]**3
            else:
                volume = 4*np.pi*(sim.r[i]**3 - sim.r[i-1]**3)
            rho[i] = sim.m[i] / volume
            pressure[i] = sim.polytropic_coef * rho[i]**sim.polytropic_index
        for i in range(len(pressure)-1):
            dpressure[i] = pressure[i+1] - pressure[i]
        dpressure[-1] = -pressure[-1]
        sim.rho_r = rho
        sim.pressure = pressure
        return self._gas_explicit_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.m, rho, dpressure)


@name_strategy("gas_vis_dis_fourpi")
class GasVisDisFourPiAccelerationStrategy(AccelerationStrategy):

    @staticmethod
    @njit
    def _gas_vis_dis_fourpi_a_func_numba(G, m_enc, j, r, rho,  v, relaxation_time, t, pressure, m, q):
        # NOTE: including the v in the acceleration equation goes against the assumptions of velocity verlet and may fail
        # Create relax_term array 
        if t < (100 * relaxation_time):
            relax_term = v / relaxation_time
        else:
            relax_term = np.zeros_like(v)
        
        # Compute pressure differences (equivalent to np.diff)
        dpressure = np.empty_like(pressure)
        for i in range(len(pressure)-1):
            dpressure[i] = pressure[i+1] - pressure[i]
        # Set last element pressure difference
        dpressure[-1] = dpressure[-2]#-pressure[-1]# dpressure[-2]
        
        # Compute viscosity differences (equivalent to np.diff)
        dq = np.empty_like(q)
        for i in range(len(q)-1):
            dq[i] = q[i+1] - q[i]
        # Set last element viscosity difference
        dq[-1] = 0
        
        # Compute and return acceleration
        return -G * m_enc / r**2 + j**2 / r**3 - 4*np.pi*r**2 * dpressure/m - relax_term - 4*np.pi*r**2 * dq/m

    def __call__(self, sim):
        #sim.rho_prime = sim.drhodr_func()
        sim.pressure_func() # update pressure and dpressure_drho
        sim.viscosity_func() # update viscosity and dq_dr
        return self._gas_vis_dis_fourpi_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.rho_r, sim.v, sim.relaxation_time, sim.t, sim.pressure, sim.m, sim.viscosity_q)
    
    
@name_strategy("no_crossing_term")
class NoCrossingTermAccelerationStrategy(AccelerationStrategy):
    @staticmethod
    #TODO: NJIT WORKING
    def _no_crossing_term_a_func_numba(G, m_enc, j, r, v):
        dr = np.gradient([0, *r])[1:]
        dv = np.gradient([0, *v])[1:]
        return -G * m_enc / r**2 + j**2 / r**3 - 10*dv/dr

    def __call__(self, sim):
        return self._no_crossing_term_a_func_numba(sim.G, sim.m_enc, sim.j, sim.r, sim.v)
    

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
    @njit(cache=True)
    def _m_enc_inclusive_numba(r, m, point_mass):
        sorted_indices = np.argsort(r)
        sorted_masses = m[sorted_indices]
        cumulative_mass = np.cumsum(sorted_masses)
        m_enc = np.empty_like(cumulative_mass)
        m_enc[sorted_indices] = cumulative_mass + point_mass
        return m_enc

    def __call__(self, sim):
        return self._m_enc_inclusive_numba(sim.r, sim.m, sim.point_mass)

@name_strategy("gaussian")
class GaussianEnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    def _m_enc_gaussian_numba(r, m, point_mass, thickness_coef):
        N = len(r)
        # 1) Build a 2D grid: X[i,j] = r[i],  Y[i,j] = r[j]
        #    so that G[i,j] = X[i,j]**2 * exp(-(X[i,j] - Y[i,j])**2 / (2σ²))
        Rgrid = r[:, None]          # shape (N,1)
        Cgrid = r[None, :]          # shape (1,N)
        G = Rgrid**2 * np.exp(- (Rgrid - Cgrid)**2 / (2*thickness_coef**2))  # (N,N)

        # 2) Approximate ∫₀ʳᵢ G(x,rⱼ) dx by cumulative trapezoid along axis=0
        #    First compute the spacings Δx:
        dx = np.diff(r)             # length N-1
        #    Now do a cumulative trapezoidal sum:
        #    For trapezoid: ∫₀ʳᵢ f(x) dx ≈ Σₖ₌₀ⁱ⁻¹ (fₖ + fₖ₊₁)/2 * Δxₖ
        #    We can build the mid-point sums and then do a cumsum:
        mid = (G[:-1, :] + G[1:, :]) * 0.5  # shape (N-1, N)
        cumint = np.concatenate([
            np.zeros((1, N)),               # integral at r=0 is zero
            np.cumsum(mid * dx[:, None], axis=0)
        ], axis=0)                           # shape (N, N)

        # 3) Now weigh by m[j] and sum over j, and multiply front factors:
        prefac = np.sqrt(2/np.pi) * thickness_coef
        #    cumint[i,j] is ∫₀ʳᵢ x² e^{-(x-rⱼ)²/(2σ²)} dx
        m_enc = point_mass + prefac * (cumint * m[None, :]).sum(axis=1)

        return m_enc

    def __call__(self, sim):
        sim.thickness_func()
        return self._m_enc_gaussian_numba(sim.r, sim.m, sim.point_mass, sim.thickness_coef)
    
@name_strategy("neighbor")
class NeighborEnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit(cache=True)
    def _m_enc_neighbor_numba(r, m, point_mass, r_min):
        sorted_indices = np.argsort(r)
        sorted_r = r[sorted_indices]
        sorted_masses = m[sorted_indices]
        
        # Calculate mass below (excluding self)
        mass_below = np.cumsum(sorted_masses) - sorted_masses
        
        # Calculate self mass contribution based on volume fractions
        n = len(r)
        self_mass = np.zeros_like(sorted_masses)
        
        # Handle first shell. It's innermost neighbor is instead rmin
        if n > 1:
            vol_fraction = (sorted_r[0]**3 - 0**3) / (sorted_r[1]**3 - 0**3)
            self_mass[0] = sorted_masses[0] * vol_fraction
        else:
            self_mass[0] = sorted_masses[0] # When there is only one shell, the self mass is the mass of the shell
            
        # Handle middle shells
        for i in range(1, n-1):
            vol_fraction = (sorted_r[i]**3 - sorted_r[i-1]**3) / (sorted_r[i+1]**3 - sorted_r[i-1]**3)
            self_mass[i] = sorted_masses[i] * vol_fraction
            
        # Handle last shell
        if n > 1:
            self_mass[-1] = sorted_masses[-1]
        
        # Calculate total enclosed mass
        tot_enc = mass_below + self_mass + point_mass
        
        # Restore original order
        m_enc = np.empty_like(tot_enc)
        m_enc[sorted_indices] = tot_enc
        return m_enc

    def __call__(self, sim):
        return self._m_enc_neighbor_numba(sim.r, sim.m, sim.point_mass, sim.r_min)
    
@name_strategy("neighbor2")
class Neighbor2EnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_neighbor2_numba(r, m, point_mass, r_min, absorbed):
        # Get indices of active (not absorbed) shells
        active_indices = np.where(~absorbed)[0]
        
        if len(active_indices) == 0:
            # All shells are absorbed, return their own mass plus point mass
            return m + point_mass
        
        # Initialize enclosed mass array
        m_enc = np.zeros_like(m)
        
        # Sort active shells by radius
        active_r = r[active_indices]
        sort_idx = np.argsort(active_r)
        sorted_active_indices = active_indices[sort_idx]
        sorted_r = r[sorted_active_indices]
        sorted_masses = m[sorted_active_indices]
        
        # Calculate mass below (excluding self) for active shells
        mass_below = np.cumsum(sorted_masses) - sorted_masses
        
        # Calculate self mass contribution based on volume fractions
        n = len(sorted_r)
        self_mass = np.zeros_like(sorted_masses)
        
        # Handle first active shell
        if n > 1:
            vol_fraction = (sorted_r[0]**3 - 0**3) / (sorted_r[1]**3 - 0**3)
            self_mass[0] = 2 * sorted_masses[0] * vol_fraction
        else:
            self_mass[0] = sorted_masses[0]
            
        # Handle middle active shells
        for i in range(1, n-1):
            vol_fraction = (sorted_r[i]**3 - sorted_r[i-1]**3) / (sorted_r[i+1]**3 - sorted_r[i-1]**3)
            self_mass[i] = 2 * sorted_masses[i] * vol_fraction
            
        # Handle last active shell
        if n > 1:
            self_mass[-1] = sorted_masses[-1]
        
        # Calculate total enclosed mass for active shells
        tot_enc = mass_below + self_mass + point_mass
        
        # Update enclosed mass for active shells only
        m_enc[sorted_active_indices] = tot_enc
        
        return m_enc

    def __call__(self, sim):
        return self._m_enc_neighbor2_numba(sim.r, sim.m, sim.point_mass, sim.r_min, sim.absorbed)
    
@name_strategy("neighbor3")
class Neighbor3EnclosedMassStrategy(EnclosedMassStrategy):
    @staticmethod
    @njit
    def _m_enc_neighbor3_numba(r, m, point_mass, r_min, absorbed):
        # Get indices of active (not absorbed) shells
        active_indices = np.where(~absorbed)[0]
        
        if len(active_indices) == 0:
            # All shells are absorbed, return their own mass plus point mass
            return m + point_mass
        
        # Initialize enclosed mass array
        m_enc = np.zeros_like(m)
        
        # Sort active shells by radius
        active_r = r[active_indices]
        sort_idx = np.argsort(active_r)
        sorted_active_indices = active_indices[sort_idx]
        sorted_r = r[sorted_active_indices]
        sorted_masses = m[sorted_active_indices]
        
        # Calculate mass below (excluding self) for active shells
        mass_below = np.zeros_like(sorted_masses) #np.cumsum(sorted_masses) - sorted_masses
        
        # Calculate self mass contribution based on volume fractions
        n = len(sorted_r)
        self_mass = np.zeros_like(sorted_masses)
        
        # Handle first active shell
        if n > 1:
            vol_fraction = (sorted_r[0]**3 - 0**3) / (sorted_r[1]**3 - 0**3)
            self_mass[0] =  sorted_masses[0] * vol_fraction
        else:
            self_mass[0] = sorted_masses[0]
            
        # Handle middle active shells
        for i in range(1, n-1):
            vol_fraction = (sorted_r[i]**3 - sorted_r[i-1]**3) / (sorted_r[i+1]**3 - sorted_r[i-1]**3)
            self_mass[i] = sorted_masses[i] * vol_fraction
            
        # Handle last active shell
        if n > 1:
            self_mass[-1] = sorted_masses[-1]
        
        # Calculate total enclosed mass for active shells
        tot_enc = mass_below + self_mass + point_mass
        
        # Update enclosed mass for active shells only
        m_enc[sorted_active_indices] = tot_enc
        
        return m_enc

    def __call__(self, sim):
        return self._m_enc_neighbor3_numba(sim.r, sim.m, sim.point_mass, sim.r_min, sim.absorbed)
    
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
    @njit(cache=True)
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
        #--------------------r[j]-thicknesses[j]-----
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
    

@name_strategy("kernel")
class KernelEnclosedMassStrategy(EnclosedMassStrategy):

    @staticmethod
    @njit(cache=True)
    def _kernel_numba(r_i, r_j, sigma):
        q = (r_i - r_j) / sigma
        return math.exp(-0.5 * q * q) / (sigma * math.sqrt(2.0 * math.pi))

    
    @staticmethod
    @njit(cache=True)
    def _m_enc_kernel_numba(r, m, point_mass, hbar2_over_m2, v):
        sorted_indices = np.argsort(r)
        sorted_r = r[sorted_indices]
        sorted_masses = m[sorted_indices]
        # sigmas = np.sqrt(hbar2_over_m2)/v
        sigmas = np.zeros_like(sorted_r)
        sigmas[0] = sorted_r[1] - sorted_r[0]
        sigmas[-1] = sorted_r[-1] - sorted_r[-2]
        sigmas[1:-1] = (sorted_r[2:] - sorted_r[:-2]) / 2
        sigmas = sigmas / 2
        m_enc = np.zeros_like(m)
        for i in range(len(r)):
            for j in range(len(r)):
                q = (sorted_r[i] - sorted_r[j]) / sigmas[j]
                norm_factor = 0.5 * (1 + math.erf(sorted_r[j] / (np.sqrt(2) * sigmas[j])))
                m_enc[i] += sorted_masses[j] * np.exp(-0.5 * q * q) / (sigmas[j] * np.sqrt(2.0 * np.pi)) / norm_factor
        m_enc[sorted_indices] = m_enc + point_mass
        return m_enc
    


    def __call__(self, sim):
        return self._m_enc_kernel_numba(sim.r, sim.m, sim.point_mass, sim.hbar2_over_m2, sim.v)
    
@name_strategy("quantum_potential")
class QuantumPotentialEnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _quantum_potential_energy_func_numba(G, m, v, m_enc, r, j, phi_q):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_q = m * phi_q
        e_tot = e_k + e_g + e_r + e_q
        return e_k, e_g, e_r, e_q, e_tot

    def __call__(self, sim):
        r_halfs = np.zeros_like(sim.r)
        r_halfs[0] = sim.r[0]/2
        r_halfs[1:] = (sim.r[1:] + sim.r[:-1])/2
        fs = splrep(sim.r, np.log(sim.rho_r), k=3)
        d1 = splev(sim.r, fs, der=1)
        d2 = splev(sim.r, fs, der=2)
        phi_q = 1/2 * d2 + 1/4 * d1**2 + 1/sim.r * d1
        phi_q = 1/2 * sim.hbar2_over_m2 * phi_q
        sim.e_k, sim.e_g, sim.e_r, sim.e_q, sim.e_tot = self._quantum_potential_energy_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j, phi_q)

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
        
@name_strategy("grant_soft")
class GrantSoftEnergyStrategy(EnergyStrategy):
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
        
@name_strategy("kin_grav_rot_absorbed")
class KinGravRotAbsorbedEnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _energy_func_numba(G, m, v, m_enc, r, j):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_tot = e_k + e_g + e_r
        return e_k, e_g, e_r, e_tot

    def __call__(self, sim):
        # this should keep the absorbed shells at their last energy before absorption
        if np.any(sim.absorbed):
            sim.e_k[~sim.absorbed], sim.e_g[~sim.absorbed], sim.e_r[~sim.absorbed], sim.e_tot[~sim.absorbed] = self._energy_func_numba(
                sim.G, sim.m[~sim.absorbed], sim.v[~sim.absorbed], sim.m_enc[~sim.absorbed], sim.r[~sim.absorbed], sim.j)
        else:
            sim.e_k, sim.e_g, sim.e_r, sim.e_tot = self._energy_func_numba(
                sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j) 

@name_strategy("kin_grav_rot_fdm_absorbed")
class KinGravRotFDMAbsorbedEnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _energy_func_numba(G, m, v, m_enc, r, j, e_q):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_tot = e_k + e_g + e_r + e_q
        return e_k, e_g, e_r, e_tot

    def __call__(self, sim):
        # this should keep the absorbed shells at their last energy before absorption
        if sim.e_q is None:
            sim.e_q = np.zeros_like(sim.r)
        if np.any(sim.absorbed):
            sim.e_k[~sim.absorbed], sim.e_g[~sim.absorbed], sim.e_r[~sim.absorbed], sim.e_tot[~sim.absorbed] = self._energy_func_numba(
                sim.G, sim.m[~sim.absorbed], sim.v[~sim.absorbed], sim.m_enc[~sim.absorbed], sim.r[~sim.absorbed], sim.granted_j if sim.granted_j is not None else 0, sim.e_q[~sim.absorbed])
        else:
            sim.e_k, sim.e_g, sim.e_r, sim.e_tot = self._energy_func_numba(
                sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.granted_j if sim.granted_j is not None else 0, sim.e_q)               
        
@name_strategy("gas")
class GasEnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _gas_energy_func_numba(G, m, v, m_enc, r, j, pressure, rho_r, polytropic_index, q):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_p = m * polytropic_index * (pressure) / rho_r
        e_q = m * q / rho_r
        e_tot = e_k + e_g + e_r + e_p + e_q
        return e_k, e_g, e_r, e_p, e_q, e_tot

    def __call__(self, sim):
        sim.e_k, sim.e_g, sim.e_r, sim.e_p, sim.e_q, sim.e_tot = self._gas_energy_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j, sim.pressure, sim.rho_r, sim.polytropic_index, sim.viscosity_q)
        
@name_strategy("gas2")
class Gas2EnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _gas2_energy_func_numba(G, m, v, m_enc, r, j, pressure, rho_r, polytropic_index, q, e_p_prev, e_q_prev, rho_r_old):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_p = e_p_prev - m * pressure * (1/rho_r - 1/rho_r_old) 
        e_q = e_q_prev - m * q * (1/rho_r - 1/rho_r_old)
        e_tot = e_k + e_g + e_r + e_p + e_q
        return e_k, e_g, e_r, e_p, e_q, e_tot

    def __call__(self, sim):
        e_p_prev = sim.e_p if sim.e_p is not None else np.zeros_like(sim.m)
        e_q_prev = sim.e_q if sim.e_q is not None else np.zeros_like(sim.m)
        sim.e_k, sim.e_g, sim.e_r, sim.e_p, sim.e_q, sim.e_tot = self._gas2_energy_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j, sim.pressure, sim.rho_r, sim.polytropic_index, sim.viscosity_q, e_p_prev, e_q_prev, sim.rho_r_old)
        
@name_strategy("gas3")
class Gas3EnergyStrategy(EnergyStrategy):
    @staticmethod
    @njit
    def _gas3_energy_func_numba(G, m, v, m_enc, r, j, pressure, rho_r, polytropic_index, q, e_p_prev, rho_r_old):
        e_k = 0.5 * m * v**2
        e_g = -G * m * m_enc / r
        e_r = 0.5 * m * j**2 / r**2
        e_p = e_p_prev - m * (pressure + q) * (1/rho_r - 1/rho_r_old) 
        e_tot = e_k + e_g + e_r + e_p
        return e_k, e_g, e_r, e_p, e_tot

    def __call__(self, sim):
        e_p_prev = sim.e_p if sim.e_p is not None else np.zeros_like(sim.m)
        sim.e_k, sim.e_g, sim.e_r, sim.e_p, sim.e_tot = self._gas3_energy_func_numba(
            sim.G, sim.m, sim.v, sim.m_enc, sim.r, sim.j, sim.pressure, sim.rho_r, sim.polytropic_index, sim.viscosity_q, e_p_prev, sim.rho_r_old)

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
def calculate_t_dyn(G, m_enc, r, absorbed, t_max):
    active = ~absorbed
    return np.min(1/np.sqrt(G * m_enc[active] / r[active]**3)) if np.any(active) else t_max

@njit
def calculate_t_dynnext(G, m_enc, r, v, dt, absorbed, t_max):
    active = ~absorbed
    if not np.any(active):
        return t_max
    r_next = r + v * dt
    return np.min(1/np.sqrt(G * m_enc[active] / r_next[active]**3))

@njit
def calculate_t_dynr(G, m_enc, r, absorbed, t_max):
    active = ~absorbed
    return np.min(1/np.sqrt(G * m_enc[active] / r[active]**3) * r[active]) if np.any(active) else t_max

#@njit
def calculate_t_j(j, r, absorbed, t_max):
    active = ~absorbed
    # not strictly correct for j different for each particle
    return np.min(r[active]**2 / j) if np.any(active) else t_max

# @njit
# def calculate_t_vel(r, v, r_max, absorbed, t_max, eps=1e-2):
#     active = ~absorbed
#     return np.min(r_max / (np.abs(v[active])+eps)) if np.any(active) else t_max

@njit
def calculate_t_vel(r, v, r_max, absorbed, t_max, eps=1e-6):
    active = ~absorbed
    return np.min(r[active] / (np.abs(v[active])+eps)) if np.any(active) else t_max

# @njit
# def calculate_t_acc(r, a, r_max, absorbed, t_max, eps=1e-2):
#     active = ~absorbed
#     return np.min(np.sqrt(r_max / (np.abs(a[active])+eps))) if np.any(active) else t_max

@njit
def calculate_t_acc(r, a, r_max, absorbed, t_max, eps=1e-6):
    active = ~absorbed
    return np.min(np.sqrt(r[active] / (np.abs(a[active])+eps))) if np.any(active) else t_max

@njit
def calculate_t_levi2(r, absorbed, t_max, G, m_enc, j):
    active = ~absorbed
    return np.min(np.sqrt(r[active]/(np.min(r[active])**2 * (-G*m_enc[active]/r[active]**2 + j**2/r[active]**3)))) if np.any(active) else t_max


@njit
def calculate_t_zero(r, v, absorbed, t_max):
    t_zero = np.inf
    for i in range(len(r)):
        if not absorbed[i] and v[i] < 0:
            t = r[i] / abs(v[i])
            if t < t_zero:
                t_zero = t
    return t_zero if t_zero < np.inf else t_max

@njit(cache=True)
def calculate_t_rmin(r, v, r_min, absorbed, t_max):
    t_rmin = np.inf
    for i in range(len(r)):
        if not absorbed[i] and v[i] < 0:
            t = (r[i] - r_min) / abs(v[i])
            if t < t_rmin:
                t_rmin = t
    return t_rmin if t_rmin < np.inf else t_max

@njit(cache=True)
def calculate_t_ref(r, v, r_min, absorbed, t_max):
    t_ref = np.inf
    for i in range(len(r)):
        if not absorbed[i] and v[i] < 0:
            t = r_min / abs(v[i])
            if t < t_ref:
                t_ref = t
    return t_ref if t_ref < np.inf else t_max

@njit
def calculate_t_rmina(r, v, a, r_min, absorbed, t_max):
    t_rmina = np.inf
    for i in range(len(r)):
        if not absorbed[i] and v[i] < 0 and a[i] < 0:
            t = np.sqrt((r[i] - r_min) / np.abs(a[i]))
            if t < t_rmina:
                t_rmina = t
    return t_rmina if t_rmina < np.inf else t_max

@njit(cache=True)
def calculate_t_doublecross(r, v, absorbed, t_max):
    t_doublecross = np.inf
    sorted_indices = np.argsort(r)
    n = len(r)
    for i in range(n):
        if not absorbed[sorted_indices[i]]:
            # Check shell 2 positions ahead in sorted order
            j = i + 2
            if j < n and not absorbed[sorted_indices[j]]:
                dr = np.abs(r[sorted_indices[i]] - r[sorted_indices[j]])
                dv = np.abs(v[sorted_indices[i]] - v[sorted_indices[j]])
                if dv > 1e-9:
                    t = dr / dv
                    if t < t_doublecross:
                        t_doublecross = t
    
    return t_doublecross if t_doublecross < np.inf else t_max

@njit(cache=True)
def calculate_t_cross(r, v, absorbed, t_max):
    n = len(r)
    t_cross = np.inf
    for i in range(n):
        if absorbed[i]:
            continue
        for j in range(i+1, n):
            if absorbed[j]:
                continue
            dr = np.abs(r[i] - r[j])
            dv = np.abs(v[i] - v[j])
            if dv > 1e-9:
                t = dr / dv
                if t > 0 and t < t_cross:
                    t_cross = t
    return t_cross if t_cross < np.inf else t_max

@njit(cache=True)
def calculate_t_cross2(r, v, absorbed, t_max):
    n = len(r)
    t_cross = np.inf
    for i in range(n):
        if absorbed[i]:
            continue
        for j in range(i+1, n):
            if absorbed[j]:
                continue
            dr = np.abs(r[i] - r[j])
            dv = np.abs(v[i] - v[j])
            if dv > 1e-9:
                t = dr / dv
                if t > 0 and t**2 < t_cross:
                    t_cross = t**2
    return t_cross if t_cross < np.inf else t_max

@njit(cache=True)
def calculate_t_crossa(r, a, absorbed, t_max):
    n = len(r)
    t_cross = np.inf
    for i in range(n):
        if absorbed[i]:
            continue
        for j in range(i+1, n):
            if absorbed[j]:
                continue
            dr = np.abs(r[i] - r[j])
            da = np.abs(a[i] - a[j])
            if da > 1e-9:
                t = np.sqrt(dr / da)
                if t > 0 and t < t_cross:
                    t_cross = t
    return t_cross if t_cross < np.inf else t_max

@njit
def calculate_t_thickness(r, v, thicknesses, absorbed, t_max):
    t_thickness = np.inf
    n = len(r) # number of particles
    for shell in range(n):
        if absorbed[shell]:
            continue
        for other_shell in range(shell+1, n):
            if absorbed[other_shell]:
                continue
            v_rel = v[shell] - v[other_shell]
            r_rel = r[shell] - r[other_shell] # pos if shell on top
            if np.abs(v_rel) > 1e-9: # avoid division by zero
                approaching = (r_rel / v_rel) < 0 
                # Check if shells are about to overlap
                crossing_time = np.abs((r_rel + np.minimum(thicknesses[shell], thicknesses[other_shell])) / v_rel)
                if approaching and crossing_time < t_thickness:
                    t_thickness = crossing_time
    return t_thickness if t_thickness < np.inf else t_max

@njit
def calculate_t_sound_old(r, pressure, rho_r, polytropic_index, t_max, r_min):
    t_sound = np.inf
    # assuming no crossings
    dr = np.empty_like(r)
    dr[1:] = r[1:] - r[:-1]
    dr[0] = dr[1] - r_min
    v_sound = np.sqrt( (1 + 1/polytropic_index) * pressure / rho_r)
    dt = dr / v_sound
    return np.min(dt) if np.any(dt < np.inf) else t_max

@njit
def calculate_t_sound(r, pressure, rho_r, polytropic_index, t_max, r_min, e_p):
    t_sound = np.inf
    # assuming no crossings
    dr = np.empty_like(r)
    dr[1:] = r[1:] - r[:-1]
    dr[0] = dr[1] - r_min
    v_sound =  np.sqrt((1+1/polytropic_index)*polytropic_index*e_p)
    dt = dr / v_sound
    return np.min(dt) if np.any(dt < np.inf) else t_max

@njit
def calculate_t_jeans(r, v, hbar2_over_m2, t_max):
    return np.min(np.sqrt(hbar2_over_m2 / (np.abs(v)))) if np.any(np.abs(v) > 1e-9) else t_max

@njit
def calculate_t_nothing(t_max):
    return t_max

@njit
def calculate_t_dbvel(v, hbar2_over_m2, absorbed, t_max, eps=1e-6):
    active = ~absorbed
    return np.min(np.sqrt(hbar2_over_m2) / (np.abs(v[active])**2+eps)) if np.any(active) else t_max

@njit
def calculate_t_vel2(v, absorbed, t_max, eps=1e-6):
    active = ~absorbed
    return np.min(1 / (np.abs(v[active])**2+eps)) if np.any(active) else t_max

@njit
def calculate_t_phase(r, m_enc, j, hbar2_over_m2, G, absorbed, t_max):
    active = ~absorbed
    # if isinstance(j, np.ndarray):
    #     jval = j[active]
    # else:
    #     jval = j
    return np.min(np.abs(np.sqrt(hbar2_over_m2) * (G * m_enc[active] / r[active] + j**2 / (2*r[active]**2) ) )) if np.any(active) else t_max


class CompositeTimeScaleStrategy(TimeScaleStrategy):
    def __init__(self, components: List[TimeScaleComponent]):
        self.components = components

    def __call__(self, sim):
        time_scales = {}
        for component in self.components:
            time_scales[component.name] = component.func(sim)
        
        for name, value in time_scales.items():
            setattr(sim, f"t_{name}", value)
        
        t_remaining = 1/sim.safety_factor * max(2*(sim.t_max - sim.t), 1)
        sim.min_time_scale = min(min(time_scales.values()), t_remaining) if time_scales.values() else t_remaining

    @classmethod
    @lru_cache(maxsize=None)
    def create(cls, *component_names):
        component_map = {
            "dyn": lambda sim: calculate_t_dyn(sim.G, sim.m_enc, sim.r, sim.absorbed, sim.t_max),
            "j": lambda sim: calculate_t_j(sim.j, sim.r, sim.absorbed, sim.t_max),
            "zero": lambda sim: calculate_t_zero(sim.r, sim.v, sim.absorbed, sim.t_max),
            "rmin": lambda sim: calculate_t_rmin(sim.r, sim.v, sim.r_min, sim.absorbed, sim.t_max),
            "rmina": lambda sim: calculate_t_rmina(sim.r, sim.v, sim.a, sim.r_min, sim.absorbed, sim.t_max),
            "vel": lambda sim: calculate_t_vel(sim.r, sim.v, sim.r_max, sim.absorbed, sim.t_max),
            "acc": lambda sim: calculate_t_acc(sim.r, sim.a, sim.r_max, sim.absorbed, sim.t_max),
            "cross": lambda sim: calculate_t_cross(sim.r, sim.v, sim.absorbed, sim.t_max),
            "crossa": lambda sim: calculate_t_crossa(sim.r, sim.a, sim.absorbed, sim.t_max),
            "cross2": lambda sim: calculate_t_cross2(sim.r, sim.v, sim.absorbed, sim.t_max),
            "dynnext": lambda sim: calculate_t_dynnext(sim.G, sim.m_enc, sim.r, sim.v, sim.dt, sim.absorbed, sim.t_max),
            "dynr": lambda sim: calculate_t_dynr(sim.G, sim.m_enc, sim.r, sim.absorbed, sim.t_max),
            "thickness": lambda sim: calculate_t_thickness(sim.r, sim.v, sim.thicknesses, sim.absorbed, sim.t_max),
            "ref": lambda sim: calculate_t_ref(sim.r, sim.v, sim.r_min, sim.absorbed, sim.t_max),
            "sound": lambda sim: calculate_t_sound(sim.r, sim.pressure, sim.rho_r, sim.polytropic_index, sim.t_max, sim.r_min, sim.e_p if sim.e_p is not None else 0),
            "jeans": lambda sim: calculate_t_jeans(sim.r, sim.v, sim.hbar2_over_m2, sim.t_max),
            "doublecross": lambda sim: calculate_t_doublecross(sim.r, sim.v, sim.absorbed, sim.t_max),
            "nothing": lambda sim: calculate_t_nothing(sim.t_max),
            "levi2": lambda sim: calculate_t_levi2(sim.r, sim.absorbed, sim.t_max, sim.G, sim.m_enc, sim.j),
            "dbvel": lambda sim: calculate_t_dbvel(sim.v, sim.hbar2_over_m2, sim.absorbed, sim.t_max),
            "vel2": lambda sim: calculate_t_vel2(sim.v, sim.absorbed, sim.t_max),
            "phase": lambda sim: calculate_t_phase(sim.r, sim.m_enc, sim.j, sim.hbar2_over_m2, sim.G, sim.absorbed, sim.t_max),
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
            "vflip": lambda sim: save_on_direction_change(sim.v, sim.prev_v if sim.prev_v is not None else sim.deque[-2]['v'] if len(sim.deque) > 1 else None),
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

@name_strategy("const_but_calculate")
class ConstButCalculateTimeStepStrategy(TimeStepStrategy):
    @staticmethod
    @njit
    def _simple_adaptive_timestep_numba(safety_factor, min_time_scale):
        return safety_factor * min_time_scale

    def __call__(self, sim):
        sim.dt_calc = max(sim.dt_min, self._simple_adaptive_timestep_numba(
            sim.safety_factor, sim.min_time_scale))

@name_strategy("equal_mass")
class EqualMassDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _equal_mass_rho_func_numba(r, m_pert, N):
        # More general approach without assuming equal spacing
        shell_volumes = 4/3 * np.pi * np.diff(np.concatenate(([0], r))**3)
        return (m_pert / N) / shell_volumes

    def __call__(self, sim):
        return self._equal_mass_rho_func_numba(sim.r, sim.m_pert, sim.N)
    
@name_strategy("background_plus_tophat")
class BackgroundPlusTophatDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_tophat_rho_func_numba(r, rho_bar, m_pert, tophat_radius):
        delta = m_pert / (4/3 * np.pi * tophat_radius**3)
        retval = np.where(r <= tophat_radius, rho_bar + delta, rho_bar)
        return retval
    
    def density_at_r(self, sim, r):
        return self._background_plus_tophat_rho_func_numba(r=r, rho_bar=sim.rho_bar, m_pert=sim.m_pert, tophat_radius=sim.tophat_radius)
    
    def __call__(self, sim):
        return self.density_at_r(sim, sim.r)
    
@name_strategy("background_plus_tophat2")
class BackgroundPlusTophatDensityStrategy2(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_tophat_rho_func_numba(r, rho_bar, delta, tophat_radius):
        retval = np.where(r <= tophat_radius, rho_bar * (1+delta), rho_bar)
        return retval
    
    def density_at_r(self, sim, r):
        return self._background_plus_tophat_rho_func_numba(r=r, rho_bar=sim.rho_bar, delta=sim.delta, tophat_radius=sim.tophat_radius)
    
    def __call__(self, sim):
        return self.density_at_r(sim, sim.r)

@name_strategy("const")
class ConstDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _const_rho_func_numba(r_max, m_pert):
        return m_pert / (4/3 * np.pi * r_max**3)
    
    def density_at_r(self, sim, r):
        return self._const_rho_func_numba(sim.r_max, sim.m_pert)

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

@name_strategy("background_plus_power_law_total") # SUCH THAT M_PERT/M_BACKGROUND = sim.delta
class BackgroundPlusPowerLawTotalDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_power_law_total_rho_func_numba(r, rho_bar, delta, gamma, r_max):
        power_law_part = delta * (3+gamma)/3 * (r/r_max)**gamma
        retval = rho_bar * (1 + power_law_part)
        return retval
    
    def density_at_r(self, sim, r):
        return self._background_plus_power_law_total_rho_func_numba(r=r, rho_bar=sim.rho_bar, delta=sim.delta, gamma=sim.gamma, r_max=sim.r_max)
    
    def __call__(self, sim):
        return self.density_at_r(sim, sim.r)
    
@name_strategy("background_plus_power_law")
class BackgroundPlusPowerLawDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_power_law_rho_func_numba(r, rho_bar, delta, gamma, r_small):
        power_law_part = delta * (r/r_small)**gamma
        retval = rho_bar * (1 + power_law_part)
        return retval
    
    def density_at_r(self, sim, r):
        if sim.r_small is None:
            sim.r_small = sim.r_min if sim.r_min > 0 else sim.r_max/sim.N
        return self._background_plus_power_law_rho_func_numba(r=r, rho_bar=sim.rho_bar, delta=sim.delta, gamma=sim.gamma, r_small=sim.r_small)
    
    def __call__(self, sim):
        return self.density_at_r(sim, sim.r)

@name_strategy("background_plus_gaussian")
class BackgroundPlusGaussianDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_gaussian_rho_func_numba(r, rho_bar, delta, gaussian_radius):
        gaussian_part = np.exp(-(r / gaussian_radius)**2)
        retval = rho_bar * (1 + delta * gaussian_part)
        return retval
    
    def density_at_r(self, sim, r):
        return self._background_plus_gaussian_rho_func_numba(r=r, rho_bar=sim.rho_bar, delta=sim.delta, gaussian_radius=sim.tophat_radius)
    
    def __call__(self, sim):
        return self.density_at_r(sim, sim.r)
    
@name_strategy("background_plus_sinc")
class BackgroundPlusSincDensityStrategy(DensityStrategy):
    @staticmethod
    @njit
    def _background_plus_sinc_rho_func_numba(r, rho_bar, delta, sinc_radius):
        sinc_part = np.sinc(r / sinc_radius)
        retval = rho_bar * (1 + delta * sinc_part)
        return retval
    
    def density_at_r(self, sim, r):
        return self._background_plus_sinc_rho_func_numba(r=r, rho_bar=sim.rho_bar, delta=sim.delta, sinc_radius=sim.tophat_radius)
    
    def __call__(self, sim):
        return self.density_at_r(sim, sim.r)

@name_strategy("const")
class ConstShellThicknessStrategy(ShellThicknessStrategy):
    @staticmethod
    @njit
    def _const_shell_thickness_numba(r, thickness_coef):
        return np.full(len(r), thickness_coef)

    def __call__(self, sim):
        sim.thicknesses = self._const_shell_thickness_numba(sim.r, sim.thickness_coef)

@name_strategy("zero")
class ZeroAngularMomentumStrategy(AngularMomentumStrategy):
    def __call__(self, sim):
        return np.zeros_like(sim.r)

@name_strategy("const")
class ConstAngularMomentumStrategy(AngularMomentumStrategy):
    def __call__(self, sim):
        return sim.j_coef

@name_strategy("gmr")
class GMRAngularMomentumStrategy(AngularMomentumStrategy):
    def __call__(self, sim):
        return sim.j_coef * np.sqrt(sim.G * sim.m_enc * sim.r_ta)
    
@name_strategy("gm")
class GMAngularMomentumStrategy(AngularMomentumStrategy):
    def __call__(self, sim):
        return sim.j_coef * np.sqrt(sim.G * sim.m_enc)
    
@name_strategy("hubble")
class HubbleInitialVelocityStrategy(InitialVelocityStrategy):
    def __call__(self, sim):
        return sim.H * sim.r
    
@name_strategy("peculiar")
class PeculiarVelocityStrategy(InitialVelocityStrategy):
    def __call__(self, sim):
        avg_delta = self.avg_delta_func(sim)
        return sim.H * sim.r * (1 - avg_delta/3)
    
    def avg_delta_func(self, sim):
        m_background = sim.rho_bar * 4/3 * np.pi * sim.r**3
        integral = (sim.m_enc - m_background) / (4*np.pi * sim.rho_bar)
        return 3 / sim.r**3 * integral 
    
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
    
@name_strategy("nothing")
class NothingRMinStrategy(RMinStrategy):
    def __call__(self, sim):
        pass
    
@name_strategy("reflect_after_leaving")
class ReflectAfterLeavingRMinStrategy(RMinStrategy):
    def __init__(self):
        self.has_been_above_rmin = None

    def __call__(self, sim):
        # Initialize tracking array if not already done
        if self.has_been_above_rmin is None:
            self.has_been_above_rmin = sim.r > sim.r_min

        # Update tracking of which shells have been above r_min
        self.has_been_above_rmin = np.logical_or(self.has_been_above_rmin, sim.r > sim.r_min)
        
        sim.r, sim.v, sim.which_reflected = self._r_min_func_numba(
            sim.r, sim.v, sim.r_min, self.has_been_above_rmin)

    @staticmethod
    @njit
    def _r_min_func_numba(r, v, r_min, has_been_above_rmin):
        which_reflected = np.zeros_like(r, dtype=np.bool_)
        for i in range(len(r)):
            if has_been_above_rmin[i] and r[i] < r_min:
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
    
@name_strategy("reflect_or_absorb_if_too_deep")
class ReflectOrAbsorbIfTooDeepStrategy(RMinStrategy):
    def __call__(self, sim):
        sim.r, sim.v, sim.a, sim.absorbed, sim.which_reflected = self._reflect_or_absorb_if_too_deep_numba(sim.r, sim.v, sim.a, sim.r_min, sim.absorbed if sim.absorbed is not None else np.zeros_like(sim.r, dtype=np.bool_))

    @staticmethod
    @njit
    def _reflect_or_absorb_if_too_deep_numba(r, v, a, r_min, absorbed):
        which_reflected = np.zeros_like(r, dtype=np.bool_)
        absorbed = np.logical_or(absorbed, r <= 0.9*r_min)
        r = np.where(absorbed, r_min, r)
        v = np.where(absorbed, 0, v)
        a = np.where(absorbed, 0, a)
        for i in range(len(r)):
            if r[i] < r_min and not absorbed[i]:
                r[i] = 2 * r_min - r[i]
                v[i] = -v[i]
                which_reflected[i] = True
        return r, v, a, absorbed, which_reflected
    
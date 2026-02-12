import numpy as np
from functools import partial
import h5py
import logging
import types
from numba import jit, njit
from simulation_strategies import *
from utils import *
from collections import deque, defaultdict

import time
from datetime import timedelta

from pprint import pprint

# Setup logging only once at the module level
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False  # Add this line

# Add to imports
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

import sys

class SphericalCollapse:
    def __init__(self, config=None):
        # Default parameters
        self._set_default_parameters()
        
        # Update if config dictionary is provided
        if config:
            self._update_from_config(config)

        # Capture initial parameters
        self._initial_params = self._capture_initial_params()
                    
        self.setup()

        self.progress_bar = None

    def _set_default_parameters(self):
        # Move all default parameter initialization here
        self.start_time = None
        self.G = 1
        self.N = 100
        self.r_max = 1
        self.r_min = 1e-6
        self.r0_min = self.r_max/self.N
        self.m_pert = 1
        self.rho_bar = 0
        self.dt = 1e-5
        self.dt_min = 0
        self.t_true = 0
        self.min_time_scale = None
        self.save_dt = 1e-4
        self.next_save_time = 0
        self.t_max = 2
        self.t = 0
        self.point_mass = 0
        self.initial_radius_strategy = "dr_start_equal"
        self.stepper_strategy = "velocity_verlet"
        self.density_strategy = "const"
        self.ang_mom_strategy = "gmr"
        self.soft_func_strategy = "const_soft"
        self.accel_strategy = "soft_grav"
        self.m_enc_strategy = "overlap_inclusive"
        self.initial_mass_strategy = "integrated_mass"
        self.r_ta_strategy = "r_ta_cycloid"
        self.t_ta_strategy = "t_ta_cycloid"
        self.initial_v_strategy = "hubble"
        self.energy_strategy = "kin_grav_rot"
        #self.shell_vol_func = types.MethodType(keep_edges_shell_vol_func, self)
        self.shell_volume_strategy = "inner_not_zero"
        self.timescale_strategy = "dyn"
        self.timestep_strategy = "simple_adaptive"
        self.thickness_strategy = "const"
        self.save_strategy = "default"
        self.r_min_strategy = "reflect"
        self.shell_density_strategy = "shell_density"
        self.pressure_strategy = "zero"
        self.drhodr_strategy = "finite_diff"
        self.viscosity_strategy = "default"
        self.point_mass_strategy = "rmin_rho_r"
        self.problematic_shell_strategy = "nothing"
        self.problem_idx = None
        self.r_small = None
        self.viscosity_cq = 0
        self.viscosity_q = 0
        self.polytropic_index = -1
        self.polytropic_coef = 0
        self.relaxation_time = 0
        self.delta = 0
        self.rho_r = None
        self.which_reflected = None
        self.absorbed = None
        self.refletion_events = []
        self.num_crossing = 0
        self.r = None
        self.v = None
        self.a = None
        self.prev_a = None
        self.prev_v = None
        self.prev_m_enc = None
        self.prev_r = None
        self.prev_dt = None
        self.m = None
        self.m_enc = None
        self.j = None
        self.j_coef = 0
        self.thickness_coef = 0
        self.softlen = 0
        self.hbar2_over_m2 = 0
        self.e_tot = None
        self.e_g = None
        self.e_k = None
        self.e_r = None
        self.e_p = None
        self.e_q = None
        self.r_ta = None
        self.t_ta = None
        self.t_thickness = None
        self.t_dyn = None
        self.t_vel = None
        self.t_acc = None
        self.t_cross = None
        self.t_crossa = None
        self.t_cross2 = None
        self.t_zero = None
        self.t_rmin = None
        self.t_rmina = None
        self.t_ref = None
        self.t_dynr = None
        self.t_dynnext = None
        self.t_sound = None
        self.t_jeans = None
        self.t_j = None
        self.gamma = 0
        self.H = 0
        self.safety_factor = 1e-3
        self.thicknesses = None
        self.snapshots = []
        self.save_filename = None
        self.deque_size = 0
        self.deque = deque(maxlen=self.deque_size)
        self.tophat_radius = 1
        self.has_been_above_rmin = None
        self.show_progress = True

    def _update_from_config(self, config):
        for key, value in config.items():
            if key not in self.__dict__:
                raise AttributeError(f"Attribute {key} does not exist in the object.")
            if callable(value):
                setattr(self, key, types.MethodType(value, self))
            else:
                setattr(self, key, value)

    def _capture_initial_params(self):
            """
            Capture all non-None and non-empty parameters before setup.
            """
            params = {}
            for attr, value in self.__dict__.items():
                if not attr.startswith('_') and value is not None and value != []:
                    if isinstance(value, (int, float, str, bool, np.number, np.ndarray)):
                        params[attr] = value
                    elif isinstance(value, types.MethodType):
                        # For methods, store the strategy name
                        params[attr] = value.__func__.__name__
            return params

    def get_parameters_dict(self):
        """
        Return the dictionary of initial simulation parameters.
        """
        return self._initial_params.copy()

    def _initialize_strategies(self):
        strategy_mappings = {
            "initial_radius_func": (InitialRadiusFactory, self.initial_radius_strategy),
            "stepper": (StepperFactory, self.stepper_strategy),
            "r_ta_func": (RTurnaroundFactory, self.r_ta_strategy),
            "t_ta_func": (TTurnaroundFactory, self.t_ta_strategy),
            "a_func": (AccelerationFactory, self.accel_strategy),
            "soft_func": (SofteningFactory, self.soft_func_strategy),
            "m_enc_func": (EnclosedMassFactory, self.m_enc_strategy),
            "timescale_func": (TimeScaleFactory, self.timescale_strategy),
            "energy_func": (EnergyFactory, self.energy_strategy),
            "timestep_func": (TimeStepFactory, self.timestep_strategy),
            "thickness_func": (ShellThicknessFactory, self.thickness_strategy),
            "rho_func": (DensityFactory, self.density_strategy),
            "initial_v_func": (InitialVelocityFactory, self.initial_v_strategy),
            "j_func": (AngularMomentumFactory, self.ang_mom_strategy),
            "save_func": (SaveFactory, self.save_strategy),
            "r_min_func": (RMinFactory, self.r_min_strategy),
            "shell_vol_func": (ShellVolumeFactory, self.shell_volume_strategy),
            "shell_density_func": (CurrentDensityFactory, self.shell_density_strategy),
            "pressure_func": (PressureFactory, self.pressure_strategy),
            "drhodr_func": (DensityDerivativeFactory, self.drhodr_strategy),
            "viscosity_func": (ArtificialViscosityFactory, self.viscosity_strategy),
            "point_mass_func": (PointMassFactory, self.point_mass_strategy),
            "initial_mass_func": (InitialMassFactory, self.initial_mass_strategy),
            "problematic_shell_func": (ProblematicShellFactory, self.problematic_shell_strategy),
        }
        for attr_name, (factory, strategy_name) in strategy_mappings.items():
            try:
                strategy_instance = factory.create(strategy_name)
                if strategy_instance is None:
                    raise ValueError(f"Strategy creation for {attr_name} returned None")
                setattr(self, attr_name, types.MethodType(strategy_instance, self))
            except Exception as e:
                logger.error(f"Error initializing {attr_name}: {str(e)}")
                raise

    def _generate_progress_bar(self, progress, bar_length=30):
        filled_length = int(bar_length * progress // 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        return f'[{bar}] {progress:.0f}%'

    def setup(self):
        pprint(self.get_parameters_dict())
        # Initialize factories
        self._initialize_strategies()
        # Initialize radial positions
        self.r = self.initial_radius_func()
        self.point_mass = self.point_mass_func()
        self.which_reflected = np.zeros_like(self.r, dtype=np.bool_)
        self.absorbed = np.zeros_like(self.r, dtype=np.bool_)
        # Initialize masses for each shell
        self.m = self.initial_mass_func()
        self.thickness_func()
        # Calculate initial enclosed mass
        self.m_enc = self.m_enc_func()
        #print(f'r={self.r}, m={self.m}, point_mass={self.point_mass}, m_enc={self.m_enc}')
        # Log initial shell density
        self.rho_r = self.shell_density_func()
        self.rho_r_old = self.rho_r.copy()
        # Initialize velocities
        self.v = self.initial_v_func()
        # Calculate initial turnaround radii and times
        self.r_ta = self.r_ta_func()
        self.t_ta = self.t_ta_func()
        # Initialize angular momentum
        self.j = self.j_func()
        self.granted_j = np.zeros_like(self.j, dtype=np.bool_)
        # Calculate initial acceleration
        self.a = self.a_func()
        # Calculate initial energy
        self.energy_func()
        # timescales
        self.timescale_func()
        self.timestep_func()
        self._save_if_necessary()
        logger.info("Simulation setup complete")

    def run(self):
        self.start_time = time.time()
        self.next_save_time = self.save_dt
        t_max = self.t_max
        
        # Check if running in a Jupyter notebook
        is_notebook = 'ipykernel' in sys.modules
        
        if (is_notebook or self.show_progress) and self.show_progress:
            with tqdm_notebook(
                total=100,
                desc="Simulation Progress",
                bar_format='{l_bar}{bar}{r_bar}',
                ncols=None,
                mininterval=1.0,  # Increased from 0.1 to reduce update frequency
                ascii=False,
                unit='%',
            ) as self.progress_bar:
                last_progress = 0
                progress_update_interval = 0.01  # Update progress every 1% instead of every step
                while self.t < t_max:
                    self._update_simulation()
                    self._save_if_necessary()
                    
                    # Update progress less frequently
                    current_progress = (self.t / t_max) * 100
                    progress_diff = current_progress - last_progress
                    if progress_diff >= progress_update_interval:
                        # Cache sorted radii to avoid repeated sorting
                        r_sorted = np.sort(self.r)
                        r_min = r_sorted[0]
                        r_max = r_sorted[-1]
                        r_diff_min = 0 if len(r_sorted) == 1 else np.min(r_sorted[1:] - r_sorted[:-1])
                        
                        # self.progress_bar.set_postfix({
                        #     'dt': f"{self.dt:.4e}",
                        #     't': f"{self.t:.4e}",
                        #     'smallest_r': f"{r_min:.4e}",
                        #     'largest_r': f"{r_max:.4e}",
                        #     'closest_shells': f"{r_diff_min:.4e}",
                        # }, refresh=False)
                        self.progress_bar.update(progress_diff)
                        last_progress = current_progress
        else:
            # Run without progress bar for maximum performance
            while self.t < t_max:
                self._update_simulation()
                self._save_if_necessary()
        
        if self.save_filename:
            save_to_hdf5(self, self.save_filename)
        return self.get_results_dict()

    def _update_simulation(self):
        self.stepper()
        #self.prev_r = self.r.copy()
        #self.rho_r_old = self.rho_r.copy()
        self.rho_r = self.shell_density_func()
        self.thickness_func()
        self.energy_func()
        self.problematic_shell_func()
        self.timescale_func()
        self.timestep_func()

    def update_deque(self):
        """
        Update the deque with the current simulation parameters.
        Each element in the deque is a dictionary containing the same parameters as in the `save` method.
        """
        data = {
            't': self.t,
            'dt': self.dt,
            'r': self.r.copy(),
            'v': self.v.copy(),
            'a': self.a.copy(),
            'rho_r': self.rho_r.copy(),
            'm_enc': self.m_enc.copy(),
            'e_tot': self.e_tot.copy() if self.e_tot is not None else None,
            'e_g': self.e_g.copy() if self.e_g is not None else None,
            'e_k': self.e_k.copy() if self.e_k is not None else None,
            'e_r': self.e_r.copy() if self.e_r is not None else None,
            't_dyn': self.t_dyn,
            't_vel': self.t_vel,
            't_acc': self.t_acc,
            't_cross': self.t_cross,
            't_crossa': self.t_crossa,
            't_cross2': self.t_cross2,
            't_zero': self.t_zero,
            't_rmin': self.t_rmin,
            't_rmina': self.t_rmina,
            't_ref': self.t_ref,
            't_thickness': self.t_thickness,
            't_dynr': self.t_dynr,
            't_dynnext': self.t_dynnext,
            't_sound': self.t_sound,
            't_j': self.t_j,
            't_jeans': self.t_jeans,
            'num_crossing': self.num_crossing,
            'm': self.m.copy(),
        }
        self.deque.append(data)

    def _save_if_necessary(self):
        should_save = False
        if self.deque_size > 0:
            self.update_deque()
         # Check if it's time for a periodic save
        if self.t >= self.next_save_time or self.t == 0:
            should_save = True
            while self.next_save_time <= self.t:
                self.next_save_time += self.save_dt 

        # Check if any save condition is met
        if self.save_func():
            # print("CONDITION MET AT TIME", self.t)
            should_save = True

        # Perform save if any condition is met
        if should_save:
            
            self.save()

    def _update_progress(self, progress):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if progress > 0:
            estimated_total_time = elapsed_time / (progress / 100)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0

        # Format times
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        remaining_str = str(timedelta(seconds=int(remaining_time)))
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

        # Generate progress bar
        progress_bar = self._generate_progress_bar(progress)

        # Log the progress
        logger.info(
            f"{current_timestamp} | Elapsed: {elapsed_str} | Remaining: {remaining_str} | {progress_bar}"
        )

    def save(self):
        # Save relevant parameters of the simulation
        # Use views instead of copies when possible to reduce memory allocation
        data = {
            't': self.t,
            'dt': self.dt,
            'r': self.r,  # Remove .copy() - use view
            'v': self.v,  # Remove .copy() - use view
            'a': self.a,  # Remove .copy() - use view
            'm_enc': self.m_enc,  # Remove .copy() - use view
            'rho_r': self.rho_r,  # Remove .copy() - use view
            'pressure': self.pressure if hasattr(self, 'pressure') else None,
            'dpressure_drho': self.dpressure_drho if hasattr(self, 'dpressure_drho') else None,
            'rho_prime': self.rho_prime if hasattr(self, 'rho_prime') else None,
            'e_tot': self.e_tot,  # Remove .copy() - use view
            'e_g': self.e_g,  # Remove .copy() - use view
            'e_k': self.e_k,  # Remove .copy() - use view
            'e_r': self.e_r,  # Remove .copy() - use view
            'e_p': self.e_p if self.e_p is not None else None,
            'e_q': self.e_q if self.e_q is not None else None,
            't_dyn': self.t_dyn,
            't_vel': self.t_vel,
            't_acc': self.t_acc,
            't_cross': self.t_cross,
            't_crossa': self.t_crossa,
            't_cross2': self.t_cross2,
            't_zero': self.t_zero,
            't_rmin': self.t_rmin,
            't_rmina': self.t_rmina,
            't_ref': self.t_ref,
            't_thickness': self.t_thickness,
            't_dynr': self.t_dynr,
            't_dynnext': self.t_dynnext,
            't_sound': self.t_sound,
            't_j': self.t_j,
            't_jeans': self.t_jeans,
            'num_crossing': self.num_crossing,
            'm': self.m,  # Remove .copy() - use view
            'deque': list(self.deque),
            't_true': self.t_true,
        }
        self.snapshots.append(data)

    def get_results_dict(self):
        results = {key: [] for key in self.snapshots[0].keys()}
        for snapshot in self.snapshots:
            for key, value in snapshot.items():
                results[key].append(value)
        return {key: np.array(value) for key, value in results.items()}  
    
    def _capture_initial_params(self):
        """
        Capture all non-None and non-empty parameters before setup.
        """
        params = {}
        for attr, value in self.__dict__.items():
            if not attr.startswith('_') and value is not None and not (isinstance(value, list) and len(value) == 0):
                if isinstance(value, (int, float, str, bool, np.number, np.ndarray)):
                    params[attr] = value
                elif isinstance(value, types.MethodType):
                    # For methods, store the strategy name
                    params[attr] = value.__func__.__name__
        return params

    def get_parameters_dict(self):
        """
        Return the dictionary of initial simulation parameters.
        """
        return self._initial_params.copy()


    def __str__(self):
        result = []
        for attr_name, attr_value in self.__dict__.items():
            if callable(attr_value):
                # If it's callable, print the attribute name
                result.append(f"{attr_name}: {attr_value.__name__}")
            else:
                # Otherwise, print the attribute name and its value
                result.append(f"{attr_name}: {attr_value}")
        return "\n".join(result)
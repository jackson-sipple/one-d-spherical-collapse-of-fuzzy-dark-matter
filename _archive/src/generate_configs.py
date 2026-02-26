import json
import os
from itertools import product
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
from astropy import units as u, constants as const
from matplotlib import pyplot as plt
from scipy.stats import beta
from scipy.special import beta as betafct
from numba import jit, jitclass
from scipy import interpolate
from scipy.interpolate import interp1d
import logging
import sys
import os
sys.path.insert(1, '/home/jsipple/one_d_spherical_collapse/one-d-spherical-collapse-of-fuzzy-dark-matter/src')
import importlib
import simulation_strategies, collapse, plotting, utils, analtyic_formulas
importlib.reload(simulation_strategies)
importlib.reload(collapse)
importlib.reload(plotting)
importlib.reload(utils)
importlib.reload(analtyic_formulas)
from simulation_strategies import *
from collapse import *
from plotting import *
from utils import *
from analtyic_formulas import *
import cProfile
import pstats

def generate_config(N=100, r_max=3, delta_i=1e-1, r_min=1e-6, dt_min=1e-6, outfile=None):
    """
    Generate a simulation configuration.

    Parameters:
    - N (int): Number of shells.
    - r_max (float): Maximum radius.
    - delta_i (float): Initial density perturbation.
    - r_min (float): Minimum radius.
    - softlen (float): Softening length.
    - safety_factor (float): Safety factor for timestep.
    - thickness_coef (float): Thickness coefficient.
    - j_coef (float): Angular momentum coefficient.
    - outfile (str, optional): Output file path.

    Returns:
    - dict: Configuration dictionary.
    """
    if outfile is None:
        outfile = f"runs/sim_N{N}_rmax{r_max}_delta{delta_i}_rmin{r_min}_dtmin{dt_min}.h5"
    else:
        outfile = outfile

    t_i = 1
    G = 1
    rho_H = 1/(6*np.pi*G*t_i**2)
    H = 2/(3*t_i)
    ss = SelfSimilarSolution(t_i, G, rho_H)

    def compute_r0_min(r_min, r_max, N):
        """
        Compute r0_min such that the spacing dR in np.linspace(r0_min, r_max, N)
        satisfies dR = r0_min - r_min. That is, the inner edge of the innermost shell is at r_min.
        
        Parameters:
        - r_min (float): The minimum radius.
        - r_max (float): The maximum radius.
        - N (int): Number of points.
        
        Returns:
        - float: The computed r0_min.
        """
        return (r_max + r_min * (N - 1)) / N

    def make_cycloid_config(r_min=1e-2, r_max=1, N=10, t_max=5000, dt_min=1e-9, rho_bar=rho_H, delta_i=1e-2, stepper_strategy='velocity_verlet', timescale_strategy='dyn', save_strategy='vflip', safety_factor=1e-4, reflect_bool=False, save_dt=1e-2, j_coeff=0, softlen=0, thickness_coef=0, irs='r0min_start_equal', mes='overlap_inclusive', pressure_strategy='zero', polytropic_index=-1, polytropic_coef=0, accel_strategy='soft_all', r_th=None, no_pm=False, outfile=None):
        if r_th is None:
            r_th = r_max
        m_pert = rho_bar * delta_i * 4/3 * np.pi * min(r_max, r_th)**3
        if no_pm:
            point_mass = 0
            shell_volume_strategy = 'keep_edges'
        else:
            point_mass = 4/3 * np.pi * r_min**3 * rho_bar * (1+delta_i)
            shell_volume_strategy = 'inner_not_zero'

        r0_min = compute_r0_min(r_min=r_min, r_max=r_max, N=N)
        r_min_strategy = 'reflect' if reflect_bool else 'absorb'
        if j_coeff > 0:
            ang_mom_strategy, r_ta_strategy = 'const', 'r_ta_cycloid'
        else:
            ang_mom_strategy, r_ta_strategy = 'zero', 'r_ta_cycloid'
        return {
            'point_mass': point_mass,
            'initial_radius_strategy': irs,
            'density_strategy': 'background_plus_tophat2',
            'delta': delta_i,
            'tophat_radius': 1,
            'r0_min': r0_min,
            'rho_bar': rho_H,
            't_max': t_max,
            'dt_min': dt_min,
            'N': N,
            'r_min': r_min,
            'r_min_strategy': r_min_strategy,
            'H': H,
            'r_max': r_max,
            'm_pert': m_pert, 
            'stepper_strategy': stepper_strategy,
            'timescale_strategy': timescale_strategy,
            'save_strategy': save_strategy,
            'safety_factor': safety_factor,
            'save_dt': save_dt,
            'j_coef': j_coeff,
            'ang_mom_strategy': ang_mom_strategy,
            'r_ta_strategy': r_ta_strategy,
            'save_strategy': 'vflip',
            'softlen': softlen,
            'thickness_coef': thickness_coef,
            'm_enc_strategy': mes,
            'pressure_strategy': pressure_strategy,
            'polytropic_index': polytropic_index,
            'polytropic_coef': polytropic_coef,
            'accel_strategy': accel_strategy,
            'shell_volume_strategy': shell_volume_strategy,
            'outfile': outfile
        }
    
    t_max = 10
    return make_cycloid_config(N=N, delta_i=delta_i, reflect_bool=True, t_max=t_max, r_min=r_min, timescale_strategy='dyn_thickness', dt_min=dt_min, safety_factor=1e-4, irs='equal_mass', no_pm=True, r_max=1.5, mes='neighbor', r_th=1, outfile=outfile)

def main():
    # Ensure the configs directory exists
    os.makedirs("configs", exist_ok=True)

    # Define the parameter values to vary
    N_values = [10, 100, 1000, 10_000]
    r_max_values = [1.5]
    delta_i_values = [1]
    r_min_values = [1e-2]
    dt_min_values = [1e-6, 1e-9, 1e-12]

    # Generate all combinations of parameters
    param_combinations = list(product(N_values, r_max_values, delta_i_values, r_min_values, dt_min_values))

    configs = []

    import time
    unix_time = int(time.time())
    for combo in param_combinations:
        N, r_max, delta_i, r_min, dt_min = combo
        outfile = f"runs/{unix_time}_N{N}_rmax{r_max}_delta{delta_i}_rmin{r_min}_dtmin{dt_min}.h5"
        config = generate_config(N=N, r_max=r_max, delta_i=delta_i, r_min=r_min, dt_min=dt_min, outfile=outfile)
        configs.append(config)

    print(f"Total configurations to generate: {len(configs)}")

    # Save configurations
    for i, config in enumerate(configs):
        filename = f"config_{i+1}.json"
        with open(os.path.join("configs", filename), "w") as f:
            json.dump(config, f, indent=2)

    print(f"Generated {len(configs)} configuration files.")

if __name__ == "__main__":
    main()
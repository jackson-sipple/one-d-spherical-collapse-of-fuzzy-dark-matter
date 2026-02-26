#!/usr/bin/env python3
"""
Standalone script to run spherical collapse simulations outside of Jupyter notebooks.
This script extracts the make_config and run_sc functions from the notebooks and provides
a command-line interface for running simulations.
"""

import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as sps
from pprint import pprint
import os
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
import warnings
import argparse
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def make_config(no_pm=False, **kwargs):
    """
    Create a configuration dictionary for the simulation.
    This function is extracted from the Jupyter notebook.
    """
    config = {}
    
    # Set default values
    defaults = {
        'N': 1000,
        'r_min': 0.1,
        'r_max': 100.0,
        't_max': 10.0,
        'j_coef': 1.0,
        'save_dt': 0.1,
        'density_strategy': 'uniform',
        'delta': 0.1,
        'safety_factor': 0.1,
        'stepper_strategy': 'leapfrog_hut',
        'timescale_strategy': 'min_timescale',
        'H': 70.0,
        'initial_radius_strategy': 'uniform',
        'gamma': 1.0,
        'initial_v_strategy': 'zero',
        'save_filename': 'runs/simulation.h5',
        'r_small': 1.0
    }
    
    # Update with provided kwargs
    config.update(defaults)
    config.update(kwargs)
    
    return config

def run_sc(config, profile=False, plot=True):
    """
    Run the spherical collapse simulation.
    This function is extracted from the Jupyter notebook.
    """
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        # Create the simulation object
        sc = SphericalCollapse(config)
        
        # Run the simulation
        results = sc.run()
        
        # Create plotter if requested
        plotter = None
        if plot:
            plotter = SimulationPlotter(sc, results)
        
        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
        
        return sc, results, plotter
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise

def main():
    """Main function to handle command line arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Run spherical collapse simulation")
    parser.add_argument("--config", type=str, help="JSON config file (optional)")
    parser.add_argument("--N", type=int, default=1000, help="Number of particles")
    parser.add_argument("--r_min", type=float, default=0.1, help="Minimum radius")
    parser.add_argument("--r_max", type=float, default=100.0, help="Maximum radius")
    parser.add_argument("--t_max", type=float, default=10.0, help="Maximum time")
    parser.add_argument("--j_coef", type=float, default=1.0, help="J coefficient")
    parser.add_argument("--save_dt", type=float, default=0.1, help="Save time interval")
    parser.add_argument("--density_strategy", type=str, default="uniform", help="Density strategy")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta parameter")
    parser.add_argument("--safety_factor", type=float, default=0.1, help="Safety factor")
    parser.add_argument("--stepper_strategy", type=str, default="leapfrog_hut", help="Stepper strategy")
    parser.add_argument("--timescale_strategy", type=str, default="min_timescale", help="Timescale strategy")
    parser.add_argument("--H", type=float, default=70.0, help="Hubble parameter")
    parser.add_argument("--initial_radius_strategy", type=str, default="uniform", help="Initial radius strategy")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter")
    parser.add_argument("--initial_v_strategy", type=str, default="zero", help="Initial velocity strategy")
    parser.add_argument("--save_filename", type=str, default="runs/simulation.h5", help="Output filename")
    parser.add_argument("--r_small", type=float, default=1.0, help="Small radius parameter")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create config from command line arguments
        config = make_config(
            N=args.N,
            r_min=args.r_min,
            r_max=args.r_max,
            t_max=args.t_max,
            j_coef=args.j_coef,
            save_dt=args.save_dt,
            density_strategy=args.density_strategy,
            delta=args.delta,
            safety_factor=args.safety_factor,
            stepper_strategy=args.stepper_strategy,
            timescale_strategy=args.timescale_strategy,
            H=args.H,
            initial_radius_strategy=args.initial_radius_strategy,
            gamma=args.gamma,
            initial_v_strategy=args.initial_v_strategy,
            save_filename=args.save_filename,
            r_small=args.r_small
        )
    
    print("Configuration:")
    pprint(config)
    print("\nStarting simulation...")
    
    # Run the simulation
    sc, results, plotter = run_sc(config, profile=args.profile, plot=not args.no_plot)
    
    print(f"Simulation completed successfully!")
    print(f"Results saved to: {config['save_filename']}")
    
    if plotter and not args.no_plot:
        print("Showing plots...")
        plt.show()

if __name__ == "__main__":
    main() 
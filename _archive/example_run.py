#!/usr/bin/env python3
"""
Example script showing how to run a simulation with the same parameters as in the notebook.
This replicates the line: 
config = make_config(N=N, r_min=r_min, r_max=r_max, t_max=t_max, j_coef=j_coef, save_dt=save_dt, density_strategy=density_strategy, delta=delta, safety_factor=safety_factor, stepper_strategy='leapfrog_hut',  timescale_strategy=tss, H=H0, initial_radius_strategy=irs, gamma=gamma, initial_v_strategy=ivs, save_filename='runs/723-2.h5', r_small=1)
sc, results, plotter = run_sc(config, profile=False, plot=True)
"""

import sys
import os

# Add the current directory to the path so we can import the standalone script
sys.path.insert(0, os.path.dirname(__file__))

from run_simulation_standalone import make_config, run_sc

def main():
    # Define your parameters (you'll need to set these to the actual values from your notebook)
    N = 1000  # Replace with your actual value
    r_min = 0.1  # Replace with your actual value
    r_max = 100.0  # Replace with your actual value
    t_max = 10.0  # Replace with your actual value
    j_coef = 1.0  # Replace with your actual value
    save_dt = 0.1  # Replace with your actual value
    density_strategy = 'uniform'  # Replace with your actual value
    delta = 0.1  # Replace with your actual value
    safety_factor = 0.1  # Replace with your actual value
    tss = 'min_timescale'  # Replace with your actual value
    H0 = 70.0  # Replace with your actual value
    irs = 'dr_start_equal'  # Replace with your actual value
    gamma = 1.0  # Replace with your actual value
    ivs = 'zero'  # Replace with your actual value
    
    # Create the configuration (exactly as in your notebook)
    config = make_config(
        save_filename='runs/STANDALONE.h5', 
    )
    
    # Run the simulation (exactly as in your notebook)
    sc, results, plotter = run_sc(config, profile=False, plot=True)
    
    print("Simulation completed successfully!")
    print(f"Results saved to: {config['save_filename']}")

if __name__ == "__main__":
    main() 
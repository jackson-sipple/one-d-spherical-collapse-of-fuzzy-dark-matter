import pytest
import numpy as np
from collapse import SphericalCollapse
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt
import warnings

@pytest.fixture
def base_config():
    return {
        "G": 1,
        "N": 1,
        "r_max": 1,
        "r_min": 0,
        "m_pert": 1,
        "point_mass": 0,
        "j_coef": 1,
        "ang_mom_strategy": "const",
        "safety_factor": 1e-5,
        'dt_min': 1e-16,
        'H': 0,
        "stepper_strategy": "beeman",
        "energy_strategy": "kin_grav_rot",
        "timescale_strategy": "dyn_vel",
        "thickness_strategy": "const",
        "t_max": -1,
        "save_dt": 1e-5,
        "save_strategy": "vflip",
    }


import h5py
import numpy as np
import logging
import os
from numba import njit

logger = logging.getLogger(__name__)

def save_to_hdf5(simulation, filename):
    """
    Save simulation data and parameters to an HDF5 file.
    
    Args:
    simulation (SphericalCollapse): The simulation object to save
    filename (str): The name of the output HDF5 file
    """
    # Only create directory if filename contains a directory path
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with h5py.File(filename, 'w') as hf:
        # Save parameters as attributes in the root group
        for key, value in simulation.get_parameters_dict().items():
            if isinstance(value, (int, float, str, bool, np.number)):
                hf.attrs[key] = value
            else:
                hf.attrs[key] = str(value)

        # Create a group for snapshots
        snapshots_group = hf.create_group('snapshots')

        # Prepare data for saving
        results = {key: [] for key in simulation.snapshots[0].keys()}
        for snapshot in simulation.snapshots:
            for key, value in snapshot.items():
                results[key].append(value)

        # Save snapshot data
        for key, value_list in results.items():
            if not value_list or all(v is None for v in value_list):
                logger.warning(f"Skipping {key} because all values are None")
                continue

            try:
                if np.isscalar(value_list[0]):
                    dataset = snapshots_group.create_dataset(key, data=np.array(value_list))
                else:
                    dataset = snapshots_group.create_dataset(key, data=np.array(value_list))
            except Exception as e:
                logger.error(f"Error creating dataset for {key}: {e}")
                continue

            logger.debug(f"Saved dataset {key} with shape {dataset.shape}")

    logger.info(f"Saved simulation data and parameters to {filename}")

def load_simulation_data(filename):
    """
    Load simulation data and parameters from an HDF5 file.
    
    Args:
    filename (str): The name of the HDF5 file to load
    
    Returns:
    tuple: (params, snapshots) where params is a dict of simulation parameters
           and snapshots is a dict of time series data
    """
    with h5py.File(filename, 'r') as hf:
        # Load parameters
        params = dict(hf.attrs)
        
        # Load snapshot data
        snapshots = {}
        for key in hf['snapshots']:
            snapshots[key] = np.array(hf['snapshots'][key])
    
    return params, snapshots

def kepler_e(menc, r, v, j):
    E = 0.5*v**2 - menc/r + j**2/(2*r**2)
    return np.sqrt(1 + 2*E*j**2/menc**2)

@njit
def fd_weights(x, x0, M):
    """
    Compute finite-difference weights for derivatives up to order M
    at x0, given nodes x (length N).
    Returns an (N x (M+1)) array w where w[j, m] is weight for f^(m)(x0).
    Uses Fornberg's algorithm (1988).
    """
    N = len(x)
    w = np.zeros((N, M+1))
    c = np.zeros((N, M+1))
    c1 = 1.0
    c4 = x[0] - x0
    c[0,0] = 1.0
    for i in range(1, N):
        mn = min(i, M)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(0, i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i-1:
                for k in range(mn, 0, -1):
                    c[i,k] = c1*(k*c[i-1,k-1] - c5*c[i-1,k]) / c2
                c[i,0] = -c1*c5*c[i-1,0] / c2
            for k in range(mn, 0, -1):
                c[j,k] = (c4*c[j,k] - k*c[j,k-1]) / c3
            c[j,0] = c4*c[j,0] / c3
        c1 = c2
    return c  # c[j, m] = weight for f^(m)(x0) from f(x[j])

@njit
def derivative_nonuniform(x, y, m=1, stencil_size=5):
    """
    Compute the m-th derivative of y at each x_i (non-uniform grid)
    using Fornberg weights on a sliding stencil of given size.
    At boundaries, uses one-sided stencils of same size.
    
    x : array of coordinates, length N
    y : array of function values f(x), length N
    m : derivative order (1,2,3,…)
    stencil_size : number of points in each stencil (>= m+1)
    """
    N = len(x)
    dy = np.zeros_like(y)
    s = stencil_size
    half = s // 2

    for i in range(N):
        # choose stencil window [i0 .. i1]
        if i < half:
            i0, i1 = 0, s
        elif i > N-1-half:
            i0, i1 = N-s-1, N-1
        else:
            i0, i1 = i-half, i+half+1
        xs = x[i0:i1+1]
        ys = y[i0:i1+1]
        # compute weights for m-th derivative at x[i]
        w = fd_weights(xs, x[i], m)
        # weights are in column m
        dy[i] = np.dot(w[:, m], ys)
    return dy
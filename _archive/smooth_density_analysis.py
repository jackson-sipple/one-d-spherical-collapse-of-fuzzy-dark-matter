import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def smooth_density_analysis(rvals, normalized_densityvals, rho_H, tval):
    """
    Analyze smoothed versions of normalized density values to understand power law behavior.
    
    Parameters:
    -----------
    rvals : array
        Radial values
    normalized_densityvals : array
        Normalized density values
    rho_H : float
        Hubble density for normalization
    tval : float
        Time value
    """
    
    # Method 1: Gaussian smoothing
    sigma = 2.0  # Adjust this parameter to control smoothing strength
    smoothed_density_gaussian = ndimage.gaussian_filter1d(normalized_densityvals, sigma=sigma)
    
    # Method 2: Savitzky-Golay filter (good for preserving features)
    window_length = min(21, len(normalized_densityvals) // 4)  # Must be odd and less than data length
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        window_length = 3
    smoothed_density_savgol = savgol_filter(normalized_densityvals, window_length=window_length, polyorder=3)
    
    # Method 3: Moving average
    window_size = 5
    smoothed_density_ma = np.convolve(normalized_densityvals, np.ones(window_size)/window_size, mode='same')
    
    # Method 4: Log-space smoothing (often better for power law analysis)
    log_density = np.log10(normalized_densityvals)
    log_r = np.log10(rvals)
    smoothed_log_density = ndimage.gaussian_filter1d(log_density, sigma=sigma)
    smoothed_density_log = 10**smoothed_log_density
    
    # Plot all smoothing methods
    plt.figure(figsize=(12, 8))
    
    # Original data
    plt.plot(rvals, normalized_densityvals, 'o', alpha=0.5, label='Original data', markersize=3)
    
    # Smoothed versions
    plt.plot(rvals, smoothed_density_gaussian, 'r-', linewidth=2, label='Gaussian smoothing')
    plt.plot(rvals, smoothed_density_savgol, 'g-', linewidth=2, label='Savitzky-Golay')
    plt.plot(rvals, smoothed_density_ma, 'b-', linewidth=2, label='Moving average')
    plt.plot(rvals, smoothed_density_log, 'm-', linewidth=2, label='Log-space smoothing')
    
    # Power law fits for comparison
    plt.plot(rvals, 1e3*(rvals/2e0)**(-8/7), 'k--', label=r'$\propto r^{-8/7}$')
    plt.plot(rvals, 1e3*(rvals/2e0)**(-2), 'k:', label=r'$\propto r^{-2}$')
    
    plt.legend()
    plt.xlabel('r')
    plt.ylabel(r'$\rho/\rho_H$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Normalized Density with Smoothing Methods (t = {tval:.3f})')
    plt.grid(True, alpha=0.3)
    
    return {
        'gaussian': smoothed_density_gaussian,
        'savgol': smoothed_density_savgol,
        'moving_avg': smoothed_density_ma,
        'log_space': smoothed_density_log
    }

def calculate_power_law_exponent(r, density, window=5):
    """
    Calculate local power law exponent using finite differences.
    
    Parameters:
    -----------
    r : array
        Radial values
    density : array
        Density values
    window : int
        Window size for local fitting
        
    Returns:
    --------
    exponents : array
        Local power law exponents
    """
    if len(r) < window:
        return np.full_like(r, np.nan)
    
    exponents = np.full_like(r, np.nan)
    for i in range(window//2, len(r) - window//2):
        r_local = r[i-window//2:i+window//2+1]
        density_local = density[i-window//2:i+window//2+1]
        
        # Fit log(density) = a * log(r) + b
        log_r_local = np.log10(r_local)
        log_density_local = np.log10(density_local)
        
        # Simple linear fit
        A = np.vstack([log_r_local, np.ones(len(log_r_local))]).T
        slope, intercept = np.linalg.lstsq(A, log_density_local, rcond=None)[0]
        exponents[i] = slope
    
    return exponents

def plot_power_law_exponents(rvals, smoothed_densities):
    """
    Plot power law exponents from different smoothing methods.
    
    Parameters:
    -----------
    rvals : array
        Radial values
    smoothed_densities : dict
        Dictionary containing smoothed density arrays
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate power law exponents for different smoothing methods
    exponent_gaussian = calculate_power_law_exponent(rvals, smoothed_densities['gaussian'])
    exponent_savgol = calculate_power_law_exponent(rvals, smoothed_densities['savgol'])
    exponent_log = calculate_power_law_exponent(rvals, smoothed_densities['log_space'])
    
    plt.plot(rvals, exponent_gaussian, 'r-', linewidth=2, label='Gaussian smoothing')
    plt.plot(rvals, exponent_savgol, 'g-', linewidth=2, label='Savitzky-Golay')
    plt.plot(rvals, exponent_log, 'm-', linewidth=2, label='Log-space smoothing')
    
    # Reference lines for expected power laws
    plt.axhline(y=-8/7, color='k', linestyle='--', alpha=0.5, label=r'$-8/7$ (expected)')
    plt.axhline(y=-2, color='k', linestyle=':', alpha=0.5, label=r'$-2$ (expected)')
    
    plt.xlabel('r')
    plt.ylabel('Power Law Exponent')
    plt.xscale('log')
    plt.title('Local Power Law Exponent from Smoothed Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Print summary statistics
    print("Power law exponent statistics:")
    print(f"Gaussian smoothing: mean = {np.nanmean(exponent_gaussian):.3f}, std = {np.nanstd(exponent_gaussian):.3f}")
    print(f"Savitzky-Golay: mean = {np.nanmean(exponent_savgol):.3f}, std = {np.nanstd(exponent_savgol):.3f}")
    print(f"Log-space smoothing: mean = {np.nanmean(exponent_log):.3f}, std = {np.nanstd(exponent_log):.3f}")
    
    return {
        'gaussian': exponent_gaussian,
        'savgol': exponent_savgol,
        'log_space': exponent_log
    }

def fit_global_power_law(rvals, density_vals, r_range=None):
    """
    Fit a global power law to the data over a specified range.
    
    Parameters:
    -----------
    rvals : array
        Radial values
    density_vals : array
        Density values
    r_range : tuple, optional
        (r_min, r_max) range to fit over
        
    Returns:
    --------
    slope : float
        Power law exponent
    intercept : float
        Intercept in log space
    r_squared : float
        R-squared value of the fit
    """
    if r_range is not None:
        r_min, r_max = r_range
        mask = (rvals >= r_min) & (rvals <= r_max)
        r_fit = rvals[mask]
        density_fit = density_vals[mask]
    else:
        r_fit = rvals
        density_fit = density_vals
    
    log_r = np.log10(r_fit)
    log_density = np.log10(density_fit)
    
    # Fit log(density) = a * log(r) + b
    A = np.vstack([log_r, np.ones(len(log_r))]).T
    slope, intercept = np.linalg.lstsq(A, log_density, rcond=None)[0]
    
    # Calculate R-squared
    y_pred = slope * log_r + intercept
    ss_res = np.sum((log_density - y_pred) ** 2)
    ss_tot = np.sum((log_density - np.mean(log_density)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared

# Example usage (uncomment and modify as needed):
"""
# After your existing code, add:
smoothed_densities = smooth_density_analysis(rvals, normalized_densityvals, rho_H, tval)
exponents = plot_power_law_exponents(rvals, smoothed_densities)

# Fit global power law to smoothed data
slope, intercept, r_sq = fit_global_power_law(rvals, smoothed_densities['log_space'])
print(f"Global power law fit: ρ ∝ r^{slope:.3f} (R² = {r_sq:.3f})")

# Plot the global fit
plt.figure(figsize=(10, 6))
plt.plot(rvals, normalized_densityvals, 'o', alpha=0.5, label='Original data', markersize=3)
plt.plot(rvals, smoothed_densities['log_space'], 'r-', linewidth=2, label='Log-space smoothing')
plt.plot(rvals, 10**intercept * rvals**slope, 'k--', linewidth=2, 
         label=f'Global fit: ρ ∝ r^{slope:.3f}')
plt.xlabel('r')
plt.ylabel(r'$\rho/\rho_H$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f'Global Power Law Fit (R² = {r_sq:.3f})')
""" 
# Add this code cell after your existing plotting code to analyze smoothed density values

# Import required libraries
from scipy import ndimage
from scipy.signal import savgol_filter

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

# Calculate local power law exponents
def calculate_power_law_exponent(r, density, window=5):
    """Calculate local power law exponent using finite differences"""
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

# Calculate power law exponents for different smoothing methods
exponent_gaussian = calculate_power_law_exponent(rvals, smoothed_density_gaussian)
exponent_savgol = calculate_power_law_exponent(rvals, smoothed_density_savgol)
exponent_log = calculate_power_law_exponent(rvals, smoothed_density_log)

# Plot power law exponents
plt.figure(figsize=(12, 6))

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

# Fit global power law to the best smoothed data (log-space smoothing)
log_r_fit = np.log10(rvals)
log_density_fit = np.log10(smoothed_density_log)

# Fit log(density) = a * log(r) + b
A = np.vstack([log_r_fit, np.ones(len(log_r_fit))]).T
slope, intercept = np.linalg.lstsq(A, log_density_fit, rcond=None)[0]

# Calculate R-squared
y_pred = slope * log_r_fit + intercept
ss_res = np.sum((log_density_fit - y_pred) ** 2)
ss_tot = np.sum((log_density_fit - np.mean(log_density_fit)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nGlobal power law fit: ρ ∝ r^{slope:.3f} (R² = {r_squared:.3f})")

# Plot the global fit
plt.figure(figsize=(10, 6))
plt.plot(rvals, normalized_densityvals, 'o', alpha=0.5, label='Original data', markersize=3)
plt.plot(rvals, smoothed_density_log, 'r-', linewidth=2, label='Log-space smoothing')
plt.plot(rvals, 10**intercept * rvals**slope, 'k--', linewidth=2, 
         label=f'Global fit: ρ ∝ r^{slope:.3f}')
plt.xlabel('r')
plt.ylabel(r'$\rho/\rho_H$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f'Global Power Law Fit (R² = {r_squared:.3f})') 
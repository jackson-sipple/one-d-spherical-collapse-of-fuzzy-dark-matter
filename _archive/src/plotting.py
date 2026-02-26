import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy.signal import argrelmax
from functools import wraps
from matplotlib.colorbar import ColorbarBase

@dataclass
class PlotConfig:
    """Configuration for a single plot."""
    property_name: str
    ylabel: str = None
    yscale: str = 'linear'
    ylim: Tuple[float, float] = None
    xlim: Tuple[float, float] = None
    title: str = None
    shell_indices: List[int] = None  # If None, plot all shells
    num_shells: Optional[int] = None  # If set, evenly sample this many shells
    plot_type: str = 'line'  # 'line', 'scatter', etc.
    
    def __post_init__(self):
        """Set default values based on property name if not specified."""
        if not self.ylabel:
            self.ylabel = self.property_name.replace('_', ' ').title()
        if not self.title:
            self.title = f'{self.ylabel} vs Time'
        
        # Ensure shell_indices and num_shells aren't both set
        if self.shell_indices is not None and self.num_shells is not None:
            raise ValueError("Cannot specify both shell_indices and num_shells")

# Decorator to add colorbar to plotting functions
def with_shell_colorbar(plot_method):
    @wraps(plot_method)
    def wrapper(self, *args, use_color_gradient=True, **kwargs):
        result = plot_method(self, *args, use_color_gradient=use_color_gradient, **kwargs)
        
        # Get fig and axes - either returned by the method or created by it
        if result is None:
            # If the method doesn't return anything, try to get the current figure
            fig = plt.gcf()
            axes = plt.gca()
        elif isinstance(result, tuple) and len(result) == 2:
            # If the method returns (fig, axes)
            fig, axes = result
        else:
            # If the method returns just the figure
            fig = result
            axes = fig.gca()
            
        # Add colorbar if color gradient is used
        if use_color_gradient and hasattr(self, 'cmap') and hasattr(self, 'norm'):
            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            
            # Create colorbar axis
            if isinstance(axes, np.ndarray):
                # For subplot grids
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            else:
                # For single plot
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                
            # Create colorbar
            cbar = ColorbarBase(cbar_ax, cmap=self.cmap, norm=self.norm,
                               orientation='vertical')
            cbar.set_label('Initial Radius')
        
        return fig, axes
    return wrapper

class SimulationPlotter:
    """A flexible plotting system for simulation results."""
    
    def __init__(self, results: Dict, sc=None):
        """
        Initialize with simulation results.
        
        Args:
            results: Dictionary containing simulation data
        """
        self.sc = sc
        self.results = results
        self.time = results['t']
        
        # Common plot settings
        self.default_figsize = (10, 6)
        self.grid = True
        
        # Create a colormap for shells based on initial radius
        # Get number of shells
        n_shells = self.results['r'].shape[1]
        
        # Get initial radii for color mapping
        self.initial_radii = self.results['r'][0, :]
        
        # Create normalized colormap based on initial radius
        self.norm = colors.Normalize(vmin=np.min(self.initial_radii), vmax=np.max(self.initial_radii))
        self.cmap = cm.viridis
        
        # Store colors for each shell
        self.shell_colors = [self.cmap(self.norm(r)) for r in self.initial_radii]
        
    def create_subplot(self, ax: plt.Axes, config: PlotConfig) -> None:
        """Create a single subplot based on the configuration."""
        if config.property_name not in self.results:
            raise ValueError(f"Property '{config.property_name}' not found in results")
            
        data = self.results[config.property_name]
        
        # Check for the special case: normalized radius and t_ta available.
        special_norm = (config.property_name == 'r_normalized' and 
                        't_ta' in self.results)
        
        # For non-special cases, we use the common time array.
        if not special_norm:
            x_data = self.time
            xlabel = 'Time'
        else:
            xlabel = 't/t_ta'
        
        if data.ndim == 1:
            # If data is a scalar property, try to normalize time if possible.
            if special_norm:
                # Use the calculated t_ta instead of self.sc.t_ta
                t_ta_val = self.results['t_ta'] if np.isscalar(self.results['t_ta']) else self.results['t_ta'][0]
                x_data = (self.time + 1) / t_ta_val
            else:
                x_data = self.time
            ax.plot(x_data, data)
        else:
            # Determine which shells to plot.
            total_shells = data.shape[1]
            if config.shell_indices is not None:
                indices = config.shell_indices
            elif config.num_shells is not None:
                indices = np.linspace(0, total_shells - 1, min(total_shells, config.num_shells), dtype=int)
            else:
                indices = range(total_shells)
            
            # Loop through each shell.
            for idx in indices:
                # Only use shell_colors if data columns match number of shells and idx is valid
                if (
                    hasattr(self, 'shell_colors') and
                    data.shape[1] == len(self.shell_colors) and
                    0 <= idx < len(self.shell_colors)
                ):
                    color = self.shell_colors[idx]
                else:
                    color = None
                if special_norm:
                    # Use the calculated t_ta instead of self.sc.t_ta
                    t_ta_val = self.results['t_ta'][idx] if not np.isscalar(self.results['t_ta']) else self.results['t_ta']
                    if t_ta_val == 0:
                        continue
                    try:
                        x_data_shell = (self.time + 1) / t_ta_val
                        ax.plot(x_data_shell, data[:, idx], label=f'Shell {idx}', color=color)
                    except:
                        pass
                else:
                    ax.plot(self.time, data[:, idx], label=f'Shell {idx}', color=color)
                
                # Only show legend if not plotting all shells.
                if len(indices) < total_shells:
                    ax.legend()
        
        # Apply plot settings.
        ax.set_xlabel(xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        ax.set_yscale(config.yscale)
        ax.grid(self.grid)
        
        # Set x limits and adjust y limits to visible data if xlim is provided.
        if config.xlim:
            ax.set_xlim(config.xlim)
            xmin, xmax = config.xlim
            if data.ndim == 1:
                if special_norm:
                    t_ta_val = self.sc.t_ta if np.isscalar(self.sc.t_ta) else self.sc.t_ta[0]
                    x_data = (self.time+1) / t_ta_val
                else:
                    x_data = self.time
                visible_mask = (x_data >= xmin) & (x_data <= xmax)
                visible_data = data[visible_mask]
            else:
                # For 2D data, this is less straightforward with different x data per shell.
                # Here we use the first shell's x-data as a proxy for determining the limits.
                if special_norm:
                    t_ta_val = self.sc.t_ta if np.isscalar(self.sc.t_ta) else self.sc.t_ta[indices[0]]
                    x_data = (self.time+1) / t_ta_val
                else:
                    x_data = self.time
                visible_mask = (x_data >= xmin) & (x_data <= xmax)
                visible_data = data[visible_mask, :]
            ymin, ymax = np.nanmin(visible_data), np.nanmax(visible_data)
            
            # Apply padding.
            if config.yscale == 'log':
                positive_data = visible_data[visible_data > 0]
                if len(positive_data) > 0:
                    ymin = np.nanmin(positive_data)
                    ymax = np.nanmax(positive_data)
                    padding_factor = 1.1
                    ax.set_ylim(ymin / padding_factor, ymax * padding_factor)
            else:
                padding = (ymax - ymin) * 0.05
                ax.set_ylim(ymin - padding, ymax + padding)
        
        # Override automatic y limits if explicitly provided.
        if config.ylim:
            ax.set_ylim(config.ylim)
            
            # Add custom tick marks for specific ylim ranges
            if config.ylim == [0, 2]:
                # Set tick marks every 0.2 for ylim [0, 2]
                y_ticks = np.arange(0, 2.1, 0.2)
                ax.set_yticks(y_ticks)
    
    def plot_multiple(self, configs: List[PlotConfig], 
                     layout: Tuple[int, int] = None,
                     figsize: Tuple[int, int] = None,
                     tight_layout: bool = True) -> None:
        """
        Create multiple subplots in a single figure.
        
        Args:
            configs: List of PlotConfig objects
            layout: Tuple of (rows, cols) for subplot layout. If None, calculated automatically
            figsize: Figure size in inches. If None, calculated based on layout
            tight_layout: Whether to apply tight_layout to the figure
        """
        n_plots = len(configs)
        
        # Calculate layout if not provided
        if layout is None:
            cols = min(3, n_plots)  # Max 3 columns
            rows = (n_plots + cols - 1) // cols
            layout = (rows, cols)
            
        # Calculate figsize if not provided
        if figsize is None:
            figsize = (self.default_figsize[0] * layout[1],
                      self.default_figsize[1] * layout[0])
            
        fig, axes = plt.subplots(*layout, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flat
            
        # Create each subplot
        for ax, config in zip(axes, configs):
            self.create_subplot(ax, config)
            
        # Hide empty subplots
        for ax in axes[n_plots:]:
            ax.set_visible(False)
            
        if tight_layout:
            plt.tight_layout()
        
        plt.show()
    
    def plot_single(self, config: PlotConfig) -> None:
        """Create a single plot."""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        self.create_subplot(ax, config)
        plt.show()
        
    def plot_energy_components(self, shell_index: Optional[int] = None) -> None:
        """Plot energy components as subplots."""
        # Create a basic config with shell_index if specified
        base_config = PlotConfig('e_tot')
        if shell_index is not None:
            base_config.shell_indices = [shell_index]

        configs = []
        for prop, label in [('e_tot', 'Total Energy'),
                           ('e_k', 'Kinetic Energy'),
                           ('e_g', 'Gravitational Energy'),
                           ('e_r', 'Rotational Energy'),
                           ('e_p', 'Pressure Energy')]:
            config = PlotConfig(
                property_name=prop,
                ylabel=label,
                shell_indices=base_config.shell_indices,
                num_shells=base_config.num_shells,
                yscale=base_config.yscale,
                ylim=base_config.ylim,
                xlim=base_config.xlim,
                plot_type=base_config.plot_type
            )
            if shell_index is not None:
                config.title = f'{label} (Shell {shell_index})'
            configs.append(config)
        
        self.plot_multiple(configs, layout=(2, 2))

    def plot_radius_analysis(self, shell_config: Optional[PlotConfig] = None, use_color_gradient=True) -> None:
        """Plot radius analysis (Configuration 1)"""
        if shell_config is None:
            base_config = PlotConfig('r')
        else:
            base_config = PlotConfig(
                property_name='r',
                shell_indices=shell_config.shell_indices,
                num_shells=shell_config.num_shells,
                yscale=shell_config.yscale,
                ylim=shell_config.ylim,
                xlim=shell_config.xlim,
                plot_type=shell_config.plot_type
            )

        # Calculate R/R_ta for each shell
        r_data = self.results['r']
        
        # Find turnaround indices for each shell
        ta_indices = []
        for i in range(r_data.shape[1]):
            # Find indices of local maxima for this shell
            maxima_indices = argrelmax(r_data[:, i])[0]
            # If maxima found, use the first one; otherwise use None
            ta_idx = maxima_indices[0] if len(maxima_indices) > 0 else None
            ta_indices.append(ta_idx)
        
        # Process shells that have turnaround points
        valid_shells = [i for i in range(r_data.shape[1]) if ta_indices[i] is not None]
        
        if valid_shells:
            # Calculate r_ta and t_ta only for shells with turnaround points
            r_ta = np.zeros(r_data.shape[1])
            t_ta = np.zeros(r_data.shape[1])
            
            for i in valid_shells:
                r_ta[i] = r_data[ta_indices[i], i]
                t_ta[i] = self.time[ta_indices[i]]
            
            # Store the time at turnaround for each shell
            self.results['t_ta'] = t_ta
            
            # Calculate normalized radius (handle shells without turnaround)
            self.results['r_normalized'] = np.zeros_like(r_data)
            for i in valid_shells:
                self.results['r_normalized'][:, i] = r_data[:, i] / r_ta[i]
        else:
            # No shells have reached turnaround yet
            self.results['t_ta'] = np.zeros(r_data.shape[1])
            self.results['r_normalized'] = r_data.copy()  # Just use unnormalized radius
        
        # Calculate total relative energy change
        e_tot = np.sum(self.results['e_tot'], axis=1)
        e_rel_change = np.abs((e_tot - e_tot[0]) / e_tot[0])
        self.results['e_rel_change'] = e_rel_change

        configs = []
        
        # Create each config with the same base settings
        configs.append(PlotConfig(
            property_name='r',
            ylabel='Radius',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            yscale=base_config.yscale,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        configs.append(PlotConfig(
            property_name='r',
            ylabel='Log(Radius)',
            yscale='log',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        configs.append(PlotConfig(
            property_name='r_normalized',
            ylabel='R/R_ta',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            yscale='linear',
            ylim=[0, 2],
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        configs.append(PlotConfig(
            property_name='e_rel_change',
            ylabel='|ΔE/E|',
            yscale='log',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        self.plot_multiple(configs, layout=(2, 2))

    def plot_velocity_acceleration(self, shell_config: Optional[PlotConfig] = None) -> None:
        """Plot velocity and acceleration analysis (Configuration 2)"""
        if shell_config is None:
            base_config = PlotConfig('v')
        else:
            base_config = PlotConfig(
                property_name='v',
                shell_indices=shell_config.shell_indices,
                num_shells=shell_config.num_shells,
                yscale=shell_config.yscale,
                ylim=shell_config.ylim,
                xlim=shell_config.xlim,
                plot_type=shell_config.plot_type
            )
        
        # Calculate log absolute values
        v_data = self.results['v']
        a_data = self.results['a']
        self.results['log_abs_v'] = np.abs(v_data)
        self.results['log_abs_a'] = np.abs(a_data)

        configs = []
        
        configs.append(PlotConfig(
            property_name='v',
            ylabel='Velocity',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            yscale=base_config.yscale,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        configs.append(PlotConfig(
            property_name='log_abs_v',
            ylabel='Log|Velocity|',
            yscale='log',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        configs.append(PlotConfig(
            property_name='a',
            ylabel='Acceleration',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            yscale=base_config.yscale,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        configs.append(PlotConfig(
            property_name='log_abs_a',
            ylabel='Log|Acceleration|',
            yscale='log',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        ))
        
        self.plot_multiple(configs, layout=(2, 2))

    def plot_mass_density_timescales(self, shell_config: Optional[PlotConfig] = None) -> None:
        """Plot mass, density and timescales analysis (Configuration 3)"""
        if shell_config is None:
            base_config = PlotConfig('m_enc')
        else:
            base_config = PlotConfig(
                property_name='m_enc',
                shell_indices=shell_config.shell_indices,
                num_shells=shell_config.num_shells,
                yscale=shell_config.yscale,
                ylim=shell_config.ylim,
                xlim=shell_config.xlim,
                plot_type=shell_config.plot_type
            )

        # Calculate shell widths for each timestep
        r_data = self.results['r']
        shell_widths = np.zeros_like(r_data)
        
        # For each timestep
        for t in range(len(self.time)):
            r_t = r_data[t]
            # Sort radii at this timestep
            sorted_indices = np.argsort(r_t)
            sorted_r = r_t[sorted_indices]
            
            # Calculate widths
            widths = np.zeros_like(sorted_r)
            # For lowest shell, use distance to r_min
            widths[0] = sorted_r[0] - self.sc.r_min if self.sc else sorted_r[0]
            # For other shells, use distance to next lower shell
            widths[1:] = sorted_r[1:] - sorted_r[:-1]
            
            # Put widths back in original order
            shell_widths[t, sorted_indices] = widths
        
        # Store shell widths in results
        self.results['shell_widths'] = shell_widths

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot enclosed mass
        mass_config = PlotConfig(**vars(base_config))
        mass_config.property_name = 'm_enc'
        mass_config.ylabel = 'Enclosed Mass'
        self.create_subplot(axes[0, 0], mass_config)
        
        # Plot density
        density_config = PlotConfig(**vars(base_config))
        density_config.property_name = 'rho_r'
        density_config.ylabel = 'Density'
        density_config.yscale = 'log'
        self.create_subplot(axes[0, 1], density_config)
        
        # Plot timescales
        timescale_data = np.column_stack([self.results['dt']])  # Start with dt
        timescale_labels = ['dt']
        
        if self.sc is not None:
            # Add dt_min as a constant array
            dt_min_array = np.full_like(self.time, self.sc.dt_min)
            timescale_data = np.column_stack([timescale_data, dt_min_array])
            timescale_labels.append('dt_min')
            
        for key in self.results.keys():
            if key.startswith('t_') and key != 't' and key != 't_ta' and self.results[key][0] is not None:
                timescale_data = np.column_stack([timescale_data, self.results[key]])
                timescale_labels.append(key.replace('t_', ''))
        
        # Temporarily store the timescale data
        self.results['_timescales'] = timescale_data
        
        timescales_config = PlotConfig(
            property_name='_timescales',
            ylabel='Timescale',
            title='Simulation Timescales',
            yscale='log',
            xlim=base_config.xlim,
            ylim=base_config.ylim
        )
        
        self.create_subplot(axes[1, 0], timescales_config)
        axes[1, 0].legend(timescale_labels)
        
        # Plot shell widths
        widths_config = PlotConfig(
            property_name='shell_widths',
            ylabel='Shell Width',
            title='Shell Widths',
            yscale='log',
            shell_indices=base_config.shell_indices,
            num_shells=base_config.num_shells,
            ylim=base_config.ylim,
            xlim=base_config.xlim,
            plot_type=base_config.plot_type
        )
        self.create_subplot(axes[1, 1], widths_config)
        
        # Clean up temporary data
        del self.results['_timescales']
        del self.results['shell_widths']
        
        plt.tight_layout()
        plt.show()

    def plot_energy_analysis(self, shell_config: Optional[PlotConfig] = None) -> None:
        """Plot energy analysis (Configuration 4)"""
        if shell_config is None:
            base_config = PlotConfig('e_tot')
        else:
            base_config = PlotConfig(
                property_name='e_tot',
                shell_indices=shell_config.shell_indices,
                num_shells=shell_config.num_shells,
                yscale=shell_config.yscale,
                ylim=shell_config.ylim,
                xlim=shell_config.xlim,
                plot_type=shell_config.plot_type
            )
        
        # Calculate energy metrics
        e_tot = np.sum(self.results['e_tot'], axis=1)
        e_rel_change = np.abs((e_tot - e_tot[0]) / e_tot[0])
        self.results['e_rel_change'] = e_rel_change
        
        # Calculate summed energy components
        e_k_sum = np.sum(self.results['e_k'], axis=1)
        e_g_sum = np.sum(self.results['e_g'], axis=1)
        e_r_sum = np.sum(self.results.get('e_r', np.zeros_like(self.results['e_k'])), axis=1)
        try:
            e_p_sum = np.sum(self.results.get('e_p', np.zeros_like(self.results['e_k'])), axis=1)
        except:
            e_p_sum = np.zeros_like(e_k_sum)
        try:
            e_q_sum = np.sum(self.results.get('e_q', np.zeros_like(self.results['e_k'])), axis=1)
        except:
            e_q_sum = np.zeros_like(e_k_sum)
        e_tot_sum = e_k_sum + e_g_sum + e_r_sum + e_p_sum + e_q_sum
        
        # Store negative total energy per shell
        self.results['neg_e_tot'] = -1 * self.results['e_tot']
        
        # Store energy component sums
        self.results['energy_components'] = np.column_stack([e_k_sum, e_g_sum, e_r_sum, e_p_sum, e_q_sum, e_tot_sum])
        self.results['log_abs_energy_components'] = np.abs(self.results['energy_components'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot relative energy change
        self.create_subplot(axes[0, 0], 
                          PlotConfig(
                              property_name='e_rel_change',
                              ylabel='|ΔE/E|',
                              yscale='log',
                              shell_indices=base_config.shell_indices,
                              num_shells=base_config.num_shells,
                              ylim=base_config.ylim,
                              xlim=base_config.xlim,
                              plot_type=base_config.plot_type
                          ))
        
        # Plot log(-total energy) per shell
        self.create_subplot(axes[0, 1],
                          PlotConfig(
                              property_name='neg_e_tot',
                              ylabel='Log(-E_total)',
                              title='Log Total Energy per Shell',
                              yscale='log',
                              shell_indices=base_config.shell_indices,
                              num_shells=base_config.num_shells,
                              ylim=base_config.ylim,
                              xlim=base_config.xlim,
                              plot_type=base_config.plot_type
                          ))
        
        # Plot energy components sum
        self.create_subplot(axes[1, 0],
                          PlotConfig(
                              property_name='energy_components',
                              ylabel='Energy Sum',
                              title='Total Energy Components',
                              yscale=base_config.yscale,
                              ylim=base_config.ylim,
                              xlim=base_config.xlim,
                              plot_type=base_config.plot_type
                          ))
        axes[1, 0].legend(['Kinetic', 'Gravitational', 'Rotational', 'Pressure', 'Quantum', 'Total'])
        
        # Plot log|energy components sum|
        self.create_subplot(axes[1, 1],
                          PlotConfig(
                              property_name='log_abs_energy_components',
                              ylabel='Log|Energy Sum|',
                              title='Log Total Energy Components',
                              yscale='log',
                              ylim=base_config.ylim,
                              xlim=base_config.xlim,
                              plot_type=base_config.plot_type
                          ))
        axes[1, 1].legend(['Log|Kinetic|', 'Log|Gravitational|', 
                          'Log|Rotational|', 'Log|Pressure|', 'Log|Quantum|', 'Log|Total|'])
        
        plt.tight_layout()
        plt.show()
        
        # Clean up temporary data
        del self.results['neg_e_tot']
        del self.results['energy_components']
        del self.results['log_abs_energy_components']

    def plot_pressure_analysis(self, shell_config: Optional[PlotConfig] = None) -> None:
        """
        Plot analysis related to the pressure.
        This creates three subplots:
          (1) Density (rho_r)
          (2) Pressure
          (3) The pressure term in the acceleration: -1/rho * dpressure_drho * rho_prime

        It expects that the simulation results dictionary contains the following keys:
            'rho_r'         -> density (or shell density),
            'pressure'      -> pressure,
            'dpressure_drho'-> derivative of pressure with respect to density,
            'rho_prime'     -> derivative of density with respect to r.
        """
        # Check for required keys
        required_keys = ['rho_r', 'pressure', 'rho_prime', 'dpressure_drho']
        missing_keys = [key for key in required_keys if key not in self.results]
        if missing_keys:
            raise ValueError(f"Missing required keys in results for pressure analysis: {missing_keys}")

        # Compute the pressure term: -1/rho * dP/drho * rho_prime
        # Note: Using self.results ensures that if your simulation run updated these
        #       values via the pressure and drhodr strategies, they are available.

        # Create plot configurations
        density_config = PlotConfig(
             property_name='rho_r',
             ylabel='Density',
             title='Density vs Time',
             yscale='log'
        )
        pressure_config = PlotConfig(
             property_name='pressure',
             ylabel='Pressure',
             title='Pressure vs Time',
             yscale='log'
        )
        dP_drho_term_config = PlotConfig(
             property_name='dpressure_drho',
             ylabel='Pressure Term',
             title='dP/dρ vs Time',
             yscale='log'
        )
        rho_prime_config = PlotConfig(
             property_name='rho_prime',
             ylabel='dρ/dr',
             title='dρ/dr vs Time',
             yscale='log'
        )

        # If a shell configuration is provided, propagate shell information.
        if shell_config is not None:
            density_config.shell_indices = shell_config.shell_indices
            pressure_config.shell_indices = shell_config.shell_indices
            dP_drho_term_config.shell_indices = shell_config.shell_indices
            rho_prime_config.shell_indices = shell_config.shell_indices

        # Plot the three subplots in a vertical layout (3 rows, 1 col)
        self.plot_multiple([density_config, pressure_config, dP_drho_term_config, rho_prime_config], layout=(2, 2))


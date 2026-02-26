import numpy as np
from scipy.interpolate import interp1d

class SelfSimilarSolution:
    """
    Analytical solutions for self-similar collapse in spherical symmetry (Bertschinger 1985).
    
    This class provides methods to calculate various quantities in the self-similar
    solution of spherical collapse, including transformations between physical
    and self-similar coordinates.
    
    Attributes:
        t_i (float): Initial time
        G (float): Gravitational constant
        rho_H (float): Background density at initial time
        theta_vals (np.ndarray): Array of theta values for interpolation
        lambda_vals (np.ndarray): Array of lambda values for interpolation
        theta_lambda (callable): Interpolation function from lambda to theta
    """
    def __init__(self, t_i, G, rho_H):
        self.t_i = t_i
        self.G = G
        self.rho_H = rho_H
        self.theta_vals = np.linspace(1e-6, 2*np.pi, 1000)
        self.lambda_vals = self.lambda_theta(self.theta_vals)
        self.theta_lambda = interp1d(self.lambda_vals, self.theta_vals, bounds_error=False, fill_value=np.nan)

        # Create the inverse mapping (lambda -> theta)
        self.theta_lambda = interp1d(
            self.lambda_vals, 
            self.theta_vals, 
            bounds_error=False, 
            fill_value=np.nan
        )

    def delta_m(self, m_pert):
        """Calculate delta_m from perturbation mass."""
        return m_pert / (4*np.pi*self.rho_H)
    
    def lambda_theta(self, theta):
        """
        Calculate lambda(theta) in the self-similar solution.
        
        Args:
            theta (float or np.ndarray): Angle parameter
            
        Returns:
            float or np.ndarray: Lambda value(s)
        """
        eps = 1e-10  # Small epsilon to avoid division by zero
        return 0.5*(1-np.cos(theta))*((theta-np.sin(theta) + eps)/np.pi)**(-8/9)
    
    def r_ta(self, r_i, m_pert):
        """Calculate turnaround radius."""
        return r_i**4 / (3*self.delta_m(m_pert))
    
    def t_ta(self, r_i, m_pert, H):
        """Calculate turnaround time."""
        return np.pi*r_i**(9/2) / (6*np.sqrt(3)*H*self.delta_m(m_pert)**(3/2))
    
    def r_theta(self, r_i, m_pert, theta):
        """Calculate r(theta)."""
        return self.r_ta(r_i, m_pert)/2 * (1-np.cos(theta))
    
    def t_theta(self, r_i, m_pert, H, theta):
        """Calculate t(theta)."""
        return self.t_ta(r_i, m_pert, H)/np.pi * (theta-np.sin(theta))
    
    def theta_t(self, r_i, m_pert, H, t):
        """Calculate theta(t) using interpolation."""
        return interp1d(
            self.theta_vals, 
            self.t_theta(r_i, m_pert, H, self.theta_vals), 
            fill_value=0
        )(t)
    
    def v_lambda(self, lam):
        """Calculate velocity in self-similar coordinates."""
        theta = self.theta_lambda(lam)
        return lam * (np.sin(theta)*(theta-np.sin(theta)))/(1-np.cos(theta))**2
    
    def shifted_t(self, t):
        """Calculate shifted time."""
        return t + self.t_i
    
    def r_cta(self, t, m_pert, H):
        """Calculate characteristic radius."""
        return (1/3)*(6*np.sqrt(3)*H/np.pi)**(8/9) * self.delta_m(m_pert)**(1/3) * self.shifted_t(t)**(8/9)
    
    def lambda_rt(self, r, t, m_pert, H):
        """Calculate lambda from physical coordinates."""
        return r / self.r_cta(t, m_pert, H)
    
    def m_lambda(self, lam, H):
        """Calculate mass in self-similar coordinates."""
        theta = self.theta_lambda(lam)
        return lam**3 * (2/H**2) * (theta-np.sin(theta))**2/(1-np.cos(theta))**3
    
    def d_lambda(self, lam, H):
        """Calculate density in self-similar coordinates."""
        return self.m_lambda(lam, H)/lam**3 * 1/(4-(9/2)*self.v_lambda(lam)/lam)
    
    def rho_H_t(self, t):
        """Calculate background density at time t."""
        return self.rho_H * (1+t/self.t_i)**(-2)
    
    def m_from_results(self, t, m_enc, m_pert, H):
        """Convert enclosed mass from simulation to self-similar coordinates."""
        return m_enc / (4/3 * np.pi * self.rho_H_t(t) * self.r_cta(t, m_pert, H)**3)
    
    def d_from_results(self, t, m_shell, r_shell, r_prev_shell):
        """Convert density from simulation to self-similar coordinates."""
        vol_shell = 4/3 * np.pi * (r_shell**3 - r_prev_shell**3)
        return (m_shell/vol_shell) / self.rho_H_t(t)
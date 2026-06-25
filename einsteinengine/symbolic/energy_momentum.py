import symengine as se
from einsteinengine.symbolic.tensor import BaseRelativityTensor

class EnergyMomentumTensor(BaseRelativityTensor):
    
    @classmethod
    def from_perfect_fluid(cls, metric, density, pressure, four_velocity_cov, verbose=False):
        r"""
        Constructs the Energy-Momentum Tensor for a Perfect Fluid.
        Formula: T_\mu\nu = (\rho + p) u_\mu u_\nu + p g_\mu\nu
        
        Args:
            metric (MetricTensor): The metric tensor of the spacetime.
            density (sp.Expr): The energy density (\rho).
            pressure (sp.Expr): The isotropic pressure (p).
            four_velocity_cov (list): The covariant four-velocity vector u_\mu (e.g., [-1, 0, 0, 0]).
        """
        if verbose:
            print(f"Building Perfect Fluid Energy-Momentum Tensor for '{metric.name}'...")
            
        dims = metric.dims
        syms = metric.syms
        g_data = metric.get_raw_data()
        
        # 1. Symenginify physical parameters
        rho = se.sympify(density)
        p = se.sympify(pressure)
        u_mu = [se.sympify(val) for val in four_velocity_cov]
        
        # 2. Initialize the 2D tensor T_\mu\nu
        T_data = [[se.sympify(0) for _ in range(dims)] for _ in range(dims)]
        
        # Enthalpy factor: (\rho + p)
        enthalpy = rho + p
        
        # 3. Apply the perfect fluid formula cell by cell
        for mu in range(dims):
            for nu in range(dims):
                term1 = enthalpy * u_mu[mu] * u_mu[nu]
                term2 = p * g_data[mu][nu]
                T_data[mu][nu] = term1 + term2
                
        fluid_name = f"Fluid_{metric.name}"
        # Returned as a doubly covariant tensor (config 'll')
        return cls(T_data, syms, config="ll", name=fluid_name, verbose=verbose)
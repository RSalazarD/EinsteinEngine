import symengine as se
from einsteinengine.symbolic.tensor import BaseRelativityTensor
from einsteinengine.symbolic.metric import MetricTensor

class ChristoffelSymbols(BaseRelativityTensor):
    """
    Computes the Christoffel Symbols of the second kind (Gamma^lambda_{mu nu})
    optimized via SymEngine backend processing.
    """

    def __init__(self, arr, syms, name="ChristoffelSymbols",verbose=False):
        # config="ull" -> 1 up, 2 down
        super().__init__(arr, syms, config="ull", name=name,verbose=verbose)

    @classmethod
    def from_metric(cls, metric: MetricTensor, verbose=False):
        
        if verbose:
            print(f"Computing Christoffel Symbols for {metric.name} in C++...")
        
        g = metric._tensor
        g_inv = metric.inv()._tensor
        syms = metric.syms
        dims = metric.dims
        
        # Initialize a 3D grid structure for Gamma (4x4x4)
        gamma = [[[se.sympify(0) for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
        
        # Partial derivative loops triggered directly in C++
        for rho in range(dims):
            for mu in range(dims):
                for nu in range(dims):
                    sum_val = se.sympify(0)
                    for sigma in range(dims):
                        # Einstein summation convention terms
                        d_g_sigma_mu_nu = se.diff(g[sigma][mu], syms[nu])
                        d_g_sigma_nu_mu = se.diff(g[sigma][nu], syms[mu])
                        d_g_mu_nu_sigma = se.diff(g[mu][nu], syms[sigma])
                        
                        term = g_inv[rho][sigma] * (d_g_sigma_mu_nu + d_g_sigma_nu_mu - d_g_mu_nu_sigma)
                        sum_val += term
                    
                    gamma[rho][mu][nu] = se.Rational(1, 2)*sum_val
                    
        return cls(gamma, syms, name=f"Christoffel_{metric.name}", verbose=verbose)
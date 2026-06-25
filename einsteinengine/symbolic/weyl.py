import symengine as se
import sympy as sp
from einsteinengine.symbolic import metric
from einsteinengine.symbolic.tensor import BaseRelativityTensor
from einsteinengine.symbolic.ricci_tensor import RicciTensor
from einsteinengine.symbolic.ricci_scalar import RicciScalar
from einsteinengine.symbolic.riemann import RiemannCurvatureTensor

class WeylTensor(BaseRelativityTensor):
    
    @classmethod
    def from_metric(cls, metric, riemann=None, verbose=False):
        dims = metric.dims
        syms = metric.syms
        
        # 1. Pipeline check: If Riemann tensor is not provided, compute it from metric
        if riemann is None:
            riemann = RiemannCurvatureTensor.from_metric(metric, verbose=verbose)
            
        # 2. Leverage your native OOP contraction pipeline to build Ricci structures
        ricci_tensor = RicciTensor.from_riemann(riemann, verbose=verbose)
        ricci_scalar_obj = RicciScalar.from_ricci_tensor_and_metric(ricci_tensor, metric, verbose=verbose)
        
        # Extract the underlying raw data arrays for high-speed element access
        r_tensor_data = ricci_tensor.get_raw_data()
        
        # Ensure the Ricci Scalar data is unboxed properly if wrapped in a single-element list
        r_scalar_data = ricci_scalar_obj.get_raw_data()
        
        # Robust unboxing: Drill down in case the scalar is trapped in nested lists (e.g., [[value]])
        while isinstance(r_scalar_data, list):
            r_scalar_data = r_scalar_data[0]
            
        # Explicitly cast to a SymEngine object. 
        r_scalar_data = se.sympify(r_scalar_data)

        g_inv = metric.inv().get_raw_data()        
        g_cov = metric.get_raw_data()

        # 3. Raise one index on the Ricci Tensor manually to build R^mu_rho for the Weyl formula
        # Mathematical definition: R^mu_rho = g^mu_alpha * R_alpha_rho
        ricci_up_tensor = ricci_tensor.raise_index(0, metric.inv(), verbose=verbose)
        ricci_up = ricci_up_tensor.get_raw_data()

        # 4. Initialize the 4D nested list to store the final Weyl components
        c_array = [[[[0 for _ in range(dims)] for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]

        if verbose:
            print(f"Building Weyl Tensor from {metric.name} components...")

        # 5. Core 4D Tensor Contraction: Trace-free subtraction of Ricci curvature from Riemann
        for mu in range(dims):
            for nu in range(dims):
                for rho in range(dims):
                    for sigma in range(dims):
                        # Inline evaluation of Kronecker delta terms
                        kronecker_mu_rho = 1 if mu == rho else 0
                        kronecker_mu_sigma = 1 if mu == sigma else 0
                        
                        # Retrieve the base Riemann component
                        term_riemann = riemann.get_raw_data()[mu][nu][rho][sigma]
                        
                        # Part 1: Ricci Tensor corrections (trace removal)
                        term_ricci = (
                            kronecker_mu_rho * r_tensor_data[nu][sigma]
                            - kronecker_mu_sigma * r_tensor_data[nu][rho]
                            + g_cov[nu][sigma] * ricci_up[mu][rho]
                            - g_cov[nu][rho] * ricci_up[mu][sigma]
                        )
                        
                        # Part 2: Ricci Scalar corrections
                        term_scalar = (
                            kronecker_mu_rho * g_cov[nu][sigma]
                            - kronecker_mu_sigma * g_cov[nu][rho]
                        )
                        
                        # Combine all geometric pieces according to the 4D Weyl formulation
                        c_val = term_riemann - sp.Rational(1, 2) * term_ricci + sp.Rational(1, 6) * r_scalar_data * term_scalar                        
                        # Enforce optimization by wrapping the result as a native SymEngine object
                        c_array[mu][nu][rho][sigma] = se.sympify(c_val)

        # Return the final object using your base relativity constructor configuration
        return cls(c_array, syms, config="ulll", name=f"Weyl_{metric.name}", verbose=verbose)
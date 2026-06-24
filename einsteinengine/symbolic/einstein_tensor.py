import symengine as se
from einsteinengine.symbolic.tensor import BaseRelativityTensor

class EinsteinTensor(BaseRelativityTensor):
    
    @classmethod
    def from_components(cls, metric, ricci_tensor, ricci_scalar, verbose=False):
        """
        Computes the Einstein Tensor (G_mu_nu) combining the geometric components:
        G_mu_nu = R_mu_nu - (1/2) * R * g_mu_nu
        """
        if verbose:
            print(f"Building Einstein Tensor for '{metric.name}'...")
            
        # 1. Extract pure C++ matrices from our objects
        g_data = metric.get_raw_data()
        R_mu_nu = ricci_tensor.get_raw_data()
        R_scalar = ricci_scalar.get_raw_data()
        
        syms = metric.syms
        dims = metric.dims
        
        # 2. Initialize a 2D matrix for G_mu_nu
        G_tensor = [[se.sympify(0) for _ in range(dims)] for _ in range(dims)]
        
        # Exact symbolic fraction to avoid floating point precision issues
        half = se.Rational(1, 2)
        
        # 3. Apply the fundamental formula cell by cell
        for mu in range(dims):
            for nu in range(dims):
                G_tensor[mu][nu] = R_mu_nu[mu][nu] - half * R_scalar * g_data[mu][nu]
                
        # 4. Instantiate as a rank-2 covariant tensor (config="ll")
        clean_name = metric.name.replace("Metric_", "")
        return cls(G_tensor, syms, config="ll", name=f"Einstein_{clean_name}", verbose=verbose)
    
    @classmethod
    def from_metric(cls, metric, verbose=False):
        """
        Convenience pipeline method: Computes the Einstein Tensor directly from a Metric.
        """
        if verbose:
            print(f"--- Auto-generating Curvature Pipeline for '{metric.name}' ---")
            
        # Local imports
        from einsteinengine.symbolic.riemann import RiemannCurvatureTensor
        from einsteinengine.symbolic.ricci_tensor import RicciTensor
        from einsteinengine.symbolic.ricci_scalar import RicciScalar

        riemann = RiemannCurvatureTensor.from_metric(metric, verbose=verbose)
        ricci_tensor = RicciTensor.from_riemann(riemann, verbose=verbose)
        ricci_scalar = RicciScalar.from_ricci_tensor_and_metric(ricci_tensor, metric, verbose=verbose)

        # Pass the calculated components to our main constructor
        return cls.from_components(metric, ricci_tensor, ricci_scalar, verbose=verbose)


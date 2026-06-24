import symengine as se
from einsteinengine.symbolic.tensor import BaseRelativityTensor

class RicciScalar(BaseRelativityTensor):
    
    @classmethod
    def from_ricci_tensor_and_metric(cls, ricci_tensor, metric, verbose=False):
        """
        Computes the Ricci Scalar by contracting the Ricci Tensor 
        with the inverse Metric Tensor using the universal contraction engine
        """
        if verbose:
            print("Building Ricci Scalar by contracting Ricci Tensor with inverse metric...")
            
        g_inv = metric.inv()
        
        # R_mu_nu has config 'll'. g^mu_nu has config 'uu'.
        scalar_tensor = ricci_tensor.multiply_and_contract(
            g_inv, 
            pairs=[(0, 0), (1, 1)], 
            new_name=f"RicciScalar_{metric.name}"
        )
        
        # Re-instantiate as RicciScalar to maintain strict semantics (Domain-Driven Design)
        return cls(
            scalar_tensor.get_raw_data(), 
            metric.syms, 
            config="",  # Empty configuration because it is rank 0
            name=scalar_tensor.name, 
            verbose=verbose
        )
    

    
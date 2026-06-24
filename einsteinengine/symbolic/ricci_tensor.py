from einsteinengine.symbolic.tensor import BaseRelativityTensor

class RicciTensor(BaseRelativityTensor):
    
    @classmethod
    def from_riemann(cls, riemann, verbose=False):
        """
        Computes the Ricci Tensor (R_mu_nu) by contracting the 1st (upper) 
        and 3rd (lower) indices of the Riemann Tensor.
        """
        if verbose:
            print(f"Building Ricci Tensor from {riemann.name}...")
            
        # Riemann has config 'ulll'. 
        # Index 0 = 'u' (lambda)
        # Index 2 = 'l' (lambda contracted)
        contracted_tensor = riemann.contract_indices(0, 2, new_name=f"Ricci_{riemann.name}")
        
        return cls(contracted_tensor.get_raw_data(), riemann.syms, config=contracted_tensor.config, name=contracted_tensor.name, verbose=verbose)
    
    
import symengine as se
from einsteinengine.symbolic.core import BaseRelativityObject

class SpinConnection(BaseRelativityObject):
    
    @classmethod
    def from_tetrad_and_christoffel(cls, tetrad, christoffel, verbose=False):
        r"""
        Computes the Spin Connection \omega_{\mu \ \ b}^{\ a} fields.
        Requires a Tetrad e_\mu^a (config 'lu') and Christoffel Symbols \Gamma^nu_{\mu\sigma}.
        """
        if verbose:
            print("Computing Spin Connection fields from Tetrad and Christoffel...")
            
        dims = tetrad.dims
        syms = tetrad.syms
        
        # 1. Gather component raw data
        e_mu_a = tetrad.get_raw_data()          # e_\mu^a (config 'lu')
        inv_tetrad = tetrad.get_inverse()
        e_up_mu_flat_b = inv_tetrad.get_raw_data()  # e^\nu_b (config 'ul')
        Gamma = christoffel.get_raw_data()      # \Gamma^\nu_{\mu\sigma}
        
        # 2. Initialize 3D array: [mu][a][b]
        omega = [[[se.sympify(0) for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
        
        # 3. Compute the analytical formulation
        for mu in range(dims):
            for a in range(dims):
                for b in range(dims):
                    tmp_sum = se.sympify(0)
                    for nu in range(dims):
                        # Term 1: e_\nu^a * \partial_\mu(e^\nu_b)
                        partial_term = se.diff(e_up_mu_flat_b[nu][b], syms[mu])
                        tmp_sum += e_mu_a[nu][a] * partial_term
                        
                        # Term 2: e_\nu^a * \Gamma^\nu_{\mu\sigma} * e^\sigma_b
                        for sigma in range(dims):
                            tmp_sum += e_mu_a[nu][a] * Gamma[nu][mu][sigma] * e_up_mu_flat_b[sigma][b]
                            
                    omega[mu][a][b] = tmp_sum
                    
        return cls(omega, syms, name=f"SpinConnection_{tetrad.name}", verbose=verbose)
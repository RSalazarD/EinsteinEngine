import symengine as se
from einsteinengine.symbolic.connection import BaseConnection
class ChristoffelSymbols(BaseConnection):
    """
    Computes the Christoffel Symbols of the second kind (Gamma^lambda_{mu nu})
    optimized via SymEngine backend processing.
    """
    @classmethod
    def from_metric(cls, metric, verbose=False):
        """
        Computes the Christoffel Symbols of the second kind (Gamma^lambda_{mu nu})
        optimized via SymEngine backend processing.
        """
        if verbose:
            print(f"Computing Christoffel Symbols for '{metric.name}' in C++...")
            
        g = metric._data
        g_inv = metric.inv()._data
        syms = metric.syms
        dims = metric.dims
        
        # Initialize a 3D grid structure for Gamma (4x4x4)
        Gamma = [[[se.sympify(0) for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
        
        # Heavy-lifting partial derivative loops triggered directly in C++
        for lambda_ in range(dims):
            for mu in range(dims):
                for nu in range(dims):
                    tmp_sum = se.sympify(0)
                    for sigma in range(dims):
                        term1 = se.diff(g[nu][sigma], syms[mu])
                        term2 = se.diff(g[mu][sigma], syms[nu])
                        term3 = se.diff(g[mu][nu], syms[sigma])
                        tmp_sum += g_inv[lambda_][sigma] * (term1 + term2 - term3)
                    
                    Gamma[lambda_][mu][nu] = se.Rational(1, 2) * tmp_sum
                    
        # Devuelve instanciando la nueva clase madre BaseConnection
        return cls(Gamma, syms, config="ull", name=f"Christoffel_{metric.name}", verbose=verbose)
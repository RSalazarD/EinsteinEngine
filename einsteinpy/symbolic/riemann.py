import symengine as se
from einsteinpy.symbolic.tensor import BaseRelativityTensor

class RiemannCurvatureTensor(BaseRelativityTensor):
    """ Riemann standard index configuration: 1 upper, 3 lower (ulll -> R^rho_{sigma mu nu})"""

    def __init__(self, arr, syms, name="Riemann", verbose=False):
        super().__init__(arr, syms,config="ulll", name=name, verbose=verbose)

    @classmethod
    def from_christoffel(cls, christoffel, verbose=False):
        """
        Computes the 256 components of the Riemann Curvature Tensor from Christoffel inputs.
        """

        if verbose:
            print("Building Riemann Curvature Tensor...")
        
        Gamma = christoffel.get_raw_data()
        syms = christoffel.syms
        dims = len(syms)
        
        # Initialize a 4D grid structure (4x4x4x4 = 256 elements)
        R = [[[[se.sympify(0) for _ in range(dims)] for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
        
        for rho in range(dims):
            for sigma in range(dims):
                for mu in range(dims):
                    for nu in range(dims):
                        # R^rho_{sigma mu nu} = d(Gamma)/d(mu) - d(Gamma)/d(nu) + contractions                        term1 = se.diff(Gamma[rho][nu][sigma], syms[mu])
                        term1 = se.diff(Gamma[rho][nu][sigma], syms[mu])
                        term2 = se.diff(Gamma[rho][mu][sigma], syms[nu])
                        
                        term3 = se.sympify(0)
                        term4 = se.sympify(0)
                        for lam in range(dims):
                            term3 += Gamma[rho][mu][lam] * Gamma[lam][nu][sigma]
                            term4 += Gamma[rho][nu][lam] * Gamma[lam][mu][sigma]
                        
                        R[rho][sigma][mu][nu] = term1 - term2 + term3 - term4
                        
        return cls(R, syms)
    
    @classmethod
    def from_metric(cls, metric, verbose=False):
        """
        Pipeline connection: Metric -> Christoffel -> Riemann Curvature Tensor
        """
        if verbose:
            print(f"Triggering pipeline execution for metric: '{metric.name}'")
    
        from einsteinpy.symbolic.christoffel import ChristoffelSymbols
        
        intermediate_christoffel = ChristoffelSymbols.from_metric(metric, verbose=verbose)
        return cls.from_christoffel(intermediate_christoffel, verbose=verbose)
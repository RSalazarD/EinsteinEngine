import symengine as se
from einsteinengine.symbolic.tensor import BaseRelativityTensor

class RiemannCurvatureTensor(BaseRelativityTensor):
    """ Riemann standard index configuration: 1 upper, 3 lower (ulll -> R^rho_{sigma mu nu})"""

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
                        
        return cls(R, syms, config="ulll", name="RiemannCurvature", verbose=verbose)    
    
    @classmethod
    def from_metric(cls, metric, verbose=False):
        """
        Pipeline connection: Metric -> Christoffel -> Riemann Curvature Tensor
        """
        if verbose:
            print(f"Triggering pipeline execution for metric: '{metric.name}'")
    
        from einsteinengine.symbolic.christoffel import ChristoffelSymbols
        
        intermediate_christoffel = ChristoffelSymbols.from_metric(metric, verbose=verbose)
        return cls.from_christoffel(intermediate_christoffel, verbose=verbose)
    
    def kretschmann_scalar(self, metric, simplify=False, verbose=False):
        """
        Computes the Kretschmann scalar K = R^{abcd} R_{abcd}.
        Automatically fetches the cached inverse metric to optimize performance.
        
        Args:
            metric (BaseRelativityTensor): Covariant metric tensor ('ll').
            simplify (bool): Whether to simplify the resulting scalar.
            verbose (bool): Whether to print detailed information about the computation.
            
        Returns:
            sp.Expr: The simplified scalar mathematical expression.
        """
        import sympy as sp
        
        if verbose:
            print(f"[{self.name}] Computing Kretschmann Scalar...")

        # 1. Fetch the inverse metric (Instantly returns if already cached!)
        try:
            metric_inv = metric.inv()
        except AttributeError:
            raise TypeError("The 'metric' argument must have an 'inv()' method.")

        # 2. Obtain the fully covariant Riemann Tensor (R_{abcd} -> config 'llll')
        if self.config == 'ulll':
            if verbose: print(f"[{self.name}] Lowering first index to get R_{{abcd}}...")
            R_down = self.lower_index(0, metric, verbose=False)
        elif self.config == 'llll':
            R_down = self
        else:
            raise ValueError(f"Unexpected index configuration '{self.config}' for Riemann tensor. Expected 'ulll' or 'llll'.")

        # 3. Obtain the fully contravariant Riemann Tensor (R^{abcd} -> config 'uuuu')
        # We sequentially raise all 4 indices using the cached inverse
        if verbose: print(f"[{self.name}] Raising all indices to get R^{{abcd}}...")
        R_up = R_down.raise_index(0, metric_inv, verbose=False) \
                     .raise_index(1, metric_inv, verbose=False) \
                     .raise_index(2, metric_inv, verbose=False) \
                     .raise_index(3, metric_inv, verbose=False)

        # 4. Perform the full contraction: sum( R_{abcd} * R^{abcd} )
        if verbose: print(f"[{self.name}] Executing full scalar contraction (256 terms)...")
        K_val = 0
        down_data = R_down.get_raw_data()
        up_data = R_up.get_raw_data()
        dims = self.dims

        for a in range(dims):
            for b in range(dims):
                for c in range(dims):
                    for d in range(dims):
                        term = down_data[a][b][c][d] * up_data[a][b][c][d]
                        if term != 0:  # Optimization: avoid adding zeros
                            K_val += term

        if simplify:
            if verbose: print(f"[{self.name}] Simplifying the invariant scalar... (This might take a while)")
            return sp.simplify(K_val)
        else:
            if verbose: print(f"[{self.name}] Returning raw scalar (Simplification skipped for performance).")
            return K_val


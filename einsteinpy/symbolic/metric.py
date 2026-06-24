import symengine as se
import sympy as sp
from einsteinpy.symbolic.tensor import BaseRelativityTensor

class MetricTensor(BaseRelativityTensor):
    """
    Class representing the Metric Tensor ($g_{\mu\nu}$)
    Inherits from BaseRelativityTensor
    """
    
    def __init__(self, arr, syms, config="ll", name="MetricTensor", verbose=False):
        # Call the parent constructor (tensor.py) to handle the heavy lifting
        # setting up indices, names, and converting everything to SymEngine
        super().__init__(arr, syms, config=config, name=name,verbose=verbose)
        
        if verbose:
            print(f"[{self.name}] Métrica validada y lista para cálculos.")

        if len(self._tensor) != len(self._tensor[0]):
            raise ValueError("Metric must be a squared matrix")
            
        self.dims = len(syms)
        
        # Cache interna para la inversa, para no recalcularla múltiples veces
        self._inverse_tensor = None

    def inv(self):
        """
        Computes the inverse metric tensor (g^{mu nu}).
        Uses a caching mechanism to avoid redundant calculations.
        """
        # If already computed, return it instantly
        if self._inverse_tensor is not None:
            return self._inverse_tensor

        # If not, run the SymPy inversion pipeline
        
        sp_matrix = sp.Matrix(self._tensor)
        dims = sp_matrix.shape[0]
        
        # Handle cases where SymEngine structures read the data as a flat 1D column
        if sp_matrix.shape == (dims * dims, 1):
            sp_matrix = sp_matrix.reshape(dims, dims)
            
        # Perform the algebraic inversion securely
        sp_inv = sp_matrix.inv()
        
        # Safely parse the SymPy Matrix back into standard nested lists
        inv_arr = sp_inv.tolist()
        
        # Store the instance before returning it
        self._inverse_tensor = self.__class__(
            inv_arr, 
            self.syms, 
            config="uu", 
            name=f"{self.name}_inv", 
            verbose=False
        )
        
        return self._inverse_tensor
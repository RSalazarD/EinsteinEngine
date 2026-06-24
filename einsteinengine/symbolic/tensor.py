from einsteinengine.symbolic.core import BaseRelativityObject
import symengine as se
import sympy as sp
class BaseRelativityTensor(BaseRelativityObject):
    """
    Base class for strict tensor objects (Metric, Riemann, Ricci, etc.).
    Inherits coordinate and array management from BaseRelativityObject.
    Adds tensor-specific properties like index configuration (e.g., 'll', 'uu', 'ul').
    """

    def __init__(self, arr, syms, config="ll", name="GenericTensor", verbose=False):
        # 1. Delegate the heavy lifting to the parent object (coordinates & SymEngine)
        super().__init__(arr, syms, name=name, verbose=verbose)
        
        # 2. Tensor-exclusive physical properties
        self.config = config
        
        # 3. Component validation
        # Ensure the length of the configuration matches the tensor rank
        # (e.g., a 2D matrix needs a 2-letter config like 'll')
        if self._data and isinstance(arr, list):
            rank = self._calculate_rank(arr)
            if len(self.config) != rank:
                raise ValueError(f"Configuration length ({len(self.config)}) does not match tensor rank ({rank})")

        if self.verbose:
            print(f"[{self.name}] Tensor initialized with index configuration: '{self.config}'.")
  
    def _calculate_rank(self, arr):
        """
        Calculates the mathematical rank (number of indices) of the tensor
        by measuring the depth of the nested lists.
        """
        if isinstance(arr, list):
            return 1 + self._calculate_rank(arr[0])
        return 0

    def _symenginify_array(self, arr):
        """
        Recursive method to convert any nested list of SymPy 
        objects or raw strings into SymEngine objects.
        """
        if isinstance(arr, (list, tuple)):
            return [self._symenginify_array(element) for element in arr]
        else:
            return se.sympify(arr)

    def get_component(self, *indices):
        """Calculates and simplifies a single component."""
        val = self._data
        for i in indices:
            val = val[i]
        return sp.simplify(sp.sympify(val))

    def _sympyfy_and_simplify(self, arr):
        """
        Recursive method to convert from SymEngine to Sympy and simplify
        """
        if isinstance(arr, (list, tuple)):
            return [self._sympyfy_and_simplify(element) for element in arr]
        else:
            # converts SymEngine in SymPy and simplifies
            sympy_expr = sp.sympify(arr)
            return sp.trigsimp(sympy_expr)
    
    def to_latex(self, *indices):
        """
        If indices are provided (e.g. to_latex(0, 1, 1)), simplifies only that component.
        If no arguments are passed, converts the entire tensor to LaTeX.
        """
        if indices:
            # Specific component 
            val = self.get_component(*indices)
            return sp.latex(val)
        else:
            # Entire tensor (caution: can be very large) 
            return sp.latex(self._data)
    
    def _repr_latex_(self):
        """
        Special method hooks into Jupyter Notebook's display system.
        When the object name is evaluated in a cell, Jupyter automatically
        calls this method to render the output as LaTeX.
        """
        return f"$$ {self.to_latex()} $$"
    
    def get_non_zero(self):
        """
        Returns a list of tuples (indices, value) for all non-zero components.
        """
        non_zeros = []
        
        # For any range tensors
        def _recurse(arr, current_indices):
            for i, val in enumerate(arr):
                if isinstance(val, list):
                    _recurse(val, current_indices + [i])
                else:
                    if val != 0:
                        non_zeros.append((current_indices + [i], val))
        
        _recurse(self._data, [])
        return non_zeros
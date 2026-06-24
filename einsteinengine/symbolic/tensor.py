import symengine as se
import sympy as sp

class BaseRelativityTensor:
    """
    Base class for all tensors in the library (Metric, Christoffel, Riemann, etc.).
    Uses SymEngine (C++) as the backend to improve performance.
    """

    def __init__(self, arr, syms, config="ll", name="GenericTensor", verbose=False):
        """
        arr: Nested list containing the mathematical components of the tensor.
        syms: Coordinate symbols (e.g., t, r, theta, phi).
        config: Covariant ('l') or contravariant ('u') indices.
        """
        self.name = name
        self.config = config
        
        # Test Comments control 
        if verbose:
            print(f"Initializing base structure (SymEngine) for: {self.name}...")

        # Converting coordinates to SymEngine        
        self.syms = [se.sympify(s) for s in syms]
        
        # Save the tensor components by converting them to SymEngine.
        # Store internally as a generic nested list.
        self._tensor = self._symenginify_array(arr)

    def _symenginify_array(self, arr):
        """
        Recursive method to convert any nested list of SymPy 
        objects or raw strings into SymEngine objects.
        """
        if isinstance(arr, (list, tuple)):
            return [self._symenginify_array(element) for element in arr]
        else:
            return se.sympify(arr)

    def get_raw_data(self):
        """Returns the unsimplified data."""
        return self._tensor

    def get_component(self, *indices):
        """Calculates and simplifies a single component."""
        val = self._tensor
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
            return sp.latex(self._tensor)
    
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
        
        _recurse(self._tensor, [])
        return non_zeros
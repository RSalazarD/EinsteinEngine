import symengine as se

class BaseRelativityObject:
    """
    Absolute base class for any geometric object in EinsteinEngine.
    Handles coordinate symbols, dimensions, and C++ SymEngine integration.
    Does NOT assume tensor transformation laws.
    """
    def __init__(self, arr, syms, name="GeometricObject", verbose=False):
        self.name = name
        self.verbose = verbose
        
        # Coordinate Management (Our implicit Chart)
        self.syms = [se.sympify(s) for s in syms]
        self.dims = len(self.syms)
        
        # Convert mathematical arrays to high-performance C++ objects
        self._data = self._symenginify_array(arr)
        
        if self.verbose:
            print(f"[{self.name}] Instantiated in {self.dims}D spacetime.")

    def get_raw_data(self):
        """Returns the internal SymEngine matrix/array."""
        return self._data

    def _symenginify_array(self, arr):
        """
        Recursively converts Python lists/SymPy objects into SymEngine objects.
        Works for 1D vectors, 2D matrices, or 4D Riemann structures.
        """
        if isinstance(arr, list):
            return [self._symenginify_array(item) for item in arr]
        else:
            # Handles integers, floats, and SymPy expressions
            return se.sympify(arr)
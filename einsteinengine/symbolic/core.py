import symengine as se
import sympy as sp
import copy

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
        self._raw_component_cache = {}
        
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

    def _calculate_rank(self, arr):
        """Calculates the mathematical rank dynamically."""
        if isinstance(arr, list):
            return 1 + self._calculate_rank(arr[0])
        return 0

    def _get_raw_component(self, *indices):
        """Return a raw nested component while reusing a lightweight cache."""
        key = tuple(indices)
        if key in self._raw_component_cache:
            return self._raw_component_cache[key]

        val = self._data
        for idx in indices:
            val = val[idx]

        self._raw_component_cache[key] = val
        return val

    def _algebraic_contraction(self, idx1, idx2):
        """
        Hidden mathematical engine: Contracts two indices of an N-dimensional nested list.
        Works recursively for any rank dynamically.
        """
        # Aseguramos que idx1 es el menor para no liarnos al reordenar
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            
        rank = self._calculate_rank(self._data)
        new_rank = rank - 2
        free_axes = [i for i in range(rank) if i not in (idx1, idx2)]
            
        # Función recursiva que construye la nueva matriz con N-2 dimensiones
        def build_contracted(current_free_coords):
            # Si ya tenemos suficientes coordenadas libres, sumamos el índice mudo (dummy)
            if len(current_free_coords) == new_rank:
                tmp_sum = se.sympify(0)
                for dummy in range(self.dims):
                    orig_coords = [0] * rank
                    orig_coords[idx1] = dummy
                    orig_coords[idx2] = dummy

                    for free_ptr, axis in enumerate(free_axes):
                        orig_coords[axis] = current_free_coords[free_ptr]

                    tmp_sum += self._get_raw_component(*orig_coords)
                return tmp_sum
            else:
                # Si faltan coordenadas, seguimos escarbando (recursión)
                return [build_contracted(current_free_coords + [d]) for d in range(self.dims)]
                
        # Lanzamos la recursión desde cero
        return build_contracted([])   
    
    def simplify(self, in_place=True, mode="full"):
        """
        Apply simplification to the object's data using a configurable strategy.

        Args:
            in_place (bool): If True, modify the object's internal data. If False,
                return a simplified copy.
            mode (str): Either 'light' for a fast structural cleanup or 'full'
                for deeper algebraic simplification.
        """
        if mode not in {"light", "full"}:
            raise ValueError("mode must be either 'light' or 'full'")

        if self.verbose:
            print(f"[{self.name}] Applying {mode} simplification...")

        def _recursive_simplify(arr):
            if isinstance(arr, list):
                return [_recursive_simplify(item) for item in arr]
            else:
                expr = sp.sympify(arr)
                if mode == "full":
                    expr = sp.simplify(expr)
                else:
                    expr = sp.cancel(expr)
                return se.sympify(expr)

        simplified_data = _recursive_simplify(self._data)

        if in_place:
            self._data = simplified_data
            if self.verbose:
                print(f"[{self.name}] Simplification complete.")
            return self
        else:
            new_obj = copy.copy(self)
            new_obj._data = simplified_data
            if self.verbose:
                print(f"[{self.name}] Created simplified copy.")
            return new_obj

    
    
    
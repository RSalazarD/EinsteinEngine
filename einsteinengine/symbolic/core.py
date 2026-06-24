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
        
        # Función auxiliar para extraer el valor en una coordenada N-dimensional
        def get_element(indices):
            val = self._data
            for i in indices:
                val = val[i]
            return val
            
        # Función recursiva que construye la nueva matriz con N-2 dimensiones
        def build_contracted(current_free_coords):
            # Si ya tenemos suficientes coordenadas libres, sumamos el índice mudo (dummy)
            if len(current_free_coords) == new_rank:
                tmp_sum = se.sympify(0)
                for dummy in range(self.dims):
                    # Reconstruimos la coordenada original completa
                    orig_coords = [0] * rank
                    orig_coords[idx1] = dummy
                    orig_coords[idx2] = dummy
                    
                    # Rellenamos el resto de huecos con las coordenadas libres
                    free_ptr = 0
                    for i in range(rank):
                        if i != idx1 and i != idx2:
                            orig_coords[i] = current_free_coords[free_ptr]
                            free_ptr += 1
                            
                    tmp_sum += get_element(orig_coords)
                return tmp_sum
            else:
                # Si faltan coordenadas, seguimos escarbando (recursión)
                return [build_contracted(current_free_coords + [d]) for d in range(self.dims)]
                
        # Lanzamos la recursión desde cero
        return build_contracted([])   
    
    def simplify(self, in_place=True):
        """
        Applies algebraic and trigonometric simplification to the object's data.
        It uses SymPy's pattern recognition engine to collapse equations 
        and then converts the data back to SymEngine for fast future computations.
        
        Args:
            in_place (bool): If True, modifies the object's internal data. 
                             If False, returns a new object with the simplified data, 
                             leaving the original untouched.
                             """
        if self.verbose:
            print(f"[{self.name}] Applying deep simplification... (This may take a while for large tensors)")

        # Recursive helper function to dig into nested lists of any tensor rank
        def _recursive_simplify(arr):
            if isinstance(arr, list):
                return [_recursive_simplify(item) for item in arr]
            else:
                # 1. Convert SymEngine object to pure SymPy (sp.sympify)
                # 2. Apply deep algebraic/trigonometric simplification (sp.simplify)
                # 3. Shield it back into C++ SymEngine for performance (se.sympify)
                return se.sympify(sp.simplify(sp.sympify(arr)))

        simplified_data = _recursive_simplify(self._data)

        if in_place:
            self._data = simplified_data
            if self.verbose:
                print(f"[{self.name}] Simplification complete.")
                
            # Return self to allow method chaining (e.g., tensor.simplify().get_raw_data())
            return self
        else:
            # Creamos un clon exacto del objeto actual (sea Tensor, Conexión, etc.)
            new_obj = copy.copy(self)
            # Le inyectamos los datos limpios al clon
            new_obj._data = simplified_data
            if self.verbose:
                print(f"[{self.name}] Created simplified copy.")
            return new_obj


    
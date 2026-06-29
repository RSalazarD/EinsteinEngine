from einsteinengine.symbolic.core import BaseRelativityObject
import symengine as se
import sympy as sp
import itertools

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

        self._component_cache = {}

        if self.verbose:
            print(f"[{self.name}] Tensor initialized with index configuration: '{self.config}'.")
  
    def _symenginify_array(self, arr):
        """
        Recursive method to convert any nested list of SymPy 
        objects or raw strings into SymEngine objects.
        """
        if isinstance(arr, (list, tuple)):
            return [self._symenginify_array(element) for element in arr]
        else:
            return se.sympify(arr)

    def get_component(self, *indices, simplify=True):
        """Return a single tensor component.

        By default the value is simplified for readability, but a fast path is
        available for performance-sensitive access where simplification is not
        required. Repeated access is cached to reduce overhead for large tensor
        workloads.
        """
        key = (tuple(indices), simplify)
        if key in self._component_cache:
            return self._component_cache[key]

        val = self._get_raw_component(*indices)

        if not simplify:
            self._component_cache[key] = val
            return val

        expr = sp.sympify(val)
        result = sp.simplify(expr)

        self._component_cache[key] = result
        return result

    def clear_component_cache(self):
        """Clear cached component values to free memory after large symbolic workloads."""
        self._component_cache.clear()
        self._raw_component_cache.clear()

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
    
    def _format_value(self, value, format="latex", simplify=False):
        """Render a scalar, component, or nested tensor structure as text."""
        if isinstance(value, list):
            if not value:
                return "[]"

            if all(isinstance(item, list) for item in value):
                rows = [self._format_value(row, format=format, simplify=simplify) for row in value]
                if format == "latex":
                    return r"\\begin{bmatrix} " + r" \\ " + r" \\ ".join(rows) + r" \\end{bmatrix}"
                return "[" + ", ".join(rows) + "]"

            if format == "latex":
                return r"\\left[ " + ", ".join(self._format_value(item, format=format, simplify=simplify) for item in value) + r" \\right]"
            return "[" + ", ".join(self._format_value(item, format=format, simplify=simplify) for item in value) + "]"

        expr = sp.sympify(value)
        if simplify:
            expr = sp.simplify(expr)

        if format == "latex":
            return sp.latex(expr)
        return sp.sstr(expr)

    def to_string(self, format="latex", simplify=False, *indices):
        """Return a human-readable representation of a tensor or one of its components.

        Args:
            format: Either 'latex' for notebook-style output or 'plain' for a lightweight text fallback.
            simplify: If True, simplify the expression before formatting.
            indices: Optional component coordinates (e.g. to_string(0, 1, 1)).
        """
        if indices:
            value = self.get_component(*indices, simplify=simplify)
            return self._format_value(value, format=format, simplify=False)

        return self._format_value(self._data, format=format, simplify=simplify)

    def to_latex(self, *indices):
        """
        If indices are provided (e.g. to_latex(0, 1, 1)), simplifies only that component.
        If no arguments are passed, converts the entire tensor to LaTeX.
        """
        return self.to_string(format="latex", simplify=True, *indices)

    def __str__(self):
        return self.to_string(format="plain")

    def __repr__(self):
        return self.to_string(format="plain")
    
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
    
    def contract_indices(self, idx1, idx2, new_name=None):
        """
        Contracts one upper ('u') and one lower ('l') index of the tensor.
        Returns a brand new BaseRelativityTensor of rank N-2.
        """
        if idx1 == idx2:
            raise ValueError("Cannot contract an index with itself.")
            
        rank = len(self.config)
        if idx1 >= rank or idx2 >= rank:
            raise IndexError(f"Tensor has rank {rank}. Cannot contract indices at {idx1} and {idx2}.")
            
        if self.config[idx1] == self.config[idx2]:
            raise ValueError(f"Invalid contraction: indices {idx1} and {idx2} have the same variance ('{self.config[idx1]}').")
            
        # Computing de new string of indices
        new_config = ""
        for i, char in enumerate(self.config):
            if i != idx1 and i != idx2:
                new_config += char
                
        if self.verbose:
            print(f"Contracting indices {idx1} and {idx2} of {self.name}...")
            
        new_data = self._algebraic_contraction(idx1, idx2)
        final_name = new_name if new_name else f"Contracted_{self.name}"
        
        # Devolvemos un nuevo tensor matemáticamente válido
        return BaseRelativityTensor(new_data, self.syms, config=new_config, name=final_name, verbose=self.verbose)
    
    def multiply_and_contract(self, other, pairs, new_name=None):
        """
        Internal contraction of two tensors (A * B).
        Avoids building the outer product intermediate tensor.
        'pairs' is a list of tuples: [(index_A, index_B), ...] which we are contracting
        """
        rank_self = len(self.config)
        rank_other = len(other.config)
        
        # Separate free indices (remaining) from contracted indices
        self_contracted = [p[0] for p in pairs]
        other_contracted = [p[1] for p in pairs]
        
        self_free = [i for i in range(rank_self) if i not in self_contracted]
        other_free = [i for i in range(rank_other) if i not in other_contracted]
        
        # Validate tensor physics (Covariant must contract with Contravariant)
        for s_idx, o_idx in pairs:
            if self.config[s_idx] == other.config[o_idx]:
                raise ValueError(f"Invalid contraction at pair ({s_idx}, {o_idx}): both have variance '{self.config[s_idx]}'.")
                
        # The new configuration is the concatenation of the free indices
        new_config = "".join([self.config[i] for i in self_free]) + \
                     "".join([other.config[i] for i in other_free])
                     
        # Fast helper function to read nested matrix values dynamically
        def get_val(obj, coords):
            return obj._get_raw_component(*coords)

        # Combinatorial Engine (einsum approach)
        dims = self.dims
        num_free = len(self_free) + len(other_free)
        num_dummy = len(pairs)
        
        # If the result is a scalar (0 free indices), handle the direct sum
        if num_free == 0:
            result = se.sympify(0)
            for dummy_vals in itertools.product(range(dims), repeat=num_dummy):
                coords_self = [0] * rank_self
                coords_other = [0] * rank_other
                for p_idx, (s_idx, o_idx) in enumerate(pairs):
                    coords_self[s_idx] = dummy_vals[p_idx]
                    coords_other[o_idx] = dummy_vals[p_idx]
                result += get_val(self, coords_self) * get_val(other, coords_other)
            
            final_name = new_name if new_name else "ContractedScalar"
            return BaseRelativityTensor(result, self.syms, config="", name=final_name, verbose=self.verbose)

        # 5. If it is not a scalar, build the resulting matrix recursively
        def build_result(free_coords_flat):
            if len(free_coords_flat) == num_free:
                tmp_sum = se.sympify(0)
                # Loop only over the dummy indices (the contracted ones)
                for dummy_vals in itertools.product(range(dims), repeat=num_dummy):
                    coords_self = [0] * rank_self
                    coords_other = [0] * rank_other
                    
                    # Set the free coordinates
                    for i, free_idx in enumerate(self_free):
                        coords_self[free_idx] = free_coords_flat[i]
                    for i, free_idx in enumerate(other_free):
                        coords_other[free_idx] = free_coords_flat[len(self_free) + i]
                        
                    # Set the dummy coordinates (the summed ones)
                    for p_idx, (s_idx, o_idx) in enumerate(pairs):
                        coords_self[s_idx] = dummy_vals[p_idx]
                        coords_other[o_idx] = dummy_vals[p_idx]
                        
                    # Multiply and sum "on the fly"
                    tmp_sum += get_val(self, coords_self) * get_val(other, coords_other)
                return tmp_sum
            else:
                return [build_result(free_coords_flat + [d]) for d in range(dims)]

        final_data = build_result([])
        final_name = new_name if new_name else f"Contracted_{self.name}_{other.name}"
        
        return BaseRelativityTensor(final_data, self.syms, config=new_config, name=final_name, verbose=self.verbose)
    
    def transform_coordinates(self, new_syms, old_coords_in_terms_of_new, new_name=None):
        """
        Transforms the tensor to a new coordinate system using Jacobian matrices.
        
        Args:
            new_syms (list): List of new coordinate symbols (e.g., [X, Y, Z, T]).
            old_coords_in_terms_of_new (list): Equations defining old symbols as functions of new symbols.
        """
        import itertools
        if self.verbose:
            print(f"[{self.name}] Performing general coordinate transformation...")
            
        rank = len(self.config)
        dims = self.dims
        new_dims = len(new_syms)
        new_se_syms = [se.sympify(s) for s in new_syms]
        
        # 1. Compute the forward Jacobian matrix: J^\mu_\alpha = \partial(old^\mu) / \partial(new^\alpha)
        # Rows: old coordinates (\mu), Columns: new coordinates (\alpha)
        jacobian = [[se.diff(old_coords_in_terms_of_new[mu], new_se_syms[alpha]) 
                     for alpha in range(new_dims)] for mu in range(dims)]
        
        # 2. Compute the inverse Jacobian matrix: J_inv^\alpha_\mu = \partial(new^\alpha) / \partial(old^\mu)
        # Required for upper ('u') contravariant indices transformation
        jac_matrix = se.Matrix(jacobian)
        jac_inverse = jac_matrix.inv()
        
        # Helper reader for old data nested positions
        def get_old_val(coords):
            val = self._data
            for c in coords: val = val[c]
            return val

        # 3. Recursive matrix builder for the transformed tensor
        def build_transformed(current_new_coords):
            if len(current_new_coords) == rank:
                # Inside the cell: apply the multi-index contraction with Jacobians
                tmp_sum = se.sympify(0)
                # Iterate over all combinations of the old coordinate indices
                for old_coords in itertools.product(range(dims), repeat=rank):
                    old_value = get_old_val(old_coords)
                    
                    # Compute the transformation multiplier factor
                    factor = se.sympify(1)
                    for i in range(rank):
                        mu = old_coords[i]          # Old index position
                        alpha = current_new_coords[i] # New index position
                        
                        if self.config[i] == 'l':
                            # Covariant index transforms with forward Jacobian
                            factor *= jacobian[mu][alpha]
                        elif self.config[i] == 'u':
                            # Contravariant index transforms with inverse Jacobian
                            factor *= jac_inverse[alpha, mu]
                            
                    tmp_sum += old_value * factor
                    
                # Substitute old variables with the new equations to eliminate old symbols
                for idx, old_s in enumerate(self.syms):
                    tmp_sum = tmp_sum.subs(old_s, old_coords_in_terms_of_new[idx])
                return tmp_sum
            else:
                return [build_transformed(current_new_coords + [d]) for d in range(new_dims)]

        transformed_data = build_transformed([])
        final_name = new_name if new_name else f"Transformed_{self.name}"
        
        # Return tensor under the new coordinate patch
        return BaseRelativityTensor(transformed_data, new_se_syms, config=self.config, name=final_name, verbose=self.verbose)

# --- Tensor Arithmetic ---

    def __add__(self, other):
        """Allows addition using the '+' operator: T = A + B"""
        if not isinstance(other, BaseRelativityTensor):
            raise TypeError("You can only add a Tensor to another Tensor.")
        if self.dims != other.dims or self.config != other.config:
            raise ValueError(f"Cannot add tensors with different indices or dimensions. Got {self.config} and {other.config}")

        # Recursive addition for any tensor rank
        def _recursive_add(a, b):
            if isinstance(a, list):
                return [_recursive_add(x, y) for x, y in zip(a, b)]
            return a + b

        new_data = _recursive_add(self._data, other._data)
        return BaseRelativityTensor(new_data, self.syms, self.config, name=f"({self.name} + {other.name})", verbose=self.verbose)

    def __sub__(self, other):
        """Allows subtraction using the '-' operator: T = A - B"""
        if not isinstance(other, BaseRelativityTensor):
            raise TypeError("You can only subtract a Tensor from another Tensor.")
        if self.dims != other.dims or self.config != other.config:
            raise ValueError("Cannot subtract tensors with different index configurations.")

        def _recursive_sub(a, b):
            if isinstance(a, list):
                return [_recursive_sub(x, y) for x, y in zip(a, b)]
            return a - b

        new_data = _recursive_sub(self._data, other._data)
        return BaseRelativityTensor(new_data, self.syms, self.config, name=f"({self.name} - {other.name})", verbose=self.verbose)

    def __mul__(self, scalar):
        """Allows right scalar multiplication: T = A * 5"""
        import symengine as se
        
        # We only allow multiplication by scalars (numbers or SymEngine/SymPy expressions)
        if isinstance(scalar, BaseRelativityTensor):
            raise TypeError("Use specific tensor product methods to multiply two tensors, not the '*' operator.")

        scalar_val = se.sympify(scalar)

        def _recursive_mul(a):
            if isinstance(a, list):
                return [_recursive_mul(x) for x in a]
            return a * scalar_val

        new_data = _recursive_mul(self._data)
        return BaseRelativityTensor(new_data, self.syms, self.config, name=f"(ScalarMul_{self.name})", verbose=self.verbose)

    def __rmul__(self, scalar):
        """Allows left scalar multiplication: T = 5 * A"""
        # Commutative operation, just call the normal __mul__
        return self.__mul__(scalar)

    def lower_index(self, pos, metric, verbose=False):
        """
        Lowers a contravariant ('u') index at the specified position using the metric tensor.
        T_{... mu ...} = sum_alpha (g_{mu alpha} * T^{... alpha ...})
        
        Args:
            pos (int): The position of the index to lower (0-indexed).
            metric (BaseRelativityTensor): The covariant metric tensor (config 'll').
        """
        import symengine as se
        # 1. Physical and dimensional validation
        if self.config[pos] != 'u':
            raise ValueError(f"Index at position {pos} is already covariant ('l') or invalid. Config: {self.config}")
        if metric.config != "ll":
            raise ValueError("Metric must be purely covariant ('ll') to lower an index.")
            
        if verbose:
            print(f"[{self.name}] Lowering index at position {pos}...")  
        dims = self.dims
        T_data = self.get_raw_data()
        g_data = metric.get_raw_data()
        rank = len(self.config)
        
        # 2. Update the configuration string dynamically (e.g., 'uul' -> 'lul')
        new_config = self.config[:pos] + 'l' + self.config[pos+1:]
        component_cache = {}

        def get_T(indices):
            key = tuple(indices)
            if key in component_cache:
                return component_cache[key]
            value = self._get_raw_component(*indices)
            component_cache[key] = value
            return value
            
        # Recursive builder to maintain exact index positions
        def build_lowered(current_indices):
            if len(current_indices) == rank:
                mu = current_indices[pos] # The index we are lowering
                term_sum = se.sympify(0)
                
                # Perform the Einstein summation over the dummy index 'alpha'
                for alpha in range(dims):
                    old_indices = list(current_indices)
                    old_indices[pos] = alpha
                    term_sum += g_data[mu][alpha] * get_T(old_indices)
                return term_sum
            else:
                return [build_lowered(current_indices + [d]) for d in range(dims)]
                
        new_data = build_lowered([])
        new_name = f"{self.name}_lowered_{pos}"
        
        # Return as a new Tensor object, allowing for method chaining
        return self.__class__(new_data, self.syms, config=new_config, name=new_name, verbose=verbose)

    def raise_index(self, pos, metric_inv, verbose=False):
        """
        Raises a covariant ('l') index at the specified position using the inverse metric tensor.
        T^{... mu ...} = sum_alpha (g^{mu alpha} * T_{... alpha ...})
        
        Args:
            pos (int): The position of the index to raise (0-indexed).
            metric_inv (BaseRelativityTensor): The contravariant inverse metric tensor (config 'uu').
        """
        import symengine as se
        
        if self.config[pos] != 'l':
            raise ValueError(f"Index at position {pos} is already contravariant ('u') or invalid. Config: {self.config}")
        if metric_inv.config != "uu":
            raise ValueError("Must provide the inverse metric ('uu') to raise an index.")
            
        if verbose:
            print(f"[{self.name}] Raising index at position {pos}...")
            
        dims = self.dims
        g_inv_data = metric_inv.get_raw_data()
        rank = len(self.config)
        
        new_config = self.config[:pos] + 'u' + self.config[pos+1:]
        component_cache = {}
        
        def get_T(indices):
            key = tuple(indices)
            if key in component_cache:
                return component_cache[key]
            value = self._get_raw_component(*indices)
            component_cache[key] = value
            return value
            
        def build_raised(current_indices):
            if len(current_indices) == rank:
                mu = current_indices[pos]
                term_sum = se.sympify(0)
                
                for alpha in range(dims):
                    old_indices = list(current_indices)
                    old_indices[pos] = alpha
                    term_sum += g_inv_data[mu][alpha] * get_T(old_indices)
                return term_sum
            else:
                return [build_raised(current_indices + [d]) for d in range(dims)]
                
        new_data = build_raised([])
        new_name = f"{self.name}_raised_{pos}"
        
        return self.__class__(new_data, self.syms, config=new_config, name=new_name, verbose=verbose)




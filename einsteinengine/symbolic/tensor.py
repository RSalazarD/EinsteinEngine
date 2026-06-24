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
        def get_val(data, coords):
            val = data
            for c in coords: val = val[c]
            return val

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
                result += get_val(self._data, coords_self) * get_val(other.get_raw_data(), coords_other)
            
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
                    tmp_sum += get_val(self._data, coords_self) * get_val(other.get_raw_data(), coords_other)
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




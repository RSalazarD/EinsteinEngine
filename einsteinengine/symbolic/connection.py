from einsteinengine.symbolic.core import BaseRelativityObject

class BaseConnection(BaseRelativityObject):
    """
    Base class for mathematical Connections (e.g., Levi-Civita, Spin, Affine).
    Inherits coordinate and array management from BaseRelativityObject.
    
    IMPORTANT: Connections are NOT tensors. They do not transform homogeneously 
    under coordinate changes, and operations like raising/lowering all indices 
    using the metric do not apply in the standard tensorial way.
    """
    def __init__(self, arr, syms, config="ull", name="GenericConnection", verbose=False):
        # 1. Delegate the heavy lifting to the parent object
        super().__init__(arr, syms, name=name, verbose=verbose)
        
        # 2. Connection index configuration (usually one upper, two lower)
        self.config = config
        
        # 3. Validation: Connections in standard GR are 3-index objects
        if self._data and isinstance(arr, list):
            rank = self._calculate_rank(arr)
            if rank != 3:
                raise ValueError(f"A connection must have exactly 3 indices. Got {rank}.")

        if self.verbose:
            print(f"[{self.name}] Connection initialized. Note: This object is NOT a tensor.")

    def _calculate_rank(self, arr):
        """
        Calculates the mathematical rank (number of indices) of the array
        by measuring the depth of the nested lists.
        """
        if isinstance(arr, list):
            return 1 + self._calculate_rank(arr[0])
        return 0

    # --- Core geometric operations for Connections ---
    
    def covariant_derivative(self, tensor_obj, verbose=False):
        r"""
        Computes the covariant derivative (\nabla_\mu) of a given tensor.
        The derivative adds a new covariant index (lower 'l') at the beginning of the tensor.
        Nabla_\mu T^{a}_{b} = \partial_\mu T^{a}_{b} + \Gamma^a_{\mu \sigma} T^\sigma_b - \Gamma^\sigma_{\mu b} T^a_\sigma
        """
        import symengine as se
        
        if verbose:
            print(f"[{self.name}] Computing covariant derivative for tensor '{tensor_obj.name}'...")
            
        dims = self.dims
        syms = self.syms
        rank = len(tensor_obj.config)
        T_data = tensor_obj.get_raw_data()
        Gamma = self._data  # self is the connection, config 'ull': Gamma[up][down][down]
        
        # Helper to extract the tensor component safely regardless of rank
        def get_T(indices):
            val = T_data
            for idx in indices:
                val = val[idx]
            return val

        # The covariant derivative ADDS a new covariant index at the front.
        # e.g., T^\alpha_\beta (config "ul") -> \nabla_\mu T^\alpha_\beta (config "lul")
        new_config = 'l' + tensor_obj.config
        
        def build_cov_dir(current_indices):
            # Base case: we have selected all coordinates for the new tensor
            if len(current_indices) == rank + 1:
                mu = current_indices[0]      # The derivative index (\nabla_\mu)
                T_indices = current_indices[1:] # The original tensor indices
                
                # 1. Base term: Partial derivative \partial_\mu T^{...}
                term = se.diff(get_T(T_indices), syms[mu])
                
                # 2. Correction terms: loop over each index of the original tensor
                for p in range(rank):
                    idx_type = tensor_obj.config[p]
                    current_idx_val = T_indices[p]
                    
                    if idx_type == 'u':
                        # Add connection term for contravariant index
                        # + \sum_\sigma \Gamma^{current_idx_val}_{\mu \sigma} * T^{... \sigma ...}
                        for sigma in range(dims):
                            mod_indices = list(T_indices)
                            mod_indices[p] = sigma
                            term += Gamma[current_idx_val][mu][sigma] * get_T(mod_indices)
                            
                    elif idx_type == 'l':
                        # Subtract connection term for covariant index
                        # - \sum_\sigma \Gamma^{\sigma}_{\mu current_idx_val} * T^{... \sigma ...}
                        for sigma in range(dims):
                            mod_indices = list(T_indices)
                            mod_indices[p] = sigma
                            term -= Gamma[sigma][mu][current_idx_val] * get_T(mod_indices)
                            
                return term
            
            # Recursive case: dive deeper into the dimensions
            else:
                return [build_cov_dir(current_indices + [d]) for d in range(dims)]
                
        cov_data = build_cov_dir([])
        
        # Local import to avoid circular dependencies
        from einsteinengine.symbolic.tensor import BaseRelativityTensor
        
        final_name = f"CovDeriv_{tensor_obj.name}"
        return BaseRelativityTensor(cov_data, syms, config=new_config, name=final_name, verbose=verbose)



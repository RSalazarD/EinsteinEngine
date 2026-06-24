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
    
    def covariant_derivative(self, tensor_obj):
        """
        Skeleton method.
        Computes the covariant derivative (Nabla_mu T...) of a given tensor,
        adding or subtracting connection terms based on the tensor's index configuration.
        """
        raise NotImplementedError("Covariant derivative logic must be implemented.")
import symengine as se
from einsteinengine.symbolic.tensor import BaseRelativityTensor

class Tetrad(BaseRelativityTensor):
    def __init__(self, arr, syms, config="lu", name="Tetrad", verbose=False):
        """
        Initializes a Tetrad (Vierbein) object.
        Default configuration 'lu' implies: index 0 is curved (lower), index 1 is flat (upper).
        """
        super().__init__(arr, syms, config=config, name=name, verbose=verbose)
        
    def get_inverse(self):
        r"""
        Computes the inverse tetrad e^\mu_a.
        Returns a new Tetrad object with configuration 'ul' (curved upper, flat lower).
        """
        # Matrix inversion of the nested list
        mat = se.Matrix(self._data)
        inv_mat = mat.inv()
        
        # The inverse matrix transposes the index positioning implicitly: e^\mu_a
        inv_data = [[inv_mat[i, j] for j in range(self.dims)] for i in range(self.dims)]
        
        return Tetrad(inv_data, self.syms, config="ul", name=f"Inverse_{self.name}", verbose=self.verbose)

    @classmethod
    def from_diagonal_metric(cls, metric, signs=None, verbose=False):
        r"""
        Factory method to automatically construct a Tetrad from a diagonal metric tensor.
        g_\mu\nu = e_\mu^a e_\nu^b \eta_{ab}
        """
        if verbose:
            print(f"Extracting diagonal Tetrad from metric '{metric.name}'...")
            
        g_data = metric.get_raw_data()
        dims = metric.dims
        syms = metric.syms
        
        # Initialize an empty 4x4 matrix for e_\mu^a
        e_data = [[se.sympify(0) for _ in range(dims)] for _ in range(dims)]
        
        # Default Minkowski signature: (-1, 1, 1, 1)
        if signs is None:
            signs = [-1] + [1] * (dims - 1)
            
        for i in range(dims):
            # For diagonal metrics, e_\mu^a = sqrt(|g_\mu\mu|)
            val = se.sqrt(se.Abs(g_data[i][i]))
            e_data[i][i] = val
            
        return cls(e_data, syms, config="lu", name=f"Tetrad_{metric.name}", verbose=verbose)
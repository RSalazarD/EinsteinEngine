import symengine as se
import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError

from einsteinengine.symbolic.tensor import BaseRelativityTensor


class MetricTensor(BaseRelativityTensor):
    r"""
    Class representing the Metric Tensor ($g_{\mu\nu}$)
    Inherits from BaseRelativityTensor
    """

    def __init__(self, arr, syms, config="ll", name="MetricTensor", verbose=False):
        # Call the parent constructor (tensor.py) to handle the heavy lifting
        # setting up indices, names, and converting everything to SymEngine
        super().__init__(arr, syms, config=config, name=name, verbose=verbose)

        if verbose:
            print(f"[{self.name}] Métrica validada y lista para cálculos.")

        self._validate_metric_structure()
        if not self.is_symmetric():
            raise ValueError("Metric tensor must be symmetric for a standard pseudo-Riemannian metric.")

        self.dims = len(syms)

        # Cache interna para la inversa, para no recalcularla múltiples veces
        self._inverse_tensor = None

    def _validate_metric_structure(self):
        """Validate that the provided data is a well-formed square matrix."""
        if not isinstance(self._data, list) or not self._data:
            raise ValueError("Metric must be provided as a non-empty nested list.")

        if not isinstance(self._data[0], list):
            raise ValueError("Metric must be a square matrix represented as a nested list.")

        rows = len(self._data)
        row_length = len(self._data[0])

        if any(not isinstance(row, list) for row in self._data):
            raise ValueError("Metric must be a square matrix represented as a nested list.")

        if any(len(row) != row_length for row in self._data):
            raise ValueError("Metric must be a square matrix with consistent row lengths.")

        if rows != row_length:
            raise ValueError("Metric must be a square matrix.")

        if rows != self.dims:
            raise ValueError("Metric dimension does not match the number of coordinate symbols.")

    def is_symmetric(self):
        """Return whether the metric is symmetric up to symbolic simplification."""
        for i in range(self.dims):
            for j in range(i + 1, self.dims):
                lhs = sp.simplify(sp.sympify(self._data[i][j]))
                rhs = sp.simplify(sp.sympify(self._data[j][i]))
                if lhs != rhs:
                    return False
        return True

    def inv(self):
        """
        Computes the inverse metric tensor (g^{mu nu}).
        Uses a caching mechanism to avoid redundant calculations.
        """
        # If already computed, return it instantly
        if self._inverse_tensor is not None:
            return self._inverse_tensor

        # If not, run the SymPy inversion pipeline
        sp_matrix = sp.Matrix(self._data)
        dims = sp_matrix.shape[0]

        # Handle cases where SymEngine structures read the data as a flat 1D column
        if sp_matrix.shape == (dims * dims, 1):
            sp_matrix = sp_matrix.reshape(dims, dims)

        determinant = sp.simplify(sp_matrix.det())
        if determinant == 0:
            raise ValueError("Metric matrix must be invertible to compute its inverse.")

        try:
            sp_inv = sp_matrix.inv()
        except NonInvertibleMatrixError as exc:
            raise ValueError("Metric matrix must be invertible to compute its inverse.") from exc

        # Safely parse the SymPy Matrix back into standard nested lists
        inv_arr = sp_inv.tolist()

        # Store the instance before returning it
        self._inverse_tensor = self.__class__(
            inv_arr,
            self.syms,
            config="uu",
            name=f"{self.name}_inv",
            verbose=False
        )

        return self._inverse_tensor
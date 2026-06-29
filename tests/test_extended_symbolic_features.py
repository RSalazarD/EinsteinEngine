import sympy as sp

from einsteinengine.symbolic.christoffel import ChristoffelSymbols
from einsteinengine.symbolic.einstein_tensor import EinsteinTensor
from einsteinengine.symbolic.geodesics import Geodesics
from einsteinengine.symbolic.metric import MetricTensor
from einsteinengine.symbolic.ricci_scalar import RicciScalar
from einsteinengine.symbolic.ricci_tensor import RicciTensor
from einsteinengine.symbolic.riemann import RiemannCurvatureTensor
from einsteinengine.symbolic.spin_connection import SpinConnection
from einsteinengine.symbolic.tensor import BaseRelativityTensor
from einsteinengine.symbolic.tetrad import Tetrad
from einsteinengine.symbolic.weyl import WeylTensor


def test_metric_inverse_and_component_access():
    x, y = sp.symbols("x y", real=True)
    metric = MetricTensor([[1, 0], [0, 1]], [x, y], name="Flat2D")

    inverse_metric = metric.inv()

    assert inverse_metric.get_component(0, 0) == 1
    assert inverse_metric.get_component(1, 1) == 1
    assert metric.get_component(0, 1) == 0


def test_tensor_component_access_and_contraction():
    x, y = sp.symbols("x y", real=True)
    tensor = BaseRelativityTensor([[1, 2], [3, 4]], [x, y], config="ul", name="TestTensor")

    assert tensor.get_component(0, 1) == 2
    assert tensor.get_non_zero()[0][0] == [0, 0]

    contracted = tensor.contract_indices(0, 1)
    assert contracted.get_component() == 5


def test_schwarzschild_christoffel_and_riemann_components():
    t, r, theta, phi = sp.symbols("t r theta phi", real=True)
    M = sp.symbols("M", real=True, positive=True)
    syms = [t, r, theta, phi]

    g_schwarzschild = [
        [-(1 - 2 * M / r), 0, 0, 0],
        [0, 1 / (1 - 2 * M / r), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta) ** 2],
    ]
    metric = MetricTensor(g_schwarzschild, syms, name="Schwarzschild")

    christoffel = ChristoffelSymbols.from_metric(metric, verbose=False)
    gamma_r_phi_phi = sp.simplify(sp.sympify(christoffel.get_raw_data()[1][3][3]))
    expected = sp.sympify("(2*M - r) * sin(theta)**2")
    assert sp.simplify(gamma_r_phi_phi - expected) == 0

    riemann = RiemannCurvatureTensor.from_metric(metric, verbose=False)
    component = sp.simplify(sp.sympify(riemann.get_raw_data()[1][0][1][0]))
    expected = sp.sympify("2*M*(2*M - r)/r**4")
    assert sp.simplify(component - expected) == 0


def test_ricci_tensor_scalar_and_einstein_tensor_for_schwarzschild():
    t, r, theta, phi = sp.symbols("t r theta phi", real=True)
    M = sp.symbols("M", real=True, positive=True)
    syms = [t, r, theta, phi]

    g_schwarzschild = [
        [-(1 - 2 * M / r), 0, 0, 0],
        [0, 1 / (1 - 2 * M / r), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta) ** 2],
    ]
    metric = MetricTensor(g_schwarzschild, syms, name="Schwarzschild")

    riemann = RiemannCurvatureTensor.from_metric(metric, verbose=False)
    ricci_tensor = RicciTensor.from_riemann(riemann, verbose=False)
    ricci_scalar = RicciScalar.from_ricci_tensor_and_metric(ricci_tensor, metric, verbose=False)
    einstein_tensor = EinsteinTensor.from_metric(metric, verbose=False)

    assert sp.simplify(ricci_tensor.get_component(0, 0)) == 0
    assert sp.simplify(ricci_scalar.get_component()) == 0
    assert sp.simplify(einstein_tensor.get_component(0, 0)) == 0


def test_weyl_tensor_is_zero_for_flat_metric():
    t, x, y, z = sp.symbols("t x y z", real=True)
    metric = MetricTensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [t, x, y, z], name="Minkowski")

    weyl = WeylTensor.from_metric(metric, verbose=False)
    component = sp.simplify(sp.sympify(weyl.get_raw_data()[0][1][0][1]))

    assert component == 0


def test_geodesics_equations_for_flat_metric():
    t, x = sp.symbols("t x", real=True)
    metric = MetricTensor([[-1, 0], [0, 1]], [t, x], name="Minkowski2D")
    christoffel = ChristoffelSymbols.from_metric(metric, verbose=False)
    geodesics = Geodesics(christoffel, param_name="tau", verbose=False)

    equations = geodesics.get_equations(substitute_velocities=True, simplify=False)

    assert len(equations) == 2
    assert sp.simplify(sp.sympify(equations[0])) == 0
    assert sp.simplify(sp.sympify(equations[1])) == 0


def test_tetrad_and_spin_connection_for_flat_metric():
    t, x, y, z = sp.symbols("t x y z", real=True)
    metric = MetricTensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [t, x, y, z], name="Minkowski")

    tetrad = Tetrad.from_diagonal_metric(metric, verbose=False)
    inverse_tetrad = tetrad.get_inverse()
    christoffel = ChristoffelSymbols.from_metric(metric, verbose=False)
    spin_connection = SpinConnection.from_tetrad_and_christoffel(tetrad, christoffel, verbose=False)

    assert tetrad.get_component(0, 0) == 1
    assert inverse_tetrad.get_component(0, 0) == 1
    assert sp.simplify(sp.sympify(spin_connection.get_raw_data()[0][0][0])) == 0

import sympy as sp
import symengine as se
from einsteinengine.symbolic.metric import MetricTensor
from einsteinengine.symbolic.christoffel import ChristoffelSymbols
from einsteinengine.symbolic.riemann import RiemannCurvatureTensor

def get_schwarzschild_setup():
    """Helper function to provide the metric and symbols for all tests."""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    syms = [t, r, theta, phi]

    g_schwarzschild = [
        [-(1 - 2*M/r), 0, 0, 0],
        [0, 1/(1 - 2*M/r), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta)**2] # type: ignore
    ]
    metric = MetricTensor(g_schwarzschild, syms, name="Schwarzschild")
    return metric, M, r, theta

def test_christoffel_radial_angular():
    """
    Tests the Christoffel symbol Gamma^r_{phi phi}.
    In Schwarzschild, this governs the centrifugal force in the radial equation.
    Theoretical result: -(r - 2*M) * sin(theta)**2
    """
    metric, M, r, theta = get_schwarzschild_setup()
    
    # Calculate Christoffel Symbols
    christoffel = ChristoffelSymbols.from_metric(metric, verbose=False)
    
    # Extract Gamma^r_{phi phi} (indices 1, 3, 3)
    raw_component = christoffel.get_raw_data()[1][3][3]
    gamma_r_phiphi = sp.simplify(sp.sympify(raw_component))
    
    expected = sp.sympify("-(r - 2*M) * sin(theta)**2")
    
    assert sp.simplify(gamma_r_phiphi - expected) == 0

def test_riemann_curvature_component():
    """
    Tests the Riemann Curvature Tensor component R^r_{trt}.
    Theoretical result: 2*M*(2*M - r)/r**4
    """
    metric, M, r, theta = get_schwarzschild_setup()
    
    # Calculate Riemann Tensor
    riemann = RiemannCurvatureTensor.from_metric(metric, verbose=False)
    
    # Extract R^r_{trt} (indices 1, 0, 1, 0)
    raw_component = riemann.get_raw_data()[1][0][1][0]
    r_r_trt = sp.simplify(sp.sympify(raw_component))
    
    expected = sp.sympify("2*M*(2*M - r)/r**4")
    
    assert sp.simplify(r_r_trt - expected) == 0


def test_tensor_string_rendering_modes():
    """Objects should support both rich LaTeX-style output and a plain-text fallback."""
    metric, _, _, _ = get_schwarzschild_setup()

    latex_output = metric.to_string(format="latex")
    plain_output = metric.to_string(format="plain")

    assert latex_output != plain_output
    assert "\\begin" in latex_output or "\\left" in latex_output
    assert isinstance(plain_output, str)
    assert len(plain_output) > 0


def test_component_access_fast_path_returns_raw_value():
    """The no-simplification path should preserve the raw underlying expression."""
    metric, _, _, _ = get_schwarzschild_setup()

    component = metric.get_component(0, 0, simplify=False)

    assert component == metric.get_raw_data()[0][0]


def test_configurable_simplification_modes():
    """Objects should support lightweight and full simplification strategies."""
    metric, _, _, _ = get_schwarzschild_setup()

    light_copy = metric.simplify(in_place=False, mode="light")
    full_copy = metric.simplify(in_place=False, mode="full")

    assert light_copy.dims == metric.dims
    assert full_copy.dims == metric.dims
    assert light_copy.name == metric.name
    assert full_copy.name == metric.name
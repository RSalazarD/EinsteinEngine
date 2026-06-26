import sympy as sp
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
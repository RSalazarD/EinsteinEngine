import sympy as sp
from einsteinengine.symbolic.metric import MetricTensor
from einsteinengine.symbolic.weyl import WeylTensor

def test_weyl_radial_tidal_force():
    """
    Tests that the Weyl tensor correctly computes the radial tidal force
    for a Schwarzschild black hole, yielding 2*M*(2*M - r)/r**4.
    """
    # 1. SETUP: Prepare the symbols and the metric
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

    # 2. ACT: Compute the Weyl Tensor
    # We turn verbose=False to keep the test terminal clean
    weyl = WeylTensor.from_metric(metric, verbose=False)
    
    # Extract component C^r_trt (Indices 1, 0, 1, 0)
    raw_component = weyl.get_raw_data()[1][0][1][0]
    c_r_trt = sp.simplify(sp.sympify(raw_component))

    # 3. ASSERT: Verify the result exactly matches theoretical physics
    expected_result = sp.sympify("2*M*(2*M - r)/r**4")
    
    # The difference between the engine result and the theory must be exactly 0
    difference = sp.simplify(c_r_trt - expected_result)
    
    assert difference == 0, f"Expected {expected_result}, but got {c_r_trt}"
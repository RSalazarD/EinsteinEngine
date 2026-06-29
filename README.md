# EinsteinEngine

EinsteinEngine is a symbolic tensor library for General Relativity built around a simple idea: make tensor calculations faster and more structured without giving up exact symbolic mathematics.

The project combines Python with SymEngine for the heavy symbolic work, and exposes an object-oriented API for common relativistic objects such as metrics, Christoffel symbols, curvature tensors, Ricci tensors, Einstein tensors, geodesics, tetrads and spin connections.

## What the library can do today

EinsteinEngine currently provides a practical toolkit for symbolic GR workflows, including:

- Metric tensor construction and validation
- Metric inversion for diagonal and standard symbolic cases
- Christoffel symbol computation
- Riemann tensor construction
- Ricci tensor and Ricci scalar computation
- Einstein tensor construction
- Geodesic equation generation
- Tetrad and spin connection support
- Tensor component access, contraction, index raising/lowering and basic tensor arithmetic
- Rich LaTeX-style rendering for notebooks and a plain-text fallback for lighter output

The focus of the library is not only correctness, but also performance in symbolic pipelines that would otherwise become very expensive in pure SymPy.

## Installation

From PyPI:

```bash
pip install einsteinengine
```

From source:

```bash
git clone https://github.com/RSalazarD/EinsteinEngine.git
cd EinsteinEngine
pip install -e .
```

## Quick start

```python
import sympy as sp
from einsteinengine.symbolic.metric import MetricTensor
from einsteinengine.symbolic.riemann import RiemannCurvatureTensor

# Coordinates and parameters
t, r, theta, phi = sp.symbols('t r theta phi', real=True)
M = sp.symbols('M', real=True)

# Schwarzschild metric
g_schwarzschild = [
    [-(1 - 2*M/r), 0, 0, 0],
    [0, 1/(1 - 2*M/r), 0, 0],
    [0, 0, r**2, 0],
    [0, 0, 0, r**2 * sp.sin(theta)**2],
]

metric = MetricTensor(g_schwarzschild, [t, r, theta, phi], name="Schwarzschild")
riemann = RiemannCurvatureTensor.from_metric(metric, verbose=False)

print(riemann.get_component(1, 0, 1, 0))
```

A typical output for the Schwarzschild case is the known exact component:

```python
2*M*(2*M - r)/r**4
```

## Current strengths

EinsteinEngine is particularly useful when you want to:

- build symbolic curvature tensors from a metric,
- test GR expressions quickly in notebooks or scripts,
- compare symbolic performance against pure SymPy,
- explore exact tensor structures without writing all the algebra by hand.

## Benchmarks

A simple benchmark script is available in the benchmarks folder to compare EinsteinEngine against pure SymPy on a representative symbolic workload.

Run it from the project root with:

```bash
python benchmarks/compare_sympy_vs_einsteinengine.py
```

The script prints timing information for both engines and a rough speedup estimate.

## Testing

The project includes a pytest suite covering core computations such as metric handling, curvature tensors, geodesics and rendering behaviour.

Run tests with:

```bash
pytest
```

## Roadmap and future directions

This project is already usable for a broad set of symbolic GR tasks, but there is still room to grow. Possible future work includes:

- more optimized tensor contractions and higher-rank workflows,
- broader support for non-diagonal and more general metrics,
- improved simplification strategies with configurable modes,
- additional GR utilities such as more advanced invariants and field equations,
- better documentation and more benchmark examples.

The goal is to keep the library practical today while expanding toward a more complete symbolic GR toolkit over time.

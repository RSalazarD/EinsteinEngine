#  EinsteinEngine

**A high-performance, object-oriented symbolic tensor engine for General Relativity, powered by Python and C++.**

EinsteinEngine is designed to solve the performance bottlenecks of traditional pure-Python symbolic calculators. By wrapping `SymEngine` (C++) backend inside a Python API, it computes Christoffel Symbols, Riemann Tensors, and other complex relativistic structures faster than standard pure Python libraries.

---

## Key Features

* **⚡ C++ Backend:** Mathematical heavy-lifting (partial derivatives, massive tensor contractions) is routed directly to `SymEngine`, bypassing Python's native performance limits.
* **🧠 Smart Memoization:** Built-in memory caching prevents redundant calculations of highly complex objects like inverse metric tensors.
* **📦 Clean Object-Oriented API:** Complex tensor pipelines are reduced to a few lines of readable code using class inheritance.
* **🛡️ Exact Mathematics:** Built to handle rational numbers securely, preventing floating-point contamination and ensuring textbook-perfect algebraic simplifications.

---

## Quick Start

EinsteinEngine calculates the entire Riemann Curvature Tensor of a Black Hole in just two lines of code:

```python
import sympy as sp
from einsteinpy.symbolic.metric import MetricTensor
from einsteinpy.symbolic.riemann import RiemannCurvatureTensor

# 1. Define your symbols and metric array
t, r, theta, phi = sp.symbols('t r theta phi', real=True)
M = sp.symbols('M', real=True)

g_schwarzschild = [
    [-(1 - 2*M/r), 0, 0, 0],
    [0, 1/(1 - 2*M/r), 0, 0],
    [0, 0, r**2, 0],
    [0, 0, 0, r**2 * sp.sin(theta)**2]
]

# 2. Run the EinsteinEngine Pipeline
metric = MetricTensor(g_schwarzschild, [t, r, theta, phi], name="Schwarzschild")
riemann = RiemannCurvatureTensor.from_metric(metric)

# 3. Extract exact, simplified textbook results
print(riemann.get_component(1, 0, 1, 0))
# Output: 2*M*(2*M - r)/r**4

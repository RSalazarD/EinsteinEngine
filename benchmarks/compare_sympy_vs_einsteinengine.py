import sys
import time
from pathlib import Path

import sympy as sp
from sympy import Matrix


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "einsteinengine").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer the installed package if available, otherwise fall back to the local source tree.
try:
    import einsteinengine  # noqa: F401
except ImportError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from einsteinengine.symbolic.metric import MetricTensor
from einsteinengine.symbolic.christoffel import ChristoffelSymbols


def benchmark_case():
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True)

    g_schwarzschild = [
        [-(1 - 2*M/r), 0, 0, 0],
        [0, 1/(1 - 2*M/r), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta)**2],
    ]

    # SymPy baseline using direct matrix operations and symbolic differentiation
    start = time.perf_counter()
    g_sp = Matrix(g_schwarzschild)
    g_inv_sp = g_sp.inv()
    gamma_sp = []
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                total = sp.Integer(0)
                for sigma in range(4):
                    term1 = sp.diff(g_sp[nu, sigma], r)
                    term2 = sp.diff(g_sp[mu, sigma], r)
                    term3 = sp.diff(g_sp[mu, nu], r)
                    total += g_inv_sp[lam, sigma] * (term1 + term2 - term3)
                gamma_sp.append(sp.simplify(total / 2))
    sympy_time = time.perf_counter() - start

    # EinsteinEngine path
    start = time.perf_counter()
    metric = MetricTensor(g_schwarzschild, [t, r, theta, phi], name="Schwarzschild")
    christoffel = ChristoffelSymbols.from_metric(metric, verbose=False)
    ee_time = time.perf_counter() - start

    return sympy_time, ee_time


def main():
    print("Running a balanced symbolic benchmark...")
    print("This compares a medium-size Christoffel-symbol workload for SymPy vs EinsteinEngine.\n")

    times = []
    for i in range(3):
        times.append(benchmark_case())

    sympy_times = [t[0] for t in times]
    ee_times = [t[1] for t in times]

    avg_sympy = sum(sympy_times) / len(sympy_times)
    avg_ee = sum(ee_times) / len(ee_times)

    print(f"Run 1: SymPy={sympy_times[0]:.4f}s | EinsteinEngine={ee_times[0]:.4f}s")
    print(f"Run 2: SymPy={sympy_times[1]:.4f}s | EinsteinEngine={ee_times[1]:.4f}s")
    print(f"Run 3: SymPy={sympy_times[2]:.4f}s | EinsteinEngine={ee_times[2]:.4f}s")
    print("-" * 50)
    print(f"Average: SymPy={avg_sympy:.4f}s | EinsteinEngine={avg_ee:.4f}s")
    print(f"Speedup: {avg_sympy / avg_ee:.2f}x faster")


if __name__ == "__main__":
    main()

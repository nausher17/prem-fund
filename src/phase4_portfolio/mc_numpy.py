"""NumPy Monte Carlo reference implementation (correctness baseline).

Simulates correlated annual club returns r ~ N(mu, Sigma) via Cholesky over
a 5-year horizon, compounding portfolio value. The C++ engine must reproduce
these quantiles within MC error before its speed matters.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "outputs" / "phase4"
HORIZON = 5
SEED = 42


def load():
    er = pd.read_csv(OUT / "expected_returns.csv", index_col=0).exp_return.values
    cov = pd.read_csv(OUT / "covariance.csv").values
    w = pd.read_csv(OUT / "weights.csv", index_col=0).max_sharpe.values
    return er, cov, w


def simulate(er, cov, w, n_paths: int, seed: int = SEED) -> np.ndarray:
    """Same algebraic reduction as the C++ engine: portfolio return
    w.(mu + Lz) = w.mu + (L^T w).z — distributionally exact."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    mu_p = float(er @ w)
    lw = L.T @ w
    value = np.ones(n_paths)
    for _ in range(HORIZON):
        z = rng.standard_normal((n_paths, len(er)))
        value *= 1.0 + mu_p + z @ lw
    return value


def summary(v: np.ndarray) -> dict:
    return {"mean": v.mean(), "p5": np.quantile(v, 0.05),
            "p50": np.quantile(v, 0.50), "p95": np.quantile(v, 0.95),
            "prob_loss": float((v < 1).mean()),
            "irr_p50": np.quantile(v, 0.50) ** (1 / HORIZON) - 1}


def main(n_paths: int = 200_000) -> dict:
    er, cov, w = load()
    t0 = time.perf_counter()
    v = simulate(er, cov, w, n_paths)
    dt = time.perf_counter() - t0
    s = summary(v)
    s["seconds"] = dt
    s["paths"] = n_paths
    print({k: round(float(x), 4) for k, x in s.items()})
    return s


if __name__ == "__main__":
    main()

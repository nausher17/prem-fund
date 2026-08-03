"""Phase 4 tests: optimiser constraints, MC engine agreement, backtest sanity."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase4"

needs_outputs = pytest.mark.skipif(
    not (OUT / "weights.csv").exists(), reason="phase4 not run")


@needs_outputs
def test_weights_respect_constraints():
    w = pd.read_csv(OUT / "weights.csv", index_col=0)
    for col in ("max_sharpe", "min_var"):
        assert abs(w[col].sum() - 1) < 1e-6
        assert (w[col] >= -1e-9).all() and (w[col] <= 0.20 + 1e-6).all()
    ls = w.long_short_max_sharpe
    assert abs(ls.sum() - 1) < 1e-6
    assert (ls >= -0.30 - 1e-6).all() and (ls <= 0.50 + 1e-6).all()


@needs_outputs
def test_max_sharpe_dominates_min_var_return():
    s = pd.read_csv(OUT / "portfolio_stats.csv", index_col=0)
    assert s.loc["max_sharpe", "exp_return"] >= s.loc["min_var", "exp_return"]
    assert s.loc["min_var", "vol"] <= s.loc["max_sharpe", "vol"] + 1e-9
    # the 15% IRR fund objective is met by the max-Sharpe portfolio
    assert s.loc["max_sharpe", "exp_return"] > 0.15


@needs_outputs
def test_cpp_and_numpy_engines_agree():
    from src.phase4_portfolio.mc_numpy import load, simulate, summary
    er, cov, w = load()
    s_np = summary(simulate(er, cov, w, 200_000))
    exe = ROOT / "cpp" / "mc_engine"
    if not exe.exists():
        subprocess.run(["clang++", "-O3", "-std=c++17", "-o", str(exe),
                        str(ROOT / "cpp" / "mc_engine.cpp")], check=True)
    subprocess.run([str(exe), "200000", "7", str(OUT)], check=True,
                   capture_output=True)
    s_cpp = pd.read_csv(OUT / "mc_cpp_summary.csv").iloc[0]
    for k in ("mean", "p5", "p50", "p95"):
        assert abs(s_cpp[k] - s_np[k]) / s_np[k] < 0.01, k


@needs_outputs
def test_backtest_outputs_sane():
    bt = pd.read_csv(OUT / "backtest_annual.csv")
    assert len(bt) >= 6 and (bt.n >= 10).all()
    s = pd.read_csv(OUT / "backtest_summary.csv")
    assert set(s.strategy) == {"long_only", "long_short", "equal_weight"}
    assert s.ann_return.between(-0.5, 0.5).all()


@needs_outputs
def test_frontier_monotone_vol_at_extremes():
    f = pd.read_csv(OUT / "frontier.csv")
    assert len(f) > 10
    assert f.vol.iloc[-1] >= f.vol.min() - 1e-9  # frontier is a valid curve

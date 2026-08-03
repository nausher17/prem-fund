"""Phase 4: expected returns, covariance, optimisation, risk.

Asset universe: the 20 PL clubs of 2023-24 (the clubs we can value with
financials). Returns are MARK-TO-MODEL: club "prices" are Transfermarkt
squad values (the only consistently observable cross-section), scaled to a
club-EV proxy so that the league median matches our comps-based EV. This is
an honest limitation, stated everywhere: clubs do not trade annually, so no
transaction-price return series exists.

Expected return per club = valuation-gap convergence + market drift:
    upside   = blend_value / market_proxy - 1        (Phase 2 model value)
    E[r]     = (1 + upside)^(1/5) - 1 + drift        (5-year convergence,
               assumption documented in assumptions note below)
    drift    = league median annual squad-value growth (panel)

Covariance: club annual log squad-value growth (9 seasons), Ledoit-Wolf
shrinkage (sklearn) — the panel is far too short for a sample covariance of
20 assets (9 obs), which is exactly what shrinkage is for.

Optimisation (scipy SLSQP): long-only max-Sharpe and min-variance with a
20% position cap; a 130/30-style long/short variant (weights in [-0.3, 0.5],
net 100%). Risk-free = latest season gilt yield.

Risk: parametric (normal) and historical VaR/CVaR at 95/99 on one-year
portfolio returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
P2_OUT = PROJECT_ROOT / "outputs" / "phase2"
OUT = PROJECT_ROOT / "outputs" / "phase4"

CONVERGENCE_YEARS = 5  # valuation gaps close over ~one ownership cycle
POSITION_CAP = 0.20


def load_inputs():
    panel = pd.read_csv(PROCESSED / "panel.csv")
    pl = panel[panel.league == "premier-league"].copy()
    pl["year"] = pl.season.str.slice(0, 4).astype(int)
    vals = pd.read_csv(P2_OUT / "valuations.csv")
    return pl, vals


def growth_matrix(pl: pd.DataFrame, clubs: list[str]) -> pd.DataFrame:
    """Club x season log squad-value growth (mark-to-model returns)."""
    g = pl.pivot_table(index="year", columns="canonical",
                       values="total_market_value_eur_m")
    g = np.log(g).diff().dropna(how="all")
    return g[clubs]


def _fill_growth(G: pd.DataFrame) -> pd.DataFrame:
    """Missing club-years (Championship spells; short-history clubs like
    Luton) take the league's cross-sectional mean return for that year —
    the least-informative defensible fill for a covariance estimate."""
    year_mean = G.mean(axis=1)
    return G.apply(lambda col: col.fillna(year_mean)).fillna(0.0)


def expected_returns(pl: pd.DataFrame, vals: pd.DataFrame) -> pd.DataFrame:
    latest = pl[pl.season == "2023-2024"].set_index("canonical")
    eur_gbp = latest.eur_gbp_season.iloc[0]
    sq_gbp = latest.total_market_value_eur_m * eur_gbp  # squad value in GBP m
    v = vals.set_index("club")
    # scale squad value to an EV proxy so league medians match comps EV
    scale = (v.comps_value_gbp_m.median() / sq_gbp.loc[v.index].median())
    market_proxy = sq_gbp.loc[v.index] * scale
    upside = v.blend_value_gbp_m / market_proxy - 1
    drift = (pl.groupby("canonical").total_market_value_eur_m
             .apply(lambda s: np.log(s.iloc[-1] / s.iloc[0]) / max(len(s) - 1, 1))
             .median())
    er = (1 + upside) ** (1 / CONVERGENCE_YEARS) - 1 + drift
    return pd.DataFrame({"market_proxy_gbp_m": market_proxy.round(0),
                         "blend_value_gbp_m": v.blend_value_gbp_m,
                         "upside": upside.round(3),
                         "exp_return": er.round(4)})


def optimise(er: pd.Series, cov: np.ndarray, rf: float,
             long_only: bool = True) -> dict[str, np.ndarray]:
    n = len(er)
    bounds = [(0.0, POSITION_CAP)] * n if long_only else [(-0.30, 0.50)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    x0 = np.full(n, 1 / n)

    def neg_sharpe(w):
        ret = w @ er
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol

    def variance(w):
        return w @ cov @ w

    out = {}
    for name, fun in (("max_sharpe", neg_sharpe), ("min_var", variance)):
        res = minimize(fun, x0, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 1000})
        if not res.success:
            raise RuntimeError(f"{name} optimisation failed: {res.message}")
        out[name] = res.x
    return out


def frontier(er: pd.Series, cov: np.ndarray, n_points: int = 25) -> pd.DataFrame:
    n = len(er)
    targets = np.linspace(er.min(), er.max(), n_points)
    rows = []
    for t in targets:
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=t: w @ er - t}]
        res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n),
                       bounds=[(0.0, POSITION_CAP)] * n, constraints=cons,
                       method="SLSQP", options={"maxiter": 1000})
        if res.success:
            rows.append({"target_return": t,
                         "vol": float(np.sqrt(res.x @ cov @ res.x))})
    return pd.DataFrame(rows)


def var_cvar(returns: np.ndarray, levels=(0.95, 0.99)) -> dict:
    out = {}
    mu, sd = returns.mean(), returns.std(ddof=1)
    from scipy.stats import norm
    for lv in levels:
        q = np.quantile(returns, 1 - lv)
        out[f"hist_var_{int(lv*100)}"] = -q
        out[f"hist_cvar_{int(lv*100)}"] = -returns[returns <= q].mean()
        z = norm.ppf(1 - lv)
        out[f"param_var_{int(lv*100)}"] = -(mu + z * sd)
        out[f"param_cvar_{int(lv*100)}"] = -(mu - sd * norm.pdf(z) / (1 - lv))
    return {k: round(float(v), 4) for k, v in out.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pl, vals = load_inputs()
    er_tab = expected_returns(pl, vals)
    clubs = er_tab.index.tolist()
    G = growth_matrix(pl, clubs)
    lw = LedoitWolf().fit(_fill_growth(G).values)
    cov = lw.covariance_
    rf = pl[pl.season == "2023-2024"].uk_gilt_10y_season.iloc[0] / 100

    er = er_tab.exp_return
    ws = optimise(er, cov, rf, long_only=True)
    ws_ls = optimise(er, cov, rf, long_only=False)
    weights = pd.DataFrame({
        "max_sharpe": ws["max_sharpe"], "min_var": ws["min_var"],
        "long_short_max_sharpe": ws_ls["max_sharpe"]}, index=clubs)
    # full precision: constraint checks re-run from this artifact (R + tests)

    stats = {}
    for name in weights.columns:
        w = weights[name].values
        ret, vol = float(w @ er), float(np.sqrt(w @ cov @ w))
        port_hist = _fill_growth(G).values @ w
        stats[name] = {"exp_return": round(ret, 4), "vol": round(vol, 4),
                       "sharpe": round((ret - rf) / vol, 3),
                       **var_cvar(port_hist)}

    er_tab.to_csv(OUT / "expected_returns.csv")
    weights.to_csv(OUT / "weights.csv")
    frontier(er, cov).to_csv(OUT / "frontier.csv", index=False)
    pd.DataFrame(stats).T.to_csv(OUT / "portfolio_stats.csv")
    np.savetxt(OUT / "covariance.csv", cov, delimiter=",",
               header=",".join(clubs), comments="")
    print(er_tab.sort_values("upside", ascending=False).to_string())
    print("\n", pd.DataFrame(stats).T.to_string())
    print(f"\nLedoit-Wolf shrinkage: {lw.shrinkage_:.3f}; rf={rf:.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

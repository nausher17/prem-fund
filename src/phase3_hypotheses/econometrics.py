"""Panel econometrics for H1-H4.

All specifications share the design rules (README methodology):
- club (entity) fixed effects absorb time-invariant club heterogeneity
  (brand, stadium, catchment) — identification comes from within-club
  variation over seasons;
- standard errors clustered by club (serial correlation within club);
- season time effects where the spec is not already differenced;
- predictors lagged one season wherever the hypothesis is causal
  (H4: age/performance in t-1 -> revenue growth in t), so regressors are
  in the information set before the outcome realises — the econometric
  mirror of the ML suite's temporal splits;
- the COI flag is never used to exclude: headline specs are re-run with
  coi_flag rows dropped as a robustness column.

Hypotheses -> specifications
  H1 (promotion overvaluation): forward squad-value growth (t -> t+1) on
      newly_promoted(t), controlling for current performance. If markets
      overweight the promotion windfall, promoted clubs' market values
      subsequently underperform: beta < 0. Entity FE + time effects.
  H2 (MCO premium): log squad value on mco_flag + performance controls.
      FE identifies WITHIN-club value shifts around joining an MCO group
      (time-invariant MCO members like Man City/Watford drop out of the
      within estimator — stated power limitation).
  H3 (UCL optionality): log squad value on ucl_spot(t-1) and position
      volatility interaction: does the market price the convexity (vol x
      near-the-money) beyond the linear qualification effect?
  H4 (age/performance -> revenue growth): revenue growth (t) on lagged
      points-per-game, lagged minutes-weighted age, lagged squad value.
      PL clubs only (revenue coverage); strongest-expected result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "outputs" / "phase3"


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PROCESSED / "panel.csv")
    panel["year"] = panel.season.str.slice(0, 4).astype(int)
    # forward (t -> t+1) squad value growth for H1: value_{t+1}/value_t - 1,
    # only defined when the club is in the panel the following season
    panel = panel.sort_values(["canonical", "year"])
    nxt = panel[["canonical", "year", "total_market_value_eur_m"]].copy()
    nxt["year"] -= 1
    nxt = nxt.rename(columns={"total_market_value_eur_m": "value_next"})
    panel = panel.merge(nxt, on=["canonical", "year"], how="left")
    panel["fwd_value_growth"] = panel.value_next / panel.total_market_value_eur_m - 1
    panel["log_value"] = np.log(panel.total_market_value_eur_m)
    # position volatility (H3): club-level trailing std of position
    panel["pos_vol"] = (panel.groupby("canonical").position
                        .transform(lambda s: s.expanding().std().shift(1)))
    return panel.set_index(["canonical", "year"])


def fit(df: pd.DataFrame, dep: str, exog: list[str],
        time_effects: bool = True, drop_coi: bool = False) -> dict:
    d = df.dropna(subset=[dep] + exog)
    if drop_coi:
        d = d[d.coi_flag == 0]
    mod = PanelOLS(d[dep], d[exog], entity_effects=True,
                   time_effects=time_effects, check_rank=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return {"res": res, "n": int(res.nobs), "d": d}


def run_all() -> pd.DataFrame:
    panel = load_panel()
    rows = []

    def record(hyp: str, spec: str, dep: str, exog: list[str], focus: str,
               subset: pd.DataFrame | None = None, **kw):
        base = subset if subset is not None else panel
        out = fit(base, dep, exog, **kw)
        r = out["res"]
        rows.append({
            "hypothesis": hyp, "spec": spec, "dep": dep, "focus_var": focus,
            "coef": r.params[focus], "se": r.std_errors[focus],
            "t": r.tstats[focus], "p": r.pvalues[focus],
            "n": out["n"], "r2_within": r.rsquared_within,
        })
        return r

    # ---- H1: promotion overvaluation --------------------------------------
    record("H1", "fwd value growth ~ promoted + ppg + position", "fwd_value_growth",
           ["newly_promoted", "ppg", "position_pct"], "newly_promoted")
    record("H1", "robustness: ex-COI", "fwd_value_growth",
           ["newly_promoted", "ppg", "position_pct"], "newly_promoted",
           drop_coi=True)

    # ---- H2: MCO premium ---------------------------------------------------
    record("H2", "log value ~ mco + ppg + position", "log_value",
           ["mco_flag", "ppg", "position_pct"], "mco_flag")
    record("H2", "integrated-scope only", "log_value",
           ["mco_integrated", "ppg", "position_pct"], "mco_integrated")
    record("H2", "robustness: ex-COI", "log_value",
           ["mco_flag", "ppg", "position_pct"], "mco_flag", drop_coi=True)

    # ---- H3: UCL optionality ----------------------------------------------
    panel_h3 = panel.copy()
    panel_h3["ucl_x_vol"] = panel_h3.ucl_spot_lag1 * panel_h3.pos_vol
    record("H3", "log value ~ ucl(t-1) + ppg", "log_value",
           ["ucl_spot_lag1", "ppg"], "ucl_spot_lag1", subset=panel_h3)
    record("H3", "convexity: + ucl(t-1) x pos_vol", "log_value",
           ["ucl_spot_lag1", "ucl_x_vol", "ppg"], "ucl_x_vol", subset=panel_h3)

    # ---- H4: age/performance -> revenue growth (PL only) -------------------
    pl = panel[panel.league == "premier-league"]
    record("H4", "rev growth ~ ppg(t-1) + mw_age(t-1)", "revenue_growth_yoy",
           ["ppg_lag1", "minutes_weighted_age_lag1"], "ppg_lag1", subset=pl)
    record("H4", "age focus (same spec)", "revenue_growth_yoy",
           ["ppg_lag1", "minutes_weighted_age_lag1"], "minutes_weighted_age_lag1",
           subset=pl)
    record("H4", "+ lagged value control", "revenue_growth_yoy",
           ["ppg_lag1", "minutes_weighted_age_lag1", "total_market_value_eur_m_lag1"],
           "ppg_lag1", subset=pl)
    record("H4", "robustness: ex-COI", "revenue_growth_yoy",
           ["ppg_lag1", "minutes_weighted_age_lag1"], "ppg_lag1", subset=pl,
           drop_coi=True)

    results = pd.DataFrame(rows)
    for c in ("coef", "se", "t"):
        results[c] = results[c].round(4)
    results["p"] = results["p"].round(4)
    results["r2_within"] = results["r2_within"].round(3)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "econometrics_results.csv", index=False)
    return results


def main() -> int:
    results = run_all()
    print(results.to_string(index=False))
    print(f"\nWrote {OUT / 'econometrics_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

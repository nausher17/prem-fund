"""Rolling annual value backtest (PL clubs, 2016-17..2022-23 signals).

Signal at season t (only information available at t):
    model_value  = revenue(t) x median core comps multiple
    market_proxy = squad value(t) in GBP x league scale (medians matched in t)
    signal       = model_value / market_proxy  (valuation gap)

Portfolios formed each season: LONG top quintile of signal, SHORT bottom
quintile (equal weight within legs); benchmark = equal-weight all clubs.
Realised return = next-season log squad-value growth (mark-to-model — clubs
do not trade; stated prominently).

Survivorship handling: clubs without a t+1 panel row (relegated) are
excluded from that year's legs AND the benchmark — consistent treatment,
documented as a caveat shared with H1. Reported: annual series, geometric
mean (IRR), Sharpe (vs gilt), max drawdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio import load_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs" / "phase4"
MULTIPLE = 2.19  # median core comps multiple (comps.py)


def run() -> pd.DataFrame:
    pl, _ = load_inputs()
    pl = pl.sort_values(["canonical", "year"])
    nxt = pl[["canonical", "year", "total_market_value_eur_m"]].copy()
    nxt["year"] -= 1
    pl = pl.merge(nxt.rename(columns={"total_market_value_eur_m": "value_next"}),
                  on=["canonical", "year"], how="left")
    pl["fwd_growth"] = np.log(pl.value_next / pl.total_market_value_eur_m)

    rows = []
    for year in range(2016, 2023):
        d = pl[(pl.year == year) & pl.revenue_gbp.notna()
               & pl.fwd_growth.notna()].copy()
        if len(d) < 10:
            continue
        sq_gbp = d.total_market_value_eur_m * d.eur_gbp_season
        model = d.revenue_gbp / 1e6 * MULTIPLE
        proxy = sq_gbp * (model.median() / sq_gbp.median())
        d["signal"] = model / proxy
        q_hi, q_lo = d.signal.quantile([0.8, 0.2])
        long = d[d.signal >= q_hi].fwd_growth.mean()
        short = d[d.signal <= q_lo].fwd_growth.mean()
        rows.append({"year": year, "n": len(d),
                     "long": long, "short": short,
                     "long_short": long - short,
                     "long_only": long,
                     "equal_weight": d.fwd_growth.mean(),
                     "rf": d.uk_gilt_10y_season.iloc[0] / 100})
    bt = pd.DataFrame(rows)

    def stats(col: str) -> dict:
        r = bt[col]
        geo = float(np.exp(r.mean()) - 1)  # log returns -> geometric annual
        sharpe = float((r.mean() - np.log(1 + bt.rf).mean()) / r.std(ddof=1))
        curve = np.exp(r.cumsum())
        mdd = float((1 - curve / curve.cummax()).max())
        return {"strategy": col, "ann_return": round(geo, 4),
                "sharpe": round(sharpe, 3), "max_drawdown": round(mdd, 4)}

    summary = pd.DataFrame([stats(c) for c in
                            ("long_only", "long_short", "equal_weight")])
    OUT.mkdir(parents=True, exist_ok=True)
    bt.round(4).to_csv(OUT / "backtest_annual.csv", index=False)
    summary.to_csv(OUT / "backtest_summary.csv", index=False)
    print(bt.round(3).to_string(index=False))
    print("\n", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
    sys.exit(0)

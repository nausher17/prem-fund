"""Comparable-transaction valuation: EV/Revenue multiples from comps.csv.

Method: the comps table (9 verified deals, citations inline) yields an
EV/Revenue distribution. Clubs are banded by status at the valuation date:

  trophy band   big-six or UCL state       -> 75th percentile multiple
  standard      everything else in the PL  -> median (ex-distressed,
                                              ex-trophy: Chelsea and Villa
                                              excluded from the base median
                                              as documented outliers)
  distressed    relegated at valuation     -> 25th percentile multiple

Low/mid/high columns expose the multiple uncertainty rather than hiding it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

BIG_SIX = ["Arsenal", "Chelsea", "Liverpool", "Manchester City",
           "Manchester United", "Tottenham Hotspur"]


def load_multiples() -> dict[str, float]:
    comps = pd.read_csv(HERE / "comps.csv")
    comps["ev_revenue"] = (comps.implied_equity_value_gbp_m
                           / comps.revenue_at_deal_gbp_m)
    core = comps[~comps.deal.isin(["Villa/NSWE", "Chelsea/BlueCo"])]
    return {
        "p25": float(comps.ev_revenue.quantile(0.25)),
        "median_core": float(core.ev_revenue.median()),
        "p75": float(comps.ev_revenue.quantile(0.75)),
    }


def comps_value(dcf_table: pd.DataFrame) -> pd.DataFrame:
    m = load_multiples()

    def band(row) -> float:
        if row.club in BIG_SIX or row.state == "UCL":
            return m["p75"]
        if row.state == "REL":
            return m["p25"]
        return m["median_core"]

    out = dcf_table[["club", "state", "revenue_fy24_gbp_m"]].copy()
    out["comps_multiple"] = dcf_table.apply(band, axis=1).round(2)
    out["comps_value_gbp_m"] = (out.revenue_fy24_gbp_m * out.comps_multiple).round(0)
    out["comps_low_gbp_m"] = (out.revenue_fy24_gbp_m * m["p25"]).round(0)
    out["comps_high_gbp_m"] = (out.revenue_fy24_gbp_m * m["p75"]).round(0)
    return out.drop(columns=["state", "revenue_fy24_gbp_m"])

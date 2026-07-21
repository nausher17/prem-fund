"""Scenario-weighted DCF over the empirical league-position Markov chain.

Valuation logic (per PL club, valuation date = end of season 2023-24):

1. Base revenue R0 = extracted fy2024 turnover (Companies House OCR layer).
2. The club's competitive state follows the empirical transition matrix
   (transitions.py). Each state carries a revenue multiplier CALIBRATED from
   the panel: median across clubs of (mean revenue in state s) / (mean
   revenue in MID seasons), computed on clubs observed in both states.
   The Championship multiplier comes from assumptions.yaml (panel has no
   post-relegation revenue rows until DNCG/Championship data lands).
3. Expected revenue: E[R_t] = R0 / m(s0) * sum_s P^t[s0, s] * m(s) * (1+g)^t,
   g = club trailing revenue CAGR shrunk 50% toward the league median.
4. EBITDA margin = 1 - wage_ratio (club 3yr mean, capped) - other_costs;
   FCF = fcf_conversion * EBITDA. Negative margins flow through honestly —
   a loss-making club can carry a negative operating DCF (the gap to its
   market price is the point of the exercise, not an error).
5. Discount at club WACC (assumptions.yaml, user-approved): season-average
   10y gilt + ERP + size/illiquidity + tier adjustment.
6. Terminal value: Gordon growth on year-H FCF when positive, else zero
   (flagged) — no terminal value is manufactured for loss-makers.

Every input and intermediate is returned so the output table is auditable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .transitions import FULL_STATES, estimate, position_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
HERE = Path(__file__).resolve().parent

ASSUMPTIONS = yaml.safe_load((HERE / "assumptions.yaml").read_text())

VALUATION_SEASON = "2023-2024"


def load_panel_pl() -> pd.DataFrame:
    panel = pd.read_csv(PROCESSED / "panel.csv")
    return panel[panel.league == "premier-league"].copy()


def calibrate_state_multipliers(pl: pd.DataFrame) -> dict[str, float]:
    """Median cross-club revenue ratio of each state vs the club's own MID
    seasons; Championship from assumptions (not observable in panel)."""
    pl = pl[pl.revenue_gbp.notna()].copy()
    pl["state"] = pl.position.map(position_state)
    ratios: dict[str, list[float]] = {s: [] for s in ("UCL", "EUR", "LOW", "REL")}
    for club, grp in pl.groupby("canonical"):
        mid = grp.loc[grp.state == "MID", "revenue_gbp"].mean()
        if not np.isfinite(mid) or mid <= 0:
            continue
        for s in ratios:
            rev_s = grp.loc[grp.state == s, "revenue_gbp"].mean()
            if np.isfinite(rev_s) and rev_s > 0:
                ratios[s].append(rev_s / mid)
    mult = {"MID": 1.0}
    for s, vals in ratios.items():
        mult[s] = float(np.median(vals)) if vals else np.nan
    # REL season is still a PL season (revenue roughly LOW-like); CHAMP is
    # the post-relegation year -> assumptions haircut vs MID
    mult["CHAMP"] = 1.0 + ASSUMPTIONS["scenarios"]["relegation_revenue_haircut"]
    if not np.isfinite(mult.get("REL", np.nan)):
        mult["REL"] = mult["LOW"]
    return mult


def club_wacc(club: str, promoted: bool, macro_row: pd.Series) -> float:
    w = ASSUMPTIONS["wacc"]
    rate = macro_row["uk_gilt_10y_season"] / 100.0
    rate += w["equity_risk_premium"]["value"]
    rate += w["size_illiquidity_premium"]["value"]
    if club in w["big_six"]:
        rate += w["tier_adjustment"]["big_six"]
    if promoted:
        rate += w["tier_adjustment"]["newly_promoted"]
    return rate


def club_growth(pl: pd.DataFrame, club: str) -> float:
    """Trailing revenue CAGR shrunk 50% toward the league median CAGR."""
    grp = pl[(pl.canonical == club) & pl.revenue_gbp.notna()].sort_values("season")
    league_g = pl.groupby("canonical").revenue_gbp.apply(
        lambda s: (s.dropna().iloc[-1] / s.dropna().iloc[0]) ** (1 / max(len(s.dropna()) - 1, 1)) - 1
        if s.notna().sum() >= 2 else np.nan).median()
    if len(grp) >= 3:
        r = grp.revenue_gbp.values
        club_g = (r[-1] / r[0]) ** (1 / (len(r) - 1)) - 1
    else:
        club_g = league_g
    g = 0.5 * club_g + 0.5 * league_g
    # clamp: no club compounds >12% or shrinks >5% p.a. for a decade
    return float(np.clip(g, -0.05, 0.12))


def dcf_club(club: str, pl: pd.DataFrame, P: pd.DataFrame,
             mult: dict[str, float], macro_row: pd.Series) -> dict:
    proj = ASSUMPTIONS["projection"]
    row = pl[(pl.canonical == club) & (pl.season == VALUATION_SEASON)].iloc[0]
    R0 = row.revenue_gbp
    s0 = position_state(int(row.position))
    # R0 was earned in state s0 (a REL-season is still a PL season), so the
    # level is normalised by mult[s0]; but a relegated club's chain STARTS
    # from the Championship next season.
    s0_chain = "CHAMP" if s0 == "REL" else s0

    w2r_hist = pl[(pl.canonical == club)].sort_values("season").wage_to_revenue.dropna().tail(3)
    # single-season wage ratios are unreliable (often a wages-only note read);
    # require two observations before trusting club-specific data
    w2r = min(float(w2r_hist.mean()) if len(w2r_hist) >= 2 else
              float(pl.wage_to_revenue.median()),
              proj["cost_structure"]["wage_to_revenue"]["cap"])
    other = proj["cost_structure"]["other_costs_share_of_revenue"]["value"]
    conv = proj["cost_structure"]["fcf_conversion_of_ebitda"]["value"]
    margin = 1.0 - w2r - other

    g = club_growth(pl, club)
    wacc = club_wacc(club, bool(row.newly_promoted), macro_row)
    H = proj["horizon_years"]
    gT = proj["terminal_growth"]

    m_vec = np.array([mult[s] for s in FULL_STATES])
    state_idx = FULL_STATES.index(s0_chain)
    norm = mult[s0]  # revenue level earned in the valuation season's state
    Pk = np.identity(len(FULL_STATES))
    pv, fcf_H = 0.0, 0.0
    for t in range(1, H + 1):
        Pk = Pk @ P.values
        exp_mult = Pk[state_idx] @ m_vec
        R_t = R0 / norm * exp_mult * (1 + g) ** t
        fcf_t = R_t * margin * conv
        pv += fcf_t / (1 + wacc) ** t
        if t == H:
            fcf_H = fcf_t
    if fcf_H > 0:
        terminal = fcf_H * (1 + gT) / (wacc - gT) / (1 + wacc) ** H
    else:
        terminal = 0.0

    return {
        "club": club, "season": VALUATION_SEASON, "state": s0,
        "revenue_fy24_gbp_m": R0 / 1e6, "wage_ratio": round(w2r, 3),
        "ebitda_margin": round(margin, 3), "growth": round(g, 4),
        "wacc": round(wacc, 4),
        "dcf_operating_gbp_m": round(pv / 1e6, 1),
        "dcf_terminal_gbp_m": round(terminal / 1e6, 1),
        "dcf_value_gbp_m": round((pv + terminal) / 1e6, 1),
        "terminal_zeroed_lossmaker": fcf_H <= 0,
        "p_ucl_5y": round(float(np.linalg.matrix_power(P.values, 5)[state_idx,
                          FULL_STATES.index("UCL")]), 3),
        "p_champ_5y": round(float(np.linalg.matrix_power(P.values, 5)[state_idx,
                            FULL_STATES.index("CHAMP")]), 3),
    }


def run() -> pd.DataFrame:
    pl = load_panel_pl()
    P, _ = estimate()
    mult = calibrate_state_multipliers(pl)
    macro_row = pl[pl.season == VALUATION_SEASON].iloc[0]
    clubs = pl[(pl.season == VALUATION_SEASON) & pl.revenue_gbp.notna()].canonical
    results = pd.DataFrame([dcf_club(c, pl, P, mult, macro_row) for c in clubs])
    results.attrs["state_multipliers"] = mult
    return results.sort_values("dcf_value_gbp_m", ascending=False, ignore_index=True)


if __name__ == "__main__":
    res = run()
    print("Calibrated state multipliers:", {k: round(v, 3) for k, v in res.attrs["state_multipliers"].items()})
    print(res.to_string(index=False))

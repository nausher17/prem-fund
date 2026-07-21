"""UCL qualification as a real option (H3 machinery).

The embedded option: qualifying for the Champions League triggers a
discontinuous incremental cash flow (calibrated UCL revenue multiplier vs the
club's base). Linear expected-value thinking prices the mean path; option
logic prices the asymmetric payoff of position volatility around the top-4
strike. H3 asks whether the market prices that difference.

Model: additive binomial lattice on final league position.
- x0    = club's 2023-24 position
- step  = club's own season-to-season position volatility from the panel
          (std of position; >=3 observed seasons, else league median),
          split into annual up/down moves of one sigma
- strike: position <= 4 at a step
- payoff per in-the-money node-year: incremental FCF
          = R0/m(s0) * (m_UCL - m_MID) * margin * fcf_conversion
- p = 0.5 physical measure. Real-option caveat (documented in the report):
  the underlying is not traded, so no risk-neutral replication exists; we
  price under the physical measure and discount at the club WACC, the
  standard practical treatment (Trigeorgis) — and exactly the assumption H3
  then stress-tests.

Cross-check column: the Markov-chain expected uplift value (sum over years
of P(UCL at t) * payoff, discounted) — the 'linear' benchmark whose gap to
the lattice value is the convexity premium at stake in H3.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from .transitions import FULL_STATES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"

HORIZON = 10


def position_volatility(pl: pd.DataFrame) -> pd.Series:
    """Per-club std of final position across panel seasons (>=3 seasons)."""
    stds = pl.groupby("canonical").position.agg(["std", "count"])
    league_median = stds.loc[stds["count"] >= 3, "std"].median()
    return stds["std"].where(stds["count"] >= 3, league_median).fillna(league_median)


def lattice_value(x0: float, sigma: float, annual_payoff: float,
                  wacc: float, horizon: int = HORIZON) -> float:
    """Expected discounted payoff on an additive random walk of position."""
    if sigma <= 0:
        sigma = 1.0
    value = 0.0
    for t in range(1, horizon + 1):
        # position after t steps: x0 + (2k - t) * sigma for k up-moves
        k = np.arange(t + 1)
        prob = np.array([comb(t, int(j)) for j in k]) * 0.5 ** t
        pos = x0 - (2 * k - t) * sigma  # up-move = position improves (falls)
        in_money = pos <= 4.0
        value += float(prob[in_money].sum()) * annual_payoff / (1 + wacc) ** t
    return value


def chain_value(P: pd.DataFrame, s0_chain: str, annual_payoff: float,
                wacc: float, horizon: int = HORIZON) -> float:
    idx = FULL_STATES.index(s0_chain)
    ucl = FULL_STATES.index("UCL")
    Pk = np.identity(len(FULL_STATES))
    value = 0.0
    for t in range(1, horizon + 1):
        Pk = Pk @ P.values
        value += Pk[idx, ucl] * annual_payoff / (1 + wacc) ** t
    return value


def option_values(dcf_table: pd.DataFrame, pl: pd.DataFrame, P: pd.DataFrame,
                  mult: dict[str, float]) -> pd.DataFrame:
    vols = position_volatility(pl)
    val_season = dcf_table.season.iloc[0]
    positions = pl[pl.season == val_season].set_index("canonical").position

    rows = []
    for r in dcf_table.itertuples():
        payoff = (r.revenue_fy24_gbp_m * 1e6 / mult[r.state if r.state != "REL" else "REL"]
                  * (mult["UCL"] - mult["MID"])
                  * r.ebitda_margin * 0.60)
        payoff = max(payoff, 0.0)  # a loss-making club still gains the uplift
        # revenue but margin<=0 zeroes it — documented limitation
        s0_chain = "CHAMP" if r.state == "REL" else r.state
        rows.append({
            "club": r.club,
            "ucl_option_gbp_m": round(lattice_value(
                float(positions[r.club]), float(vols[r.club]), payoff, r.wacc) / 1e6, 1),
            "ucl_linear_gbp_m": round(chain_value(
                P, s0_chain, payoff, r.wacc) / 1e6, 1),
        })
    out = pd.DataFrame(rows)
    out["convexity_premium_gbp_m"] = (out.ucl_option_gbp_m - out.ucl_linear_gbp_m).round(1)
    return out

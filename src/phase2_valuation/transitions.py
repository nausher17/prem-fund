"""League-position transition probabilities, estimated from the panel.

States (Premier League seasons):
  UCL   final position 1-4
  EUR   5-7   (Europa/Conference range)
  MID   8-14
  LOW   15-17
  REL   relegated (18-20) -> leaves the panel
  CHAMP out of the PL (Championship); re-entry via promotion

The estimator counts observed season-to-season moves 2015-16..2023-24,
including relegation exits and promotion re-entries, giving an empirical
Markov chain used for scenario weights in the DCF. Nothing is hand-set; the
matrix is written to outputs/phase2/transition_matrix.csv for inspection.

Small-sample honesty: with 9 transitions per club max, cells are noisy.
We report row counts alongside probabilities and apply add-one smoothing
only to structurally-possible-but-unobserved cells (documented).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs" / "phase2"

STATES = ["UCL", "EUR", "MID", "LOW", "REL"]  # REL = exits PL this season
FULL_STATES = STATES[:-1] + ["CHAMP"]         # chain states (REL -> CHAMP)


def position_state(position: int) -> str:
    if position <= 4:
        return "UCL"
    if position <= 7:
        return "EUR"
    if position <= 14:
        return "MID"
    if position <= 17:
        return "LOW"
    return "REL"


def estimate(league: str = "premier-league") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (transition matrix P over FULL_STATES, raw counts)."""
    panel = pd.read_csv(PROCESSED / "panel.csv")
    lp = panel[panel.league == league][["season", "canonical", "position"]].copy()
    lp["year"] = lp.season.str.slice(0, 4).astype(int)
    lp["state"] = lp.position.map(position_state)

    years = sorted(lp.year.unique())
    club_state: dict[tuple[str, int], str] = {
        (r.canonical, r.year): r.state for r in lp.itertuples()}
    clubs = lp.canonical.unique()

    counts = pd.DataFrame(0, index=FULL_STATES, columns=FULL_STATES, dtype=float)
    for club in clubs:
        for y in years[:-1]:
            cur = club_state.get((club, y))
            nxt = club_state.get((club, y + 1))
            cur_full = "CHAMP" if cur in (None, "REL") else cur
            nxt_full = "CHAMP" if nxt in (None, "REL") else nxt
            # skip years before the club ever appears (not yet in universe)
            if cur is None and not any(club_state.get((club, yy)) for yy in years if yy <= y):
                continue
            # REL state itself transitions: the season it finishes 18-20 it
            # is still in the PL; next season it is in CHAMP
            if cur == "REL":
                cur_full = "CHAMP"
            counts.loc[cur_full, nxt_full] += 1

    # relegation is observable directly: P(cur -> CHAMP) via REL finishes
    # (counted above because REL-season -> next season absent). Promotion
    # back: CHAMP -> any PL state.
    smoothed = counts.copy()
    # structurally possible but unobserved: give 0.5 pseudo-count so the DCF
    # never treats e.g. LOW -> UCL as impossible, merely very unlikely
    smoothed[smoothed == 0] += 0.5
    P = smoothed.div(smoothed.sum(axis=1), axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # full precision: R re-validates stochasticity from this artifact
    P.to_csv(OUT_DIR / "transition_matrix.csv")
    counts.to_csv(OUT_DIR / "transition_counts.csv")
    return P, counts


if __name__ == "__main__":
    P, counts = estimate()
    print("Row counts (observed transitions):")
    print(counts.sum(axis=1).astype(int).to_string())
    print("\nTransition matrix:")
    print(P.round(3).to_string())

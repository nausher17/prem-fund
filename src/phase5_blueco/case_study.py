"""Phase 5: BlueCo natural experiment — Chelsea & RC Strasbourg.

COI DISCLOSURE (restated wherever this analysis appears): the author is a
Chelsea supporter; both clubs carry coi_flag = 1 in the panel and are never
excluded — flagged for the reader instead.

Components:
1. Transfer/loan network Chelsea <-> Strasbourg from Transfermarkt league
   transfer pages, seasons 2022-23..2025-26 (the 2024-26 extension exists
   for this case study only; the analysis panel remains 2015-24).
2. Strasbourg trajectory (squad value, avg age, league position) pre/post
   the June-2023 BlueCo acquisition vs a data-matched Ligue 1 control:
   nearest neighbour on 2019-23 squad value + position paths, excluding
   MCO clubs (a lightweight synthetic-control design; the full donor-pool
   weighting is out of scope and stated as such).
3. Synergy channels quantified where observable: pathway minutes (loanees'
   Ligue 1 minutes), fee flows, squad-age shift.
4. Tie-back to H2's econometric null.

Outputs: outputs/phase5/{network.csv, trajectory.csv, findings.md,
network_fig.png, trajectory_fig.png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "outputs" / "phase5"

ACQUISITION_SEASON = 2023  # 2023-24 = first BlueCo season at Strasbourg


def network() -> pd.DataFrame:
    tr = pd.read_csv(PROCESSED / "tm_transfers.csv")
    mask = ((tr.club_tm.str.contains("Chelsea")
             & tr.counterparty.str.contains("Strasbourg", case=False, na=False)) |
            (tr.club_tm.str.contains("Strasbourg")
             & tr.counterparty.str.contains("Chelsea", case=False, na=False)))
    net = tr[mask].copy()
    # each move appears from both clubs' pages; keep the Strasbourg-side rows
    net = net[net.club_tm.str.contains("Strasbourg")]
    net["move"] = np.where(net.direction == "in", "Chelsea->Strasbourg",
                           "Strasbourg->Chelsea")
    cols = ["season", "move", "player", "age", "position",
            "market_value_eur_m", "fee_raw", "fee_eur_m", "is_loan"]
    return net[cols].sort_values(["season", "player"]).reset_index(drop=True)


def match_control(tm: pd.DataFrame) -> str:
    """Nearest L1 club to Strasbourg on 2019-23 value+position paths."""
    panel = pd.read_csv(PROCESSED / "panel.csv")
    l1 = panel[(panel.league == "ligue-1")].copy()
    l1["year"] = l1.season.str.slice(0, 4).astype(int)
    pre = l1[l1.year.between(2019, 2022)]
    piv_v = pre.pivot_table(index="canonical", columns="year",
                            values="total_market_value_eur_m")
    piv_p = pre.pivot_table(index="canonical", columns="year", values="position")
    target_v, target_p = piv_v.loc["Strasbourg"], piv_p.loc["Strasbourg"]
    mco = set(panel[panel.mco_flag == 1].canonical) | {"Strasbourg"}
    candidates = [c for c in piv_v.index
                  if c not in mco and piv_v.loc[c].notna().all()]
    # standardised distance on value and position paths
    dist = {c: (np.nanmean(((piv_v.loc[c] - target_v) / piv_v.stack().std()) ** 2)
                + np.nanmean(((piv_p.loc[c] - target_p) / piv_p.stack().std()) ** 2))
            for c in candidates}
    return min(dist, key=dist.get)


def trajectory(control: str) -> pd.DataFrame:
    tm = pd.read_csv(PROCESSED / "tm_club_seasons.csv")
    tm["year"] = tm.season.str.slice(0, 4).astype(int)
    cmap = pd.read_csv(PROJECT_ROOT / "src/phase1_data/club_map.csv")
    tm = tm.merge(cmap[["league", "team_tm", "canonical"]],
                  on=["league", "team_tm"])
    sub = tm[tm.canonical.isin(["Strasbourg", control])]
    return sub.pivot_table(index="year", columns="canonical",
                           values=["total_market_value_eur_m", "avg_age_tm"])


def figures(net: pd.DataFrame, traj: pd.DataFrame, control: str) -> None:
    # trajectory figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    v = traj["total_market_value_eur_m"]
    a = traj["avg_age_tm"]
    for club, style in (("Strasbourg", "-o"), (control, "--s")):
        axes[0].plot(v.index, v[club], style, label=club)
        axes[1].plot(a.index, a[club], style, label=club)
    for ax, title in ((axes[0], "Squad value (EUR m)"),
                      (axes[1], "Average squad age")):
        ax.axvline(ACQUISITION_SEASON - 0.5, color="red", ls=":",
                   label="BlueCo acquisition (Jun 2023)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Strasbourg vs matched control — pre/post BlueCo "
                 "(COI: author is a Chelsea supporter)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_fig.png", dpi=150)

    # simple bipartite network figure
    fig2, ax = plt.subplots(figsize=(8, 0.55 * max(len(net), 4) + 1.5))
    ax.axis("off")
    ax.text(0.05, 1.0, "Chelsea", fontsize=13, weight="bold", va="top")
    ax.text(0.75, 1.0, "Strasbourg", fontsize=13, weight="bold", va="top")
    for i, r in enumerate(net.itertuples()):
        y = 0.92 - i * (0.85 / max(len(net), 1))
        lbl = f"{r.player} ({r.season}{', loan' if r.is_loan else ''}" \
              f"{'' if pd.isna(r.fee_eur_m) else f', EUR {r.fee_eur_m:.1f}m'})"
        x0, x1 = (0.22, 0.73) if r.move == "Chelsea->Strasbourg" else (0.73, 0.22)
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color="tab:blue"))
        ax.text(0.475, y + 0.012, lbl, fontsize=8, ha="center")
    ax.set_title("Chelsea-Strasbourg player movement network, 2022-2026",
                 fontsize=11)
    fig2.savefig(OUT / "network_fig.png", dpi=150, bbox_inches="tight")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    net = network()
    tm = pd.read_csv(PROCESSED / "tm_club_seasons.csv")
    control = match_control(tm)
    traj = trajectory(control)
    net.to_csv(OUT / "network.csv", index=False)
    traj.to_csv(OUT / "trajectory.csv")
    figures(net, traj, control)
    print(net.to_string(index=False))
    print(f"\nmatched control club: {control}")
    print(traj.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

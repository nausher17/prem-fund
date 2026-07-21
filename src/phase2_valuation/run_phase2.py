"""Phase 2 entry point: club valuations from three independent lenses.

    .venv/bin/python -m src.phase2_valuation.run_phase2

Outputs (outputs/phase2/):
- valuations.csv          DCF + comps + UCL option per PL club (FY2024)
- transition_matrix.csv   empirical Markov chain (written by transitions.py)
- trophy_premium.csv      Forbes 2024 benchmark vs model values

French clubs are absent pending DNCG financials (documented Phase 1 open
item) — the framework is league-agnostic once revenue lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import dcf
from .comps import comps_value
from .transitions import estimate
from .ucl_option import option_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs" / "phase2"
HERE = Path(__file__).resolve().parent


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pl = dcf.load_panel_pl()
    P, _ = estimate()

    table = dcf.run()
    mult = table.attrs["state_multipliers"]
    table = table.merge(comps_value(table), on="club")
    table = table.merge(option_values(table, pl, P, mult), on="club")
    table["blend_value_gbp_m"] = ((table.dcf_value_gbp_m
                                   + table.comps_value_gbp_m) / 2).round(1)
    table.to_csv(OUT / "valuations.csv", index=False)
    print(f"Wrote {OUT / 'valuations.csv'} ({len(table)} clubs)")

    # Forbes benchmark -> trophy-asset premium
    forbes = pd.read_csv(HERE / "forbes_2024.csv")
    macro = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "macro_monthly.csv")
    macro["month"] = pd.to_datetime(macro["month"])
    usd_gbp = float(macro.loc[macro.month == "2024-05-01", "usd_per_gbp"].iloc[0])
    forbes["forbes_gbp_m"] = (forbes.forbes_2024_usd_bn * 1000 / usd_gbp).round(0)
    tp = forbes.merge(
        table[["club", "dcf_value_gbp_m", "comps_value_gbp_m", "blend_value_gbp_m"]],
        on="club")
    tp["trophy_premium_gbp_m"] = (tp.forbes_gbp_m - tp.dcf_value_gbp_m).round(0)
    tp["premium_x_dcf"] = (tp.forbes_gbp_m / tp.dcf_value_gbp_m).round(2)
    tp.to_csv(OUT / "trophy_premium.csv", index=False)

    print("\nValuations (GBP m, FY2024):")
    cols = ["club", "state", "dcf_value_gbp_m", "comps_value_gbp_m",
            "blend_value_gbp_m", "ucl_option_gbp_m", "convexity_premium_gbp_m"]
    print(table[cols].to_string(index=False))
    print(f"\nTrophy premium (Forbes May-2024 @ {usd_gbp:.3f} USD/GBP):")
    print(tp[["club", "forbes_gbp_m", "dcf_value_gbp_m",
              "trophy_premium_gbp_m", "premium_x_dcf"]].to_string(index=False))
    print(f"\nMedian Forbes/DCF multiple: {tp.premium_x_dcf.median():.2f}x "
          "— the DCF undershoot quantifies the trophy-asset premium (finding, "
          "not error: cash flows alone do not price scarcity/prestige).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

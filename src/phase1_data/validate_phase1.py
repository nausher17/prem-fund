"""Three-layer Phase 1 validation. Writes data/validation/phase1_validation_report.md.

Layer A — schema/range: league sizes, points identities, GF==GA aggregates,
          plausible market values/ages, no negatives.
Layer B — cross-source reconciliation: football-data <-> Transfermarkt club
          universes (1:1 via club_map), player minutes vs matches played,
          Transfermarkt's published average age vs our recomputed squad mean age.
Layer C — R re-computation (r/phase1_validation.R), run separately by the
          entry point; its output CSV is folded into the report here.

Every check prints PASS/FAIL and the report records values, not just verdicts.
Exits non-zero on any failure. Known gaps (Companies House financials) are
recorded explicitly as OPEN items, never papered over.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
VALIDATION = PROJECT_ROOT / "data" / "validation"
HERE = Path(__file__).resolve().parent

results: list[tuple[str, str, str]] = []  # (layer, check, PASS/FAIL/OPEN + detail)


def check(layer: str, name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    results.append((layer, name, f"{status} {detail}".strip()))
    print(f"  [{status}] {name} {detail}")


def open_item(layer: str, name: str, detail: str) -> None:
    results.append((layer, name, f"OPEN {detail}"))
    print(f"  [OPEN] {name} {detail}")


def layer_a() -> None:
    print("Layer A — schema/range checks")
    t = pd.read_csv(PROCESSED / "league_tables.csv")
    sizes = t.groupby(["league", "season"]).size()
    expected = {(lg, s): 18 if (lg, s) == ("ligue-1", "2023-2024") else 20
                for lg, s in sizes.index}
    bad_sizes = {k: v for k, v in sizes.items() if v != expected[k]}
    check("A", "league sizes (20 clubs; L1 2023-24 has 18)", not bad_sizes, str(bad_sizes))
    check("A", "points identity 3w+d", (t.points_raw == 3 * t.w + t.d).all())
    check("A", "matches identity w+d+l == mp", (t.mp == t.w + t.d + t.l).all())
    agg = t.groupby(["league", "season"])[["gf", "ga"]].sum()
    check("A", "aggregate GF == GA per league-season", (agg.gf == agg.ga).all())
    check("A", "no negative goals/points", ((t[["gf", "ga", "points"]] >= 0).all().all()))

    tm = pd.read_csv(PROCESSED / "tm_club_seasons.csv")
    check("A", "squad values in [10, 2500] EUR m",
          tm.total_market_value_eur_m.between(10, 2500).all(),
          f"range {tm.total_market_value_eur_m.min():.0f}-{tm.total_market_value_eur_m.max():.0f}")
    check("A", "TM avg ages in [20, 32]", tm.avg_age_tm.between(20, 32).all())

    m = pd.read_csv(PROCESSED / "macro_monthly.csv")
    check("A", "gilt yields in [0, 8]%", m.uk_gilt_10y.dropna().between(0, 8).all())


def layer_b() -> None:
    print("Layer B — cross-source reconciliation")
    cmap = pd.read_csv(HERE / "club_map.csv")
    t = pd.read_csv(PROCESSED / "league_tables.csv")
    t = t[t.season != "2014-2015"]
    tm = pd.read_csv(PROCESSED / "tm_club_seasons.csv")

    fd_univ = set(map(tuple, t[["league", "season", "team"]].values))
    fd_mapped = {(lg, s, cmap.set_index(["league", "team_fd"]).canonical.get((lg, tf)))
                 for lg, s, tf in fd_univ}
    tm_mapped = {(r.league, r.season,
                  cmap.set_index(["league", "team_tm"]).canonical.get((r.league, r.team_tm)))
                 for r in tm.itertuples()}
    check("B", "club universes identical across sources (358 club-seasons)",
          fd_mapped == tm_mapped,
          f"fd-only={len(fd_mapped - tm_mapped)}, tm-only={len(tm_mapped - fd_mapped)}")

    players_path = PROCESSED / "tm_player_seasons.csv"
    if players_path.exists():
        p = pd.read_csv(players_path)
        mins = p.groupby(["league", "season", "club_tm"]).minutes.sum().reset_index()
        mins = mins.merge(cmap[["league", "team_tm", "team_fd"]],
                          left_on=["league", "club_tm"], right_on=["league", "team_tm"])
        mins = mins.merge(t[["league", "season", "team", "mp"]],
                          left_on=["league", "season", "team_fd"],
                          right_on=["league", "season", "team"])
        ratio = mins.minutes / (mins.mp * 11 * 90)
        check("B", "player minutes reconcile with matches played (±7%)",
              ratio.between(0.93, 1.07).all(),
              f"min ratio {ratio.min():.3f}, max {ratio.max():.3f}, n={len(mins)}")

        # TM's published squad avg age vs our player-level squad mean age
        ours = p[p.age.notna()].groupby(["league", "season", "club_tm"]).age.mean().reset_index()
        cmp_age = ours.merge(tm[["league", "season", "team_tm", "avg_age_tm"]],
                             left_on=["league", "season", "club_tm"],
                             right_on=["league", "season", "team_tm"])
        corr = cmp_age.age.corr(cmp_age.avg_age_tm)
        # Not identical by construction: overview age is squad-composition on a
        # reference date; ours averages everyone who appeared in the season page.
        check("B", "squad mean age vs TM published avg age (corr > 0.75)",
              corr > 0.75, f"corr={corr:.3f}")
    else:
        open_item("B", "player-level reconciliation", "tm_player_seasons.csv not yet built")

    open_item("B", "financial statements (revenue/wages/EBITDA/debt)",
              "Companies House API key not yet provided; Deloitte Money League "
              "reconciliation deferred until financials land")


def layer_c() -> None:
    print("Layer C — R cross-validation")
    r_out = VALIDATION / "phase1_r_checks.csv"
    if r_out.exists():
        r = pd.read_csv(r_out)
        ok = (r.status != "FAIL").all()
        check("C", "R re-computation agrees with Python", ok,
              "; ".join(f"{row.check}={row.status}" for row in r.itertuples()))
    else:
        open_item("C", "R validation", "run `Rscript r/phase1_validation.R` first")


def write_report() -> None:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 1 validation report — {date.today().isoformat()}", ""]
    for layer in ("A", "B", "C"):
        lines.append({"A": "## Layer A — schema/range",
                      "B": "## Layer B — cross-source reconciliation",
                      "C": "## Layer C — R cross-validation"}[layer])
        for lay, name, status in results:
            if lay == layer:
                lines.append(f"- **{name}**: {status}")
        lines.append("")
    lines += [
        "## Known open items",
        "- Financial-statement features pending `COMPANIES_HOUSE_API_KEY` "
        "(free registration; see README). Panel intentionally ships without "
        "revenue/wages/EBITDA/debt columns rather than proxying them.",
        "- French club financials will come from DNCG annual reports "
        "(manual, cited extraction) in a later Phase 1 iteration.",
        "",
        "## Conflict of interest",
        "- Author is a Chelsea supporter. Chelsea and Strasbourg carry "
        "`coi_flag = 1` in the panel (flagged, never excluded).",
    ]
    (VALIDATION / "phase1_validation_report.md").write_text("\n".join(lines))
    print(f"\nReport -> {VALIDATION / 'phase1_validation_report.md'}")


def main() -> int:
    layer_a()
    layer_b()
    layer_c()
    write_report()
    failed = [r for r in results if r[2].startswith("FAIL")]
    if failed:
        print(f"\n{len(failed)} validation check(s) FAILED")
        return 1
    print("\nAll validation checks passed (open items listed in report).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

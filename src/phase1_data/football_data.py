"""Match results and computed standings from football-data.co.uk.

football-data.co.uk publishes canonical final-score CSVs per league-season
(E0 = Premier League, F1 = Ligue 1). We download them (throttled + cached),
then compute league tables from the raw results: points, W/D/L, GF/GA, GD,
final position.

Two integrity notes, both documented because they affect the computed tables:

1. Point deductions are NOT in the results and must be applied on top.
   In the 2015-16..2023-24 window the only material top-flight deductions are
   Premier League 2023-24: Everton -8 (two PSR charges, final after appeals),
   Nottingham Forest -4. Sources: Premier League commission rulings, Nov 2023 /
   Mar 2024 (Everton appeal reduced 10->6, second charge +2; Forest Mar 2024).
2. Ligue 1 2019-20 was abandoned after matchday ~28 (COVID) and settled on a
   points-per-game basis; the computed table over played matches reproduces the
   official ordering at the top (PSG champions) but MP differs by club.

Tie-breaks: points, then goal difference, then goals scored (Premier League
rules; Ligue 1 uses GD too). Head-to-head edge cases are not modelled — fine
for our features (points, GD, position band), flagged in limitations.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import pandas as pd

from .http_cache import ThrottledCachedSession

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "football_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LEAGUES = {"premier-league": "E0", "ligue-1": "F1"}
# 2014-15 is included only to derive promotion flags for 2015-16 (league
# membership deltas); the analysis panel itself starts at 2015-16.
DEFAULT_SEASONS = [f"{y}-{y + 1}" for y in range(2014, 2024)]

# (league, season, team) -> points adjustment. See module docstring for sources.
POINT_DEDUCTIONS = {
    ("premier-league", "2023-2024", "Everton"): -8,
    ("premier-league", "2023-2024", "Nott'm Forest"): -4,
}

# Test oracle (public record), in football-data.co.uk team naming.
EXPECTED_CHAMPIONS = {
    ("premier-league", "2014-2015"): "Chelsea",
    ("ligue-1", "2014-2015"): "Paris SG",
    ("premier-league", "2015-2016"): "Leicester",
    ("premier-league", "2016-2017"): "Chelsea",
    ("premier-league", "2017-2018"): "Man City",
    ("premier-league", "2018-2019"): "Man City",
    ("premier-league", "2019-2020"): "Liverpool",
    ("premier-league", "2020-2021"): "Man City",
    ("premier-league", "2021-2022"): "Man City",
    ("premier-league", "2022-2023"): "Man City",
    ("premier-league", "2023-2024"): "Man City",
    ("ligue-1", "2015-2016"): "Paris SG",
    ("ligue-1", "2016-2017"): "Monaco",
    ("ligue-1", "2017-2018"): "Paris SG",
    ("ligue-1", "2018-2019"): "Paris SG",
    ("ligue-1", "2019-2020"): "Paris SG",
    ("ligue-1", "2020-2021"): "Lille",
    ("ligue-1", "2021-2022"): "Paris SG",
    ("ligue-1", "2022-2023"): "Paris SG",
    ("ligue-1", "2023-2024"): "Paris SG",
}


def csv_url(league: str, season: str) -> str:
    """e.g. https://www.football-data.co.uk/mmz4281/1516/E0.csv"""
    start, end = season.split("-")
    code = f"{start[2:]}{end[2:]}"
    return f"https://www.football-data.co.uk/mmz4281/{code}/{LEAGUES[league]}.csv"


def load_matches(
    league: str, season: str, session: ThrottledCachedSession
) -> pd.DataFrame:
    text = session.get(csv_url(league, season))
    df = pd.read_csv(io.StringIO(text), encoding_errors="replace")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    keep = [c for c in (
        "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HS", "AS", "HST", "AST",
    ) if c in df.columns]
    df = df[keep].copy()
    df.insert(0, "league", league)
    df.insert(1, "season", season)
    return df


def compute_table(matches: pd.DataFrame) -> pd.DataFrame:
    """League table from match results, with documented point deductions."""
    league = matches["league"].iloc[0]
    season = matches["season"].iloc[0]

    rows = {}
    for team in pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).unique():
        rows[team] = dict(team=team, mp=0, w=0, d=0, l=0, gf=0, ga=0)
    for m in matches.itertuples():
        hg, ag = int(m.FTHG), int(m.FTAG)
        home, away = rows[m.HomeTeam], rows[m.AwayTeam]
        home["mp"] += 1; away["mp"] += 1
        home["gf"] += hg; home["ga"] += ag
        away["gf"] += ag; away["ga"] += hg
        if hg > ag:
            home["w"] += 1; away["l"] += 1
        elif hg < ag:
            away["w"] += 1; home["l"] += 1
        else:
            home["d"] += 1; away["d"] += 1

    table = pd.DataFrame(rows.values())
    table["gd"] = table["gf"] - table["ga"]
    table["points_raw"] = 3 * table["w"] + table["d"]
    table["deduction"] = [
        POINT_DEDUCTIONS.get((league, season, t), 0) for t in table["team"]
    ]
    table["points"] = table["points_raw"] + table["deduction"]
    table = table.sort_values(
        ["points", "gd", "gf"], ascending=False, ignore_index=True
    )
    table["position"] = table.index + 1
    table.insert(0, "league", league)
    table.insert(1, "season", season)
    return table


def build(
    leagues: list[str], seasons: list[str],
    session: ThrottledCachedSession | None = None,
) -> dict[str, pd.DataFrame]:
    session = session or ThrottledCachedSession(RAW_DIR)
    all_matches, all_tables = [], []
    for league in leagues:
        for season in seasons:
            log.info("football-data: %s %s", league, season)
            matches = load_matches(league, season, session)
            all_matches.append(matches)
            all_tables.append(compute_table(matches))
    return {
        "matches": pd.concat(all_matches, ignore_index=True),
        "league_tables": pd.concat(all_tables, ignore_index=True),
    }


def sanity_check_champions(tables: pd.DataFrame) -> bool:
    ok = True
    champs = tables[tables["position"] == 1]
    for (league, season), expected in EXPECTED_CHAMPIONS.items():
        subset = champs[(champs["league"] == league) & (champs["season"] == season)]
        if subset.empty:
            continue  # not scraped in this run
        actual = subset.iloc[0]["team"]
        match = actual == expected
        print(f"  [{'OK ' if match else 'FAIL'}] {league:>14} {season}: "
              f"{actual} (expected {expected})")
        ok &= match
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES),
                        choices=list(LEAGUES))
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    frames = build(args.leagues, args.seasons)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        out = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out} ({len(df)} rows)")

    print("\nSanity check — champions per season:")
    if not sanity_check_champions(frames["league_tables"]):
        print("CHAMPIONS SANITY CHECK FAILED — data layer is broken, do not proceed.")
        return 1
    print("Champions sanity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

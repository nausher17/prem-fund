"""Phase 1 unit + integration tests.

Unit tests run with no data files. Integration tests validate the processed
CSVs when present (skipped otherwise, so the suite works in a clean checkout
before the scrape has run).
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from src.phase1_data import football_data
from src.phase1_data.transfermarkt import parse_minutes, parse_money_eur_m

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CLUB_MAP = Path(__file__).resolve().parents[1] / "src" / "phase1_data" / "club_map.csv"


# -- money / minutes parsing --------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("€13.84m", 13.84),
    ("€1.02bn", 1020.0),
    ("€500k", 0.5),
    ("Loan fee: €5.00m", 5.0),
    ("€121.00m", 121.0),
])
def test_parse_money(text, expected):
    assert parse_money_eur_m(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["-", "?", "", "free transfer", "loan transfer"])
def test_parse_money_unparseable_is_nan(text):
    assert math.isnan(parse_money_eur_m(text))


@pytest.mark.parametrize("text,expected", [
    ("2.020'", 2020.0), ("90'", 90.0), ("-", 0.0), ("", 0.0),
])
def test_parse_minutes(text, expected):
    assert parse_minutes(text) == expected


# -- league table computation --------------------------------------------------

def _toy_matches():
    # A beats B 2-0, B draws C 1-1, C beats A 3-1 -> all on 3 pts,
    # GD: C +2, A -1, B -1; GF: C 4, B 1, A 3 -> order C, A, B
    rows = [
        ("A", "B", 2, 0, "H"),
        ("B", "C", 1, 1, "D"),
        ("C", "A", 3, 1, "H"),
    ]
    return pd.DataFrame([
        {"league": "premier-league", "season": "2015-2016",
         "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag, "FTR": r}
        for h, a, hg, ag, r in rows])


def test_compute_table_ordering():
    table = football_data.compute_table(_toy_matches())
    assert list(table["team"]) == ["C", "A", "B"]
    assert list(table["points"]) == [4, 3, 1]
    assert list(table["position"]) == [1, 2, 3]
    assert table.loc[0, "gd"] == 2


def test_compute_table_applies_deductions(monkeypatch):
    monkeypatch.setitem(
        football_data.POINT_DEDUCTIONS,
        ("premier-league", "2015-2016", "C"), -3)
    table = football_data.compute_table(_toy_matches())
    c = table[table["team"] == "C"].iloc[0]
    assert c["points"] == 1 and c["points_raw"] == 4
    # deduction drops C level with B on 1pt, but C's +2 GD wins the tiebreak
    assert c["position"] == 2
    assert list(table["team"]) == ["A", "C", "B"]


def test_real_deductions_registered():
    assert football_data.POINT_DEDUCTIONS[
        ("premier-league", "2023-2024", "Everton")] == -8
    assert football_data.POINT_DEDUCTIONS[
        ("premier-league", "2023-2024", "Nott'm Forest")] == -4


# -- club map integrity ---------------------------------------------------------

def test_club_map_unique_and_coi():
    cmap = pd.read_csv(CLUB_MAP)
    assert not cmap.duplicated(["league", "canonical"]).any()
    assert not cmap.duplicated(["league", "team_fd"]).any()
    assert not cmap.duplicated(["league", "team_tm"]).any()
    assert not cmap["tm_club_id"].duplicated().any()
    assert sorted(cmap[cmap.coi_flag == 1].canonical) == ["Chelsea", "Strasbourg"]


# -- integration checks on processed data (skipped before first scrape) --------

needs = lambda name: pytest.mark.skipif(  # noqa: E731
    not (PROCESSED / name).exists(), reason=f"{name} not built yet")


@needs("league_tables.csv")
def test_champions_match_reality():
    tables = pd.read_csv(PROCESSED / "league_tables.csv")
    assert football_data.sanity_check_champions(tables)


@needs("league_tables.csv")
def test_points_consistency():
    t = pd.read_csv(PROCESSED / "league_tables.csv")
    assert (t["points_raw"] == 3 * t["w"] + t["d"]).all()
    assert (t["mp"] == t["w"] + t["d"] + t["l"]).all()
    assert (t["gf"] >= 0).all() and (t["ga"] >= 0).all()
    # every league-season's GF must equal GA in aggregate
    sums = t.groupby(["league", "season"])[["gf", "ga"]].sum()
    assert (sums["gf"] == sums["ga"]).all()


@needs("tm_club_seasons.csv")
def test_tm_values_plausible():
    tm = pd.read_csv(PROCESSED / "tm_club_seasons.csv")
    assert tm["total_market_value_eur_m"].between(10, 2500).all()
    assert tm["avg_age_tm"].between(20, 32).all()
    assert not tm["total_market_value_eur_m"].isna().any()


@needs("tm_club_seasons.csv")
def test_club_map_covers_all_panel_rows():
    cmap = pd.read_csv(CLUB_MAP)
    tm = pd.read_csv(PROCESSED / "tm_club_seasons.csv")
    lt = pd.read_csv(PROCESSED / "league_tables.csv")
    lt = lt[lt["season"] != "2014-2015"]  # membership-only season
    unmapped_tm = set(map(tuple, tm[["league", "team_tm"]].values)) - \
        set(map(tuple, cmap[["league", "team_tm"]].values))
    unmapped_fd = set(map(tuple, lt[["league", "team"]].values)) - \
        set(map(tuple, cmap[["league", "team_fd"]].values))
    assert not unmapped_tm and not unmapped_fd


@needs("macro_monthly.csv")
def test_macro_ranges():
    m = pd.read_csv(PROCESSED / "macro_monthly.csv")
    assert m["uk_gilt_10y"].dropna().between(0, 8).all()
    assert m["fr_oat_10y"].dropna().between(-1, 8).all()
    assert m["eur_gbp"].dropna().between(0.6, 1.1).all()


@needs("tm_player_seasons.csv")
def test_player_minutes_plausible():
    p = pd.read_csv(PROCESSED / "tm_player_seasons.csv")
    lt = pd.read_csv(PROCESSED / "league_tables.csv")
    # total recorded minutes per club-season ~= mp * 11 * 90 (TM books 90/match)
    got = p.groupby(["league", "season", "club_tm"])["minutes"].sum().reset_index()
    cmap = pd.read_csv(CLUB_MAP)
    got = got.merge(cmap[["league", "team_tm", "team_fd"]],
                    left_on=["league", "club_tm"], right_on=["league", "team_tm"])
    got = got.merge(lt[["league", "season", "team", "mp"]],
                    left_on=["league", "season", "team_fd"],
                    right_on=["league", "season", "team"])
    expected = got["mp"] * 11 * 90
    ratio = got["minutes"] / expected
    assert ratio.between(0.93, 1.07).all(), \
        got.loc[~ratio.between(0.93, 1.07), ["league", "season", "club_tm"]]


@needs("panel.csv")
def test_panel_no_leakage_lags():
    panel = pd.read_csv(PROCESSED / "panel.csv")
    # a club absent in season t-1 must have NaN lags in t (no gap bridging)
    promoted = panel[panel["newly_promoted"] == 1]
    assert promoted["ppg_lag1"].isna().all()
    # and Leicester's title season must carry their 2014-15 lag correctly
    lei = panel[(panel.canonical == "Leicester City") & (panel.season == "2015-2016")]
    assert not lei.empty and lei.iloc[0]["newly_promoted"] == 0

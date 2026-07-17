"""Assemble the club-season panel dataset (~63 engineered features).

Inputs (all produced by sibling modules, all real data):
- league_tables.csv / matches.csv     football-data.co.uk (2014-15 extra season
                                      is used only for promotion/relegation flags)
- tm_club_seasons.csv                 Transfermarkt league overviews
- tm_transfers.csv                    Transfermarkt league transfer pages
- tm_player_seasons.csv               Transfermarkt performance pages (minutes)
- macro_monthly.csv                   FRED
- club_map.csv                        hand-curated cross-source name mapping
- mco_registry.yaml                   hand-curated MCO ownership registry (H2)

Financial-statement features (revenue, wages, EBITDA, debt) are ABSENT until
the Companies House key is provided — they are not zero-filled or proxied; the
validation report records the gap explicitly.

Feature blocks: performance / market-value & transfers / squad structure /
status flags / macro / one-season lags. Lags are only defined where the club
was in the panel in the immediately preceding season (no gap-bridging: a club
returning from a relegation season has NaN lags, which is the honest choice).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
HERE = Path(__file__).resolve().parent

PANEL_SEASONS = [f"{y}-{y + 1}" for y in range(2015, 2024)]
COVID_SEASONS = {"2019-2020", "2020-2021"}

# UCL qualification spots by final league position (H3 'strike').
# PL: top 4 throughout the window. Ligue 1: top 3 (3rd via playoff for most of
# the window — treated as a qualifying spot; actual group-stage participation
# is curated separately in Phase 2 where the revenue jump is priced).
UCL_SPOTS = {"premier-league": 4, "ligue-1": 3}


def _season_start(season: str) -> int:
    return int(season.split("-")[0])


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "tables": PROCESSED / "league_tables.csv",
        "matches": PROCESSED / "matches.csv",
        "tm_clubs": PROCESSED / "tm_club_seasons.csv",
        "tm_transfers": PROCESSED / "tm_transfers.csv",
        "tm_players": PROCESSED / "tm_player_seasons.csv",
        "macro": PROCESSED / "macro_monthly.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Panel inputs missing (run the scrapers first):\n  " + "\n  ".join(missing))
    data = {k: pd.read_csv(p) for k, p in paths.items()}
    data["club_map"] = pd.read_csv(HERE / "club_map.csv")
    return data


def canonicalize(df: pd.DataFrame, club_map: pd.DataFrame, name_col: str,
                 map_col: str) -> pd.DataFrame:
    """Attach canonical club names; fall back to the source name for clubs
    outside the map (2014-15-only clubs used purely for membership flags)."""
    out = df.merge(
        club_map[["league", map_col, "canonical", "coi_flag"]],
        left_on=["league", name_col], right_on=["league", map_col], how="left")
    out["canonical"] = out["canonical"].fillna(out[name_col])
    return out.drop(columns=[map_col])


# -- feature blocks -----------------------------------------------------------

def performance_block(tables: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    t = tables.copy()
    t["ppg"] = t["points"] / t["mp"]
    t["gf_pg"] = t["gf"] / t["mp"]
    t["ga_pg"] = t["ga"] / t["mp"]
    t["gd_pg"] = t["gd"] / t["mp"]
    t["win_rate"] = t["w"] / t["mp"]
    t["draw_rate"] = t["d"] / t["mp"]
    t["title_winner"] = (t["position"] == 1).astype(int)
    t["top4"] = (t["position"] <= 4).astype(int)
    t["position_pct"] = t["position"] / t.groupby(["league", "season"])["position"].transform("max")

    # Home/away splits and shots from match-level data.
    m = matches.copy()
    m["HS"] = pd.to_numeric(m.get("HS"), errors="coerce")
    m["AS"] = pd.to_numeric(m.get("AS"), errors="coerce")
    m["HST"] = pd.to_numeric(m.get("HST"), errors="coerce")
    m["AST"] = pd.to_numeric(m.get("AST"), errors="coerce")

    home = m.groupby(["league", "season", "HomeTeam"]).agg(
        home_pts=("FTR", lambda r: (3 * (r == "H") + (r == "D")).sum()),
        home_mp=("FTR", "size"),
        shots_for_h=("HS", "sum"), shots_against_h=("AS", "sum"),
        sot_for_h=("HST", "sum"),
        cs_h=("FTAG", lambda g: (g == 0).sum()),
    ).reset_index().rename(columns={"HomeTeam": "team"})
    away = m.groupby(["league", "season", "AwayTeam"]).agg(
        away_pts=("FTR", lambda r: (3 * (r == "A") + (r == "D")).sum()),
        away_mp=("FTR", "size"),
        shots_for_a=("AS", "sum"), shots_against_a=("HS", "sum"),
        sot_for_a=("AST", "sum"),
        cs_a=("FTHG", lambda g: (g == 0).sum()),
    ).reset_index().rename(columns={"AwayTeam": "team"})

    t = t.merge(home, on=["league", "season", "team"], how="left")
    t = t.merge(away, on=["league", "season", "team"], how="left")
    t["home_ppg"] = t["home_pts"] / t["home_mp"]
    t["away_ppg"] = t["away_pts"] / t["away_mp"]
    t["shots_pg"] = (t["shots_for_h"] + t["shots_for_a"]) / t["mp"]
    t["shots_against_pg"] = (t["shots_against_h"] + t["shots_against_a"]) / t["mp"]
    t["sot_pg"] = (t["sot_for_h"] + t["sot_for_a"]) / t["mp"]
    t["shot_conversion"] = t["gf"] / (t["shots_for_h"] + t["shots_for_a"])
    t["clean_sheet_rate"] = (t["cs_h"] + t["cs_a"]) / t["mp"]

    keep = ["league", "season", "team", "canonical", "coi_flag", "position",
            "mp", "w", "d", "l", "gf", "ga", "gd", "points", "deduction",
            "ppg", "gf_pg", "ga_pg", "gd_pg", "win_rate", "draw_rate",
            "title_winner", "top4", "position_pct", "home_ppg", "away_ppg",
            "shots_pg", "shots_against_pg", "sot_pg", "shot_conversion",
            "clean_sheet_rate"]
    return t[keep]


def market_value_block(tm_clubs: pd.DataFrame, tm_transfers: pd.DataFrame) -> pd.DataFrame:
    mv = tm_clubs[["league", "season", "canonical", "squad_size", "avg_age_tm",
                   "foreigners", "avg_market_value_eur_m",
                   "total_market_value_eur_m"]].copy()
    mv["foreigners_share"] = mv["foreigners"] / mv["squad_size"]
    mv["log_squad_value"] = np.log(mv["total_market_value_eur_m"])

    tr = tm_transfers.copy()
    fees = tr.groupby(["league", "season", "canonical", "direction"]).agg(
        fee_sum=("fee_eur_m", "sum"), n=("player", "size"),
        n_loans=("is_loan", "sum"), n_free=("is_free", "sum"),
    ).unstack("direction")
    fees.columns = [f"{a}_{b}" for a, b in fees.columns]
    fees = fees.reset_index()
    fees["transfer_spend_eur_m"] = fees.get("fee_sum_in", 0.0)
    fees["transfer_income_eur_m"] = fees.get("fee_sum_out", 0.0)
    fees["net_spend_eur_m"] = fees["transfer_spend_eur_m"] - fees["transfer_income_eur_m"]
    fees = fees.rename(columns={
        "n_in": "n_signings", "n_out": "n_departures",
        "n_loans_in": "loans_in", "n_loans_out": "loans_out"})
    keep = ["league", "season", "canonical", "transfer_spend_eur_m",
            "transfer_income_eur_m", "net_spend_eur_m", "n_signings",
            "n_departures", "loans_in", "loans_out"]
    out = mv.merge(fees[keep], on=["league", "season", "canonical"], how="left")
    out["net_spend_to_value"] = out["net_spend_eur_m"] / out["total_market_value_eur_m"]
    return out


def squad_structure_block(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    p = p[p["age"].notna()]
    p["minutes"] = p["minutes"].fillna(0.0)

    def agg(group: pd.DataFrame) -> pd.Series:
        used = group[group["minutes"] > 0]
        total_min = used["minutes"].sum()
        w = used["minutes"] / total_min if total_min else None
        shares = used["minutes"] / total_min if total_min else pd.Series(dtype=float)
        return pd.Series({
            "squad_mean_age": group["age"].mean(),
            "squad_age_std": group["age"].std(),
            "minutes_weighted_age": (used["age"] * w).sum() if total_min else np.nan,
            "share_min_u21": used.loc[used["age"] <= 21, "minutes"].sum() / total_min if total_min else np.nan,
            "share_min_u23": used.loc[used["age"] <= 23, "minutes"].sum() / total_min if total_min else np.nan,
            "share_min_30plus": used.loc[used["age"] >= 30, "minutes"].sum() / total_min if total_min else np.nan,
            "n_players_used": int((group["minutes"] > 0).sum()),
            "minutes_hhi": float((shares ** 2).sum()) if total_min else np.nan,
            "top5_minutes_share": shares.nlargest(5).sum() if total_min else np.nan,
            "total_minutes_recorded": total_min,
        })
    out = p.groupby(["league", "season", "canonical"]).apply(agg, include_groups=False)
    return out.reset_index()


def status_block(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Promotion/relegation from league-membership deltas + MCO/UCL/COVID flags."""
    member = {(r.league, r.season): set() for r in membership.itertuples()}
    for r in membership.itertuples():
        member[(r.league, r.season)].add(r.canonical)

    def prev_season(season: str) -> str:
        y = _season_start(season)
        return f"{y - 1}-{y}"

    def next_season(season: str) -> str:
        y = _season_start(season)
        return f"{y + 1}-{y + 2}"

    out = panel[["league", "season", "canonical", "position"]].copy()
    out["newly_promoted"] = [
        int(row.canonical not in member.get((row.league, prev_season(row.season)), set()))
        for row in out.itertuples()]
    out["relegated_after"] = [
        (int(row.canonical not in member[(row.league, next_season(row.season))])
         if (row.league, next_season(row.season)) in member else np.nan)
        for row in out.itertuples()]
    out["ucl_spot"] = (out["position"] <= out["league"].map(UCL_SPOTS)).astype(int)
    out["covid_season"] = out["season"].isin(COVID_SEASONS).astype(int)
    out["is_premier_league"] = (out["league"] == "premier-league").astype(int)

    # MCO flags from the registry (season-bounded).
    registry = yaml.safe_load((HERE / "mco_registry.yaml").read_text())
    unverified = 0
    mco_rows = []
    for group in registry["groups"]:
        for m in group["members"]:
            unverified += 0 if m.get("verified") else 1
            mco_rows.append({**m, "group": group["group"], "scope": group["scope"]})
    if unverified:
        log.warning("MCO registry: %d entries pending citation verification "
                    "(H2 must not be estimated until this is 0)", unverified)

    def mco_lookup(row) -> tuple[int, int, str]:
        year = _season_start(row.season)
        for m in mco_rows:
            if m["club"] != row.canonical or m["league"] != row.league:
                continue
            frm = _season_start(m["from_season"])
            to = _season_start(m["to_season"]) if m.get("to_season") else 9999
            if frm <= year <= to:
                return 1, int(m["scope"] == "integrated"), m["group"]
        return 0, 0, ""

    flags = [mco_lookup(r) for r in out.itertuples()]
    out["mco_flag"] = [f[0] for f in flags]
    out["mco_integrated"] = [f[1] for f in flags]
    out["mco_group"] = [f[2] for f in flags]
    return out.drop(columns=["position"])


def macro_block(macro: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    m = macro.copy()
    m["month"] = pd.to_datetime(m["month"])
    rows = []
    for season in seasons:
        y = _season_start(season)
        window = m[(m["month"] >= f"{y}-08-01") & (m["month"] <= f"{y + 1}-05-31")]
        rows.append({
            "season": season,
            "uk_gilt_10y_season": window["uk_gilt_10y"].mean(),
            "fr_oat_10y_season": window["fr_oat_10y"].mean(),
            "uk_cpi_yoy_season": window["uk_cpi_yoy"].mean(),
            "fr_cpi_yoy_season": window["fr_cpi_yoy"].mean(),
            "eur_gbp_season": window["eur_gbp"].mean(),
        })
    return pd.DataFrame(rows)


def financial_block(seasons: list[str]) -> pd.DataFrame:
    """Companies House OCR financials (UK clubs only; French clubs pending
    DNCG — columns stay NaN for them, documented in the validation report).

    fy -> season mapping: club financial years end Apr..Aug, covering the
    season that finished that summer (fy ending 2024-06-30 = season 2023-24).
    Revenue growth is computed on the filings table itself so relegation
    seasons still anchor the growth of a club's return year correctly.
    """
    path = PROCESSED / "ch_financials.csv"
    if not path.exists():
        log.warning("ch_financials.csv missing — panel built without "
                    "financial-statement features")
        return pd.DataFrame(columns=["league", "season", "canonical"])
    fin = pd.read_csv(path)
    fin = fin[fin["status"].str.startswith(("ok",))].copy()
    ends = pd.to_datetime(fin["fy_end"])
    bad_months = ~ends.dt.month.between(4, 8)
    if bad_months.any():
        log.warning("%d filings with financial years ending outside Apr-Aug "
                    "dropped from season mapping: %s", bad_months.sum(),
                    fin.loc[bad_months, ["club", "fy_end"]].values.tolist())
        fin = fin[~bad_months]
        ends = ends[~bad_months]
    fin["season"] = ends.dt.year.map(lambda y: f"{y - 1}-{y}")
    # duplicates (year-end changes produced overlapping filings): keep the
    # later, fuller filing
    fin = (fin.sort_values(["club", "season", "fy_end"])
              .drop_duplicates(["club", "season"], keep="last"))
    fin = fin.sort_values(["club", "season"])
    fin["revenue_growth_yoy"] = fin.groupby("club")["revenue"].pct_change()
    # growth only valid across consecutive seasons
    year = fin["season"].str.slice(0, 4).astype(int)
    consecutive = year.diff() == 1
    fin.loc[~consecutive.fillna(False), "revenue_growth_yoy"] = float("nan")

    out = fin.rename(columns={
        "club": "canonical",
        "revenue": "revenue_gbp",
        "staff_costs": "staff_costs_gbp",
        "operating_result": "operating_result_gbp",
        "result_for_year": "result_for_year_gbp",
    })[["canonical", "season", "revenue_gbp", "staff_costs_gbp",
        "operating_result_gbp", "result_for_year_gbp", "revenue_growth_yoy"]]
    out["wage_to_revenue"] = out["staff_costs_gbp"] / out["revenue_gbp"]
    out["operating_margin"] = out["operating_result_gbp"] / out["revenue_gbp"]
    out["log_revenue"] = np.log(out["revenue_gbp"])
    out.insert(0, "league", "premier-league")
    return out[out["season"].isin(seasons)]


LAG_FEATURES = ["ppg", "position", "points", "gd_pg", "total_market_value_eur_m",
                "net_spend_eur_m", "minutes_weighted_age", "squad_mean_age",
                "newly_promoted", "ucl_spot"]


def add_lags(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["season_start"] = p["season"].map(_season_start)
    lagged = p[["canonical", "season_start"] + LAG_FEATURES].copy()
    lagged["season_start"] += 1  # aligns season t-1 values onto season t
    lagged = lagged.rename(columns={c: f"{c}_lag1" for c in LAG_FEATURES})
    out = p.merge(lagged, on=["canonical", "season_start"], how="left")
    out["squad_value_growth_yoy"] = (
        out["total_market_value_eur_m"] / out["total_market_value_eur_m_lag1"] - 1)
    return out


def build() -> pd.DataFrame:
    d = load_inputs()
    cmap = d["club_map"]

    tables = canonicalize(d["tables"], cmap, "team", "team_fd")
    matches = d["matches"]
    tm_clubs = canonicalize(d["tm_clubs"], cmap, "team_tm", "team_tm").drop(columns=["coi_flag"])
    tm_transfers = canonicalize(d["tm_transfers"], cmap, "club_tm", "team_tm").drop(columns=["coi_flag"])
    tm_players = canonicalize(d["tm_players"], cmap, "club_tm", "team_tm").drop(columns=["coi_flag"])

    membership = tables[["league", "season", "canonical"]].drop_duplicates()

    perf = performance_block(tables, matches)
    perf = perf[perf["season"].isin(PANEL_SEASONS)]

    panel = perf.merge(market_value_block(tm_clubs, tm_transfers),
                       on=["league", "season", "canonical"], how="left")
    panel = panel.merge(squad_structure_block(tm_players),
                        on=["league", "season", "canonical"], how="left")
    panel = panel.merge(status_block(panel, membership),
                        on=["league", "season", "canonical"], how="left")
    panel = panel.merge(macro_block(d["macro"], PANEL_SEASONS), on="season", how="left")
    panel = panel.merge(financial_block(PANEL_SEASONS),
                        on=["league", "season", "canonical"], how="left")
    panel = add_lags(panel)
    panel["coi_flag"] = panel["coi_flag"].fillna(0).astype(int)
    return panel.sort_values(["league", "season", "position"], ignore_index=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    panel = build()
    n_features = len([c for c in panel.columns
                      if c not in ("league", "season", "team", "canonical", "season_start")])
    out_csv = PROCESSED / "panel.csv"
    panel.to_csv(out_csv, index=False)
    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    print(f"Wrote {out_csv}: {len(panel)} club-seasons x {n_features} features")
    print(f"Seasons: {panel.season.min()} .. {panel.season.max()}")
    print(f"COI-flagged rows: {int(panel.coi_flag.sum())} "
          f"(Chelsea + Strasbourg club-seasons)")
    print(f"MCO club-seasons: {int(panel.mco_flag.sum())}")
    pl_mask = panel.league == "premier-league"
    print(f"Financials (PL only): revenue {int(panel.loc[pl_mask, 'revenue_gbp'].notna().sum())}"
          f"/{int(pl_mask.sum())}, wage-to-revenue "
          f"{int(panel.loc[pl_mask, 'wage_to_revenue'].notna().sum())}/{int(pl_mask.sum())}. "
          "French clubs pending DNCG (documented limitation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

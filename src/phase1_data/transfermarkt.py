"""Transfermarkt scraper: squad values, transfers, player minutes/ages.

Three page families, all throttled (>=3.5s) and cached under
data/raw/transfermarkt/:

1. League overview (`/startseite/wettbewerb/{comp}/plus/?saison_id={y}`):
   one page per league-season -> club list with TM club id, squad size,
   average age, total & average squad market value.
2. League transfers (`/transfers/wettbewerb/{comp}/plus/?saison_id={y}&s_w=&leihe=1&intern=0`):
   one page per league-season -> every in/out transfer per club with age,
   position, market value, counterparty and fee (loans included: Phase 5's
   Chelsea-Strasbourg network needs them).
3. Club performance data (`/leistungsdaten/verein/{id}/reldata/{comp}%26{y}/plus/1`):
   one page per club-season -> player-level age, appearances, minutes.
   This replaces FBref player minutes (FBref blocks programmatic access,
   see README) and feeds minutes-weighted squad age features.

robots.txt note: transfermarkt.com's `User-agent: *` policy allows these
paths (only /ceapi, /quickselect, /jumplist and a nav endpoint are
disallowed); the ThrottledCachedSession checks this at runtime. Their bans on
AI *crawlers* (ClaudeBot etc.) target training-data collection bots, not
individual rate-limited research use.

Money strings ("€13.84m", "€1.02bn", "€500k", "-", "?") are parsed to float
millions of euros; unparseable values become NaN with the raw string kept.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .http_cache import ThrottledCachedSession

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LEAGUES = {
    "premier-league": {"comp": "GB1", "slug": "premier-league"},
    "ligue-1": {"comp": "FR1", "slug": "ligue-1"},
}
DEFAULT_SEASONS = [f"{y}-{y + 1}" for y in range(2015, 2024)]

BASE = "https://www.transfermarkt.com"


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def overview_url(league: str, season: str) -> str:
    cfg = LEAGUES[league]
    return (f"{BASE}/{cfg['slug']}/startseite/wettbewerb/{cfg['comp']}"
            f"/plus/?saison_id={season_start_year(season)}")


def transfers_url(league: str, season: str) -> str:
    cfg = LEAGUES[league]
    return (f"{BASE}/{cfg['slug']}/transfers/wettbewerb/{cfg['comp']}"
            f"/plus/?saison_id={season_start_year(season)}&s_w=&leihe=1&intern=0")


def leistungsdaten_url(club_slug: str, club_id: int, league: str, season: str) -> str:
    cfg = LEAGUES[league]
    return (f"{BASE}/{club_slug}/leistungsdaten/verein/{club_id}"
            f"/reldata/{cfg['comp']}%26{season_start_year(season)}/plus/1")


# -- low-level parsing helpers ------------------------------------------------

def parse_money_eur_m(text: str) -> float:
    """'€13.84m' -> 13.84, '€1.02bn' -> 1020.0, '€500k' -> 0.5, else NaN."""
    if not text:
        return float("nan")
    m = re.search(r"€([\d.]+)(bn|m|k)", text.replace(",", "."))
    if not m:
        return float("nan")
    value = float(m.group(1))
    return value * {"bn": 1000.0, "m": 1.0, "k": 0.001}[m.group(2)]


def parse_minutes(text: str) -> float:
    """"2.020'" -> 2020.0 (dot = thousands separator); '-' -> 0."""
    cleaned = text.replace(".", "").replace("'", "").strip()
    return float(cleaned) if cleaned.isdigit() else 0.0


def _club_id_slug(href: str) -> tuple[int, str] | None:
    m = re.match(r"/([^/]+)/(?:startseite|kader|transfers)/verein/(\d+)", href or "")
    return (int(m.group(2)), m.group(1)) if m else None


# -- page parsers --------------------------------------------------------------

def parse_league_overview(html: str, league: str, season: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="items")
    rows = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 7:
            continue
        link = cells[1].find("a")
        ident = _club_id_slug(link.get("href")) if link else None
        if ident is None:
            continue
        rows.append({
            "league": league,
            "season": season,
            "team_tm": link.get_text(strip=True),
            "tm_club_id": ident[0],
            "tm_slug": ident[1],
            "squad_size": int(cells[2].get_text(strip=True) or 0),
            "avg_age_tm": float(cells[3].get_text(strip=True) or "nan"),
            "foreigners": int(cells[4].get_text(strip=True) or 0),
            "avg_market_value_eur_m": parse_money_eur_m(cells[5].get_text(strip=True)),
            "total_market_value_eur_m": parse_money_eur_m(cells[6].get_text(strip=True)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No clubs parsed from overview {league} {season}")
    return df


def parse_league_transfers(html: str, league: str, season: str) -> pd.DataFrame:
    """Each club section ('box') holds an In table and an Out table."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for box in soup.find_all("div", class_="box"):
        h2 = box.find("h2")
        link = h2.find("a") if h2 else None
        if not link:
            continue
        ident = _club_id_slug(link.get("href"))
        club = link.get_text(strip=True) or h2.get_text(strip=True)
        for table in box.find_all("table"):
            ths = [th.get_text(strip=True) for th in table.find_all("th")]
            if not ths or ths[0] not in ("In", "Out"):
                continue
            direction = ths[0].lower()
            tbody = table.find("tbody")
            if tbody is None:
                continue
            for tr in tbody.find_all("tr", recursive=False):
                cells = tr.find_all("td", recursive=False)
                if len(cells) < 8:
                    continue  # 'No arrivals/departures' filler rows
                player_link = cells[0].find("a")
                fee_text = cells[8].get_text(" ", strip=True) if len(cells) > 8 \
                    else cells[-1].get_text(" ", strip=True)
                rows.append({
                    "league": league,
                    "season": season,
                    "club_tm": club,
                    "tm_club_id": ident[0] if ident else None,
                    "direction": direction,
                    "player": (player_link.get_text(strip=True) if player_link
                               else cells[0].get_text(" ", strip=True)),
                    "age": pd.to_numeric(cells[1].get_text(strip=True), errors="coerce"),
                    "position": cells[3].get_text(strip=True),
                    "market_value_eur_m": parse_money_eur_m(cells[5].get_text(strip=True)),
                    "counterparty": cells[7].get_text(" ", strip=True),
                    "fee_raw": fee_text,
                    "fee_eur_m": parse_money_eur_m(fee_text),
                    "is_loan": bool(re.search(r"loan", fee_text, re.I)),
                    "is_free": bool(re.search(r"free", fee_text, re.I)),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No transfers parsed for {league} {season}")
    return df


def parse_leistungsdaten(
    html: str, league: str, season: str, club_tm: str, tm_club_id: int
) -> pd.DataFrame:
    """Player rows: age, in-squad, appearances, minutes (last column)."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="items")
    if table is None:
        raise ValueError(f"No player table for {club_tm} {season}")
    rows = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 10:
            continue
        name_link = cells[1].find("a")
        position_td = cells[1].find_all("tr")
        position = (position_td[-1].get_text(strip=True) if position_td else "")
        appearances = pd.to_numeric(cells[5].get_text(strip=True), errors="coerce")
        rows.append({
            "league": league,
            "season": season,
            "club_tm": club_tm,
            "tm_club_id": tm_club_id,
            "player": name_link.get_text(strip=True) if name_link else "",
            "position": position,
            "age": pd.to_numeric(cells[2].get_text(strip=True), errors="coerce"),
            "in_squad": pd.to_numeric(cells[4].get_text(strip=True), errors="coerce"),
            "appearances": 0 if pd.isna(appearances) else appearances,
            "minutes": parse_minutes(cells[-1].get_text(strip=True)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No players parsed for {club_tm} {season}")
    return df


# -- orchestration -------------------------------------------------------------

def scrape_overviews(leagues, seasons, session) -> pd.DataFrame:
    frames = []
    for league in leagues:
        for season in seasons:
            log.info("TM overview: %s %s", league, season)
            frames.append(parse_league_overview(
                session.get(overview_url(league, season)), league, season))
    return pd.concat(frames, ignore_index=True)


def scrape_transfers(leagues, seasons, session) -> pd.DataFrame:
    frames = []
    for league in leagues:
        for season in seasons:
            log.info("TM transfers: %s %s", league, season)
            frames.append(parse_league_transfers(
                session.get(transfers_url(league, season)), league, season))
    return pd.concat(frames, ignore_index=True)


def scrape_players(clubs: pd.DataFrame, session) -> pd.DataFrame:
    """`clubs` = output of scrape_overviews (needs tm_slug/tm_club_id)."""
    frames = []
    failures = []
    for row in clubs.itertuples():
        log.info("TM players: %s %s %s", row.league, row.season, row.team_tm)
        try:
            html = session.get(leistungsdaten_url(
                row.tm_slug, row.tm_club_id, row.league, row.season))
            frames.append(parse_leistungsdaten(
                html, row.league, row.season, row.team_tm, row.tm_club_id))
        except Exception as exc:  # noqa: BLE001 - collect, report, fail loudly at end
            log.error("FAILED %s %s %s: %s", row.league, row.season, row.team_tm, exc)
            failures.append((row.league, row.season, row.team_tm, str(exc)))
    if failures:
        print(f"\n{len(failures)} club-season player pages FAILED:")
        for f in failures:
            print("  ", f)
    if not frames:
        raise RuntimeError("No player data scraped at all")
    df = pd.concat(frames, ignore_index=True)
    df.attrs["failures"] = failures
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=["overview", "transfers", "players", "all"],
                        default="all")
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES),
                        choices=list(LEAGUES))
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    session = ThrottledCachedSession(RAW_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    n_failures = 0
    if args.what in ("overview", "all"):
        clubs = scrape_overviews(args.leagues, args.seasons, session)
        out = PROCESSED_DIR / "tm_club_seasons.csv"
        clubs.to_csv(out, index=False)
        print(f"Wrote {out} ({len(clubs)} rows)")

    if args.what in ("transfers", "all"):
        transfers = scrape_transfers(args.leagues, args.seasons, session)
        out = PROCESSED_DIR / "tm_transfers.csv"
        transfers.to_csv(out, index=False)
        print(f"Wrote {out} ({len(transfers)} rows)")

    if args.what in ("players", "all"):
        clubs = pd.read_csv(PROCESSED_DIR / "tm_club_seasons.csv")
        clubs = clubs[clubs["league"].isin(args.leagues)
                      & clubs["season"].isin(args.seasons)]
        players = scrape_players(clubs, session)
        out = PROCESSED_DIR / "tm_player_seasons.csv"
        players.to_csv(out, index=False)
        print(f"Wrote {out} ({len(players)} rows)")
        n_failures = len(players.attrs.get("failures", []))

    return 1 if n_failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Companies House client: UK club financial statements.

Requires COMPANIES_HOUSE_API_KEY (free registration at
https://developer.company-information.service.gov.uk). Auth is HTTP Basic
with the key as username. Official rate limit is 600 requests / 5 minutes;
we throttle to ~1 req/s and cache everything under data/raw/companies_house/.

Pipeline (subcommands):
  registry  search candidate companies for every UK club in club_map.csv and
            write ch_registry_candidates.csv for human verification. The
            verified mapping lives in CH_REGISTRY below (club -> company
            number), because picking the right legal entity (operating company
            vs holding vs plc) is a judgment call that must not be automated.
  accounts  for each verified club: list accounts filings, download the iXBRL
            (xhtml) document per financial year where available, parse core
            line items with ixbrlparse -> ch_financials.csv.

Coverage honesty: iXBRL tagging of full accounts only became widespread in
the late 2010s; some filings are PDF-only and are recorded as missing
(reason='pdf_only'), never guessed. French clubs are out of scope here (DNCG
reports instead, later step).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

from .http_cache import ThrottledCachedSession

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "companies_house"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

API = "https://api.company-information.service.gov.uk"
DOC_API = "https://document-api.company-information.service.gov.uk"

# Verified club -> Companies House number. Populated from the `registry`
# subcommand's candidates AFTER manual verification (SIC 93120, registered
# office, incorporation era, and — decisively — whether its filed accounts
# carry the club's football turnover). Empty entries are pending.
CH_REGISTRY: dict[str, str] = {
    # PILOT SET — verified from search candidates (operating companies, all
    # century-old incorporations matching the clubs' founding eras except
    # Chelsea, whose operating entity dates from the 1980s restructuring).
    # Parsed revenues are validated against publicly reported figures before
    # the registry is extended to all clubs.
    "Manchester United": "00095489",  # MANCHESTER UNITED FOOTBALL CLUB LIMITED, inc. 1907
    "Liverpool": "00035668",          # THE LIVERPOOL FC AND ATHLETIC GROUNDS LTD, inc. 1892
    "Everton": "00036624",            # EVERTON FOOTBALL CLUB COMPANY, LIMITED, inc. 1892
    "Brighton": "00081077",           # BRIGHTON AND HOVE ALBION FC LIMITED, inc. 1904
    "Chelsea": "02536231",            # CHELSEA FC HOLDINGS LIMITED — consolidated football
                                      # accounts (club-level 01965149 showed GBP 474.8m vs
                                      # published GBP 512.5m FY23, i.e. excludes women's/tours)
    # Extended set. Entity choice = the operating company that carries the
    # club's football turnover; every pick is gated downstream by the
    # public-figure revenue benchmarks (wrong entity => benchmark failure).
    "Arsenal": "00109244",            # THE ARSENAL FOOTBALL CLUB LIMITED, inc. 1910
    "Aston Villa": "03375789",        # ASTON VILLA FOOTBALL CLUB LIMITED
    "Bournemouth": "06632170",        # AFC BOURNEMOUTH LIMITED (trading co; 02393821 files filleted accounts)
    "Brentford": "03642327",          # BRENTFORD FC LIMITED
    "Burnley": "00054222",            # BURNLEY FOOTBALL & ATHLETIC COMPANY LTD, inc. 1897
    "Cardiff City": "00109065",       # CARDIFF CITY FOOTBALL CLUB LIMITED, inc. 1910
    "Crystal Palace": "07270793",     # CPFC LIMITED (operating co post-2010)
    "Fulham": "02114486",             # FULHAM FOOTBALL CLUB LIMITED
    "Huddersfield": "01771361",       # HUDDERSFIELD TOWN AFC LIMITED
    "Hull City": "04032392",          # HULL CITY TIGERS LIMITED (operating co)
    "Leeds United": "06233875",       # LEEDS UNITED FOOTBALL CLUB LIMITED
    "Leicester City": "04593477",     # LEICESTER CITY FOOTBALL CLUB LIMITED
    "Luton Town": "06133975",         # LUTON TOWN FOOTBALL CLUB 2020 LTD
    "Manchester City": "00040946",    # MANCHESTER CITY FOOTBALL CLUB LIMITED, inc. 1894
    "Middlesbrough": "01947851",      # MIDDLESBROUGH F&A CO (1986) LIMITED
    "Newcastle United": "02529667",   # NEWCASTLE UNITED LIMITED (group; club-level 05981582 files s479A exemption statements only)
    "Norwich City": "00154044",       # NORWICH CITY FOOTBALL CLUB PLC, inc. 1919
    "Nottingham Forest": "01630402",  # NOTTINGHAM FOREST FOOTBALL CLUB LIMITED
    "Sheffield United": "00061564",   # THE SHEFFIELD UNITED FOOTBALL CLUB LIMITED, inc. 1899
    "Southampton": "00053301",        # SOUTHAMPTON FOOTBALL CLUB LIMITED, inc. 1897
    "Stoke City": "00099885",         # STOKE CITY FOOTBALL CLUB LIMITED, inc. 1908
    "Sunderland": "00049116",         # SUNDERLAND AFC LIMITED, inc. 1896
    "Swansea City": "04305508",       # SWANSEA CITY FOOTBALL 2002 LIMITED
    "Tottenham Hotspur": "01706358",  # TOTTENHAM HOTSPUR LIMITED (group reporting entity)
    "Watford": "00104194",            # WATFORD ASSOCIATION FOOTBALL CLUB LIMITED, inc. 1909
    "West Brom": "07230595",          # WEST BROMWICH ALBION GROUP LIMITED (consolidated)
    "West Ham": "00066516",           # WEST HAM UNITED FOOTBALL CLUB LIMITED, inc. 1900
    "Wolves": "01989823",             # WOLVERHAMPTON WANDERERS FC (1986) LIMITED
}


def require_api_key() -> str:
    key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise SystemExit(
            "COMPANIES_HOUSE_API_KEY is not set. Register a free key at "
            "https://developer.company-information.service.gov.uk/ and export it."
        )
    return key.strip()


def make_session() -> ThrottledCachedSession:
    return ThrottledCachedSession(RAW_DIR, min_delay=1.0,
                                  auth=(require_api_key(), ""))


# -- registry construction ----------------------------------------------------

def search_candidates(session: ThrottledCachedSession) -> pd.DataFrame:
    cmap = pd.read_csv(Path(__file__).resolve().parent / "club_map.csv")
    uk = cmap[cmap.league == "premier-league"]
    rows = []
    for club in uk.canonical:
        q = f"{club} football club"
        url = f"{API}/search/companies?q={q.replace(' ', '+')}&items_per_page=10"
        payload = json.loads(session.get(url))
        for item in payload.get("items", []):
            rows.append({
                "club": club,
                "title": item.get("title"),
                "company_number": item.get("company_number"),
                "status": item.get("company_status"),
                "incorporated": item.get("date_of_creation"),
                "address": (item.get("address_snippet") or "")[:60],
            })
    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "ch_registry_candidates.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} candidates for {df.club.nunique()} clubs) — "
          "verify manually, then fill CH_REGISTRY.")
    return df


def profile(session: ThrottledCachedSession, number: str) -> dict:
    return json.loads(session.get(f"{API}/company/{number}"))


# -- accounts download + parsing ------------------------------------------------

def accounts_filings(session: ThrottledCachedSession, number: str) -> list[dict]:
    url = f"{API}/company/{number}/filing-history?category=accounts&items_per_page=100"
    payload = json.loads(session.get(url))
    return [f for f in payload.get("items", [])
            if f.get("type", "").startswith("AA")]  # annual accounts


def fetch_ixbrl(session: ThrottledCachedSession, filing: dict) -> str | None:
    """Return the filing's iXBRL xhtml, or None if only PDF exists."""
    meta_link = filing.get("links", {}).get("document_metadata")
    if not meta_link:
        return None
    meta = json.loads(session.get(meta_link))
    if "application/xhtml+xml" not in meta.get("resources", {}):
        return None
    return session.get(meta["links"]["document"] + "?fmt=xhtml")


# Concept names differ across taxonomy vintages (FRS101/102/full IFRS).
CONCEPTS = {
    "revenue": {"Turnover", "TurnoverRevenue", "Revenue", "RevenueFromContractsWithCustomers"},
    "staff_costs": {"StaffCosts", "StaffCostsEmployeeBenefitsExpense", "WagesSalaries"},
    "operating_profit": {"OperatingProfitLoss"},
    "depreciation_amortisation": {
        "DepreciationAmortisationImpairmentExpense",
        "DepreciationAndAmortisationExpense", "DepreciationImpairmentExpense"},
    "creditors_after_one_year": {
        "CreditorsAmountsFallingDueAfterOneYear", "Creditors"},
    "cash": {"CashBankOnHand", "CashCashEquivalents"},
}


def parse_ixbrl_facts(xhtml: str) -> dict[str, float]:
    from ixbrlparse import IXBRL
    doc = IXBRL(BytesIO(xhtml.encode("utf-8")))
    # keep the fact with the LATEST instant/period per concept (= current FY,
    # not the comparative column), largest absolute value as tie-break
    facts: dict[str, tuple] = {}
    for n in doc.numeric:
        name = n.name
        for ours, aliases in CONCEPTS.items():
            if name in aliases and n.value is not None:
                key_date = str(getattr(n.context, "instant", None)
                               or getattr(n.context, "enddate", ""))
                cur = facts.get(ours)
                cand = (key_date, abs(n.value), float(n.value))
                if cur is None or cand[:2] > cur[:2]:
                    facts[ours] = cand
    return {k: v[2] for k, v in facts.items()}


def pull_accounts(session: ThrottledCachedSession) -> pd.DataFrame:
    if not CH_REGISTRY:
        raise SystemExit("CH_REGISTRY is empty — run the `registry` subcommand "
                         "and verify company numbers first.")
    rows = []
    for club, number in CH_REGISTRY.items():
        for filing in accounts_filings(session, number):
            made_up_to = filing.get("description_values", {}).get("made_up_date") \
                or filing.get("action_date")
            try:
                xhtml = fetch_ixbrl(session, filing)
            except Exception as exc:  # noqa: BLE001
                log.error("%s %s: document fetch failed: %s", club, made_up_to, exc)
                xhtml = None
            if xhtml is None:
                rows.append({"club": club, "company_number": number,
                             "made_up_to": made_up_to, "available": 0,
                             "reason": "pdf_only_or_missing"})
                continue
            facts = parse_ixbrl_facts(xhtml)
            rows.append({"club": club, "company_number": number,
                         "made_up_to": made_up_to, "available": 1, "reason": "",
                         **facts})
            log.info("%s %s: %s", club, made_up_to,
                     {k: round(v / 1e6, 1) for k, v in facts.items()})
    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "ch_financials.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} filings, {df.available.sum()} parsed)")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["registry", "accounts"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    session = make_session()
    if args.what == "registry":
        search_candidates(session)
    else:
        pull_accounts(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())

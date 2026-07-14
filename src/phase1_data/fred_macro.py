"""Macro series for discount-rate inputs, from FRED.

Series (all monthly unless noted):
- IRLTLT01GBM156N  UK 10-year gilt yield, % p.a.  (risk-free base, UK clubs)
- IRLTLT01FRM156N  France 10-year OAT yield, %    (risk-free base, FR clubs)
- CPALTT01GBM659N  UK CPI inflation, % y/y
- CPALTT01FRM659N  France CPI inflation, % y/y
- DEXUSUK          USD per GBP (daily -> monthly mean)
- DEXUSEU          USD per EUR (daily -> monthly mean)
  -> eur_gbp cross rate, needed to reconcile Transfermarkt EUR values with
     Companies House GBP financials.

Access: if FRED_API_KEY is set we use the official API; otherwise we fall
back to FRED's public `fredgraph.csv` endpoint (no key required, same data).
Both paths are cached under data/raw/fred/.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from .http_cache import ThrottledCachedSession

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fred"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SERIES = {
    "IRLTLT01GBM156N": "uk_gilt_10y",
    "IRLTLT01FRM156N": "fr_oat_10y",
    "CPALTT01GBM659N": "uk_cpi_yoy",
    "CPALTT01FRM659N": "fr_cpi_yoy",
    "DEXUSUK": "usd_per_gbp",
    "DEXUSEU": "usd_per_eur",
}

START = "2010-01-01"  # some history pre-panel for smoothing/lags


def fetch_series(series_id: str, session: ThrottledCachedSession) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        url = ("https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={api_key}&file_type=json"
               f"&observation_start={START}")
        import json
        payload = json.loads(session.get(url))
        obs = pd.DataFrame(payload["observations"])
        s = pd.Series(pd.to_numeric(obs["value"], errors="coerce").values,
                      index=pd.to_datetime(obs["date"]), name=series_id)
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
        df = pd.read_csv(io.StringIO(session.get(url)))
        date_col = df.columns[0]  # 'DATE' or 'observation_date' depending on vintage
        s = pd.Series(pd.to_numeric(df[series_id], errors="coerce").values,
                      index=pd.to_datetime(df[date_col]), name=series_id)
    s = s.dropna()
    if s.empty:
        raise ValueError(f"FRED series {series_id} came back empty")
    return s


def build(session: ThrottledCachedSession | None = None) -> pd.DataFrame:
    session = session or ThrottledCachedSession(RAW_DIR, min_delay=1.0)
    monthly = {}
    for series_id, name in SERIES.items():
        log.info("FRED: %s (%s)", series_id, name)
        s = fetch_series(series_id, session)
        monthly[name] = s.resample("MS").mean()  # daily FX -> monthly mean
    df = pd.DataFrame(monthly)
    df["eur_gbp"] = df["usd_per_eur"] / df["usd_per_gbp"]  # GBP per EUR
    df.index.name = "month"
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    df = build()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "macro_monthly.csv"
    df.to_csv(out)
    print(f"Wrote {out} ({len(df)} months, {df.index.min():%Y-%m} .. {df.index.max():%Y-%m})")
    print(df.tail(3).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

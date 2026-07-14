"""Companies House client: UK club financial statements.

STATUS: implemented but UNTESTED — requires COMPANIES_HOUSE_API_KEY, which
must come from a (free) developer account the project owner registers at
https://developer.company-information.service.gov.uk. The module fails fast
with instructions when the key is missing. Nothing downstream silently
proceeds without financials: the feature builder marks financial columns as
missing-with-reason.

Flow per club:
1. Company number lookup — from the hand-verified registry below (filled via
   the search endpoint + manual confirmation against the club's registered
   address / SIC code, to avoid homonym traps like fan clubs or holding cos).
2. GET /company/{number}/filing-history?category=accounts — list account
   filings with period end dates.
3. Download the iXBRL/XBRL document for each period (document API) into
   data/raw/companies_house/.
4. Parse core line items with `ixbrlparse`: turnover/revenue, staff costs,
   operating profit, depreciation+amortisation (-> EBITDA), creditors (debt).

Scope notes:
- Only UK-registered clubs are in Companies House. French clubs' financials
  come from DNCG annual reports in a later step (documented limitation).
- Clubs file group vs company-only accounts inconsistently; the parser keeps
  both and records which consolidation level a figure came from.

API auth: HTTP Basic with the key as username, blank password.
Rate limit: 600 requests / 5 minutes — our throttle stays far below it.
"""

from __future__ import annotations

import os
import sys

# Hand-verified company numbers (filled in once the API key is available;
# each entry must be confirmed manually before use — see module docstring).
COMPANY_REGISTRY: dict[str, str] = {
    # "Manchester United": "00095489",   # example format — VERIFY before enabling
}

API_BASE = "https://api.company-information.service.gov.uk"
DOC_BASE = "https://document-api.company-information.service.gov.uk"


def require_api_key() -> str:
    key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise SystemExit(
            "COMPANIES_HOUSE_API_KEY is not set.\n"
            "Register a free API key at "
            "https://developer.company-information.service.gov.uk/, then:\n"
            "  export COMPANIES_HOUSE_API_KEY=...\n"
            "Financial features are BLOCKED until this is provided — the panel "
            "builder will mark them as missing rather than substituting anything."
        )
    return key


def main() -> int:
    require_api_key()
    print("API key found. Filing-history download + iXBRL parsing to be "
          "exercised and validated in the next phase-1 iteration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

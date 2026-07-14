"""Phase 1 entry point: regenerates every Phase 1 output from scratch.

All scrapers are cache-first, so a re-run touches the network only for pages
not yet cached (a truly clean machine re-scrapes everything, throttled).

    .venv/bin/python -m src.phase1_data.run_phase1
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from . import build_panel, football_data, fred_macro, transfermarkt, validate_phase1

    print("=" * 70, "\nPhase 1 — data architecture\n", "=" * 70)

    print("\n[1/6] football-data.co.uk results + standings")
    if football_data.main([]) != 0:
        return 1

    print("\n[2/6] Transfermarkt (overview, transfers, players)")
    if transfermarkt.main(["--what", "all"]) != 0:
        return 1

    print("\n[3/6] FRED macro series")
    if fred_macro.main() != 0:
        return 1

    print("\n[4/6] Companies House financials")
    if os.environ.get("COMPANIES_HOUSE_API_KEY"):
        print("  key present — module to be exercised in next iteration")
    else:
        print("  SKIPPED: COMPANIES_HOUSE_API_KEY not set (see README). "
              "Panel ships without financial-statement features.")

    print("\n[5/6] Panel assembly")
    if build_panel.main() != 0:
        return 1

    print("\n[6/6] Validation (A/B layers, then R, then report)")
    r = subprocess.run(["Rscript", str(PROJECT_ROOT / "r" / "phase1_validation.R")],
                       cwd=PROJECT_ROOT)
    if r.returncode != 0:
        print("R validation FAILED")
        return 1
    return validate_phase1.main()


if __name__ == "__main__":
    sys.exit(main())

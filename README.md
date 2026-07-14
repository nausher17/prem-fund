# Multi-Club Investment Fund — Quantitative Valuation & Portfolio Simulation of Premier League Clubs

A portfolio-grade quantitative finance project that models Premier League and Ligue 1
football clubs as alternative assets: fundamental valuation (DCF, comparable
transactions, real options), econometric tests of four market-inefficiency hypotheses,
and construction/backtesting of an optimised club portfolio targeting a 15% IRR.

## Investment hypotheses

| # | Hypothesis | Prior expectation |
|---|-----------|-------------------|
| H1 | Newly promoted clubs are systematically overvalued in their first PL season | To be tested |
| H2 | Multi-club ownership (MCO) structures carry a valuation premium | Likely null — reported honestly |
| H3 | The embedded real option of UCL qualification is mispriced by linear expected-value thinking | Likely null — reported honestly |
| H4 | Squad demographic structure and on-pitch performance predict future revenue growth | Expected strongest result |

## Conflict-of-interest disclosure

The author is a **Chelsea FC supporter**. Chelsea and RC Strasbourg (both owned by
BlueCo, and jointly the subject of the Phase 5 multi-club-ownership case study) are
**flagged in the dataset with `coi_flag = 1`, not excluded**. All analysis touching
these clubs should be read with this disclosure in mind; the flag allows any reader to
re-run results excluding them.

## Data sources & scraping ethics

All scraping is throttled (≥3.5s between live requests per host), cached under
`data/raw/` (re-runs never re-scrape), and robots.txt-checked at runtime before
every live fetch.

- **football-data.co.uk** — canonical match-result CSVs (E0/F1); league tables are
  computed from results, with the 2023–24 PL point deductions (Everton −8,
  Nottingham Forest −4) applied and sourced in `football_data.py`.
- **Transfermarkt** — squad market values and average ages (league overview pages),
  transfers incl. loans (league transfer pages), player ages/appearances/minutes
  (club performance-data pages). Its robots.txt permits these paths for generic
  agents; its AI-crawler bans target training-data bots, not rate-limited personal
  research use.
- **FRED** — gilt/OAT yields, CPI, FX for discount-rate inputs. Uses the official
  API when `FRED_API_KEY` is set, else FRED's public `fredgraph.csv` endpoint.
- **Companies House API** — UK club financial statements. Requires
  `COMPANIES_HOUSE_API_KEY` (free registration); the pipeline fails fast with
  instructions if missing. French club financials (DNCG reports) are a documented
  later step.

**Source changes from the original design (2026-07-14):**

- **FBref was dropped**: fbref.com now returns HTTP 403 to all non-browser clients
  (including for robots.txt itself). Rather than spoof browser fingerprints, we
  replaced it with football-data.co.uk (results) + Transfermarkt (player minutes/
  ages) — same variables, equally real data.
- **Understat was evaluated and rejected**: its robots.txt disallows all crawling
  (`User-agent: * / Disallow: /`), so no xG features are used.
- **Real data only.** No synthetic or demo data anywhere in the pipeline. If a source
  is unavailable, the pipeline stops and says so.

## Repository layout

```
prem-fund/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # cached scrapes + API pulls (gitignored)
│   ├── processed/    # cleaned panel dataset
│   └── validation/   # R cross-check outputs
├── src/
│   ├── phase1_data/        # scrapers, API clients, feature engineering
│   ├── phase2_valuation/   # DCF, comps, real options
│   ├── phase3_hypotheses/  # panel econometrics + ML suite
│   ├── phase4_portfolio/   # optimisation, VaR/CVaR, Monte Carlo (C++), backtest
│   ├── phase5_blueco/      # Chelsea–Strasbourg network case study
│   └── phase6_dashboard/   # Streamlit app + reporting
├── cpp/                    # Monte Carlo engine source + build files
├── r/                      # R validation scripts
├── outputs/                # figures, tables, phase archives
└── tests/                  # unit + integration tests
```

## How to run

```bash
# Python 3.12 environment (managed with uv, but plain venv works too)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Phase 1 — data layer
.venv/bin/python -m src.phase1_data.run_phase1
```

Each phase has a single entry-point script that regenerates all of its outputs from
scratch (from cache where scrapes are involved). Phase archives are written to
`outputs/phaseN.tar.gz`.

## Methodology principles

1. Temporal train/test splits only — never random splits on panel data.
2. Panel regressions use club fixed effects with standard errors clustered by club.
3. Key Python results are re-verified in R (`r/` directory).
4. All randomness is seeded; dependencies are pinned.
5. Null results are reported as findings, not buried.

## Status

- [x] Phase 1 — Data architecture *(complete 2026-07-14; one open item: financial-statement
      features pending `COMPANIES_HOUSE_API_KEY` — see `data/validation/phase1_validation_report.md`)*
- [ ] Phase 2 — Valuation engine
- [ ] Phase 3 — Hypothesis testing
- [ ] Phase 4 — Portfolio construction & risk
- [ ] Phase 5 — BlueCo case study
- [ ] Phase 6 — Dashboard & reporting

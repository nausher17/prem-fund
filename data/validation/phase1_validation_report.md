# Phase 1 validation report — 2026-07-17

## Layer A — schema/range
- **league sizes (20 clubs; L1 2023-24 has 18)**: PASS {}
- **points identity 3w+d**: PASS
- **matches identity w+d+l == mp**: PASS
- **aggregate GF == GA per league-season**: PASS
- **no negative goals/points**: PASS
- **squad values in [10, 2500] EUR m**: PASS range 18-1460
- **TM avg ages in [20, 32]**: PASS
- **gilt yields in [0, 8]%**: PASS

## Layer B — cross-source reconciliation
- **club universes identical across sources (358 club-seasons)**: PASS fd-only=0, tm-only=0
- **player minutes reconcile with matches played (±7%)**: PASS min ratio 0.990, max 1.001, n=358
- **squad mean age vs TM published avg age (corr > 0.75)**: PASS corr=0.998
- **PL revenue coverage = 100%**: PASS 100%
- **wage-to-revenue in [0.15, 1.7] (sector-plausible)**: PASS median 0.61, n=155
- **cited public revenue benchmarks (4 clubs)**: PASS []
- **no hard prior-year mismatches in panel-relevant filings**: PASS 0 flagged
- **French club financials (DNCG annual reports)**: OPEN not in Companies House; separate cited-extraction step — financial features are PL-only until then
- **staff-costs measure caveat**: OPEN 20/180 PL club-seasons lack a machine-readable total staff costs row; where only 'wages and salaries' was readable it is used (excludes social security/pensions) — flagged per row in ch_financials.csv

## Layer C — R cross-validation
- **R re-computation agrees with Python**: PASS standings recomputation (398 club-seasons)=OK; champions vs public record (18 league-seasons)=OK; panel ppg consistency=OK; minutes-weighted age plausible range=OK; squad value plausible range=OK

## Known open items
- Financial-statement features pending `COMPANIES_HOUSE_API_KEY` (free registration; see README). Panel intentionally ships without revenue/wages/EBITDA/debt columns rather than proxying them.
- French club financials will come from DNCG annual reports (manual, cited extraction) in a later Phase 1 iteration.

## Conflict of interest
- Author is a Chelsea supporter. Chelsea and Strasbourg carry `coi_flag = 1` in the panel (flagged, never excluded).
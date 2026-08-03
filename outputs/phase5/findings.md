# Phase 5 — BlueCo natural experiment: Chelsea & RC Strasbourg

> **Conflict-of-interest disclosure (restated):** the author is a Chelsea
> supporter. Both clubs carry `coi_flag = 1` throughout the project and are
> flagged, never excluded. Every number below is reproducible from cached
> Transfermarkt pages via `src/phase5_blueco/case_study.py`.

## 1. The player-pathway network (2023-24 .. 2025-26)

16 Chelsea↔Strasbourg movements since BlueCo acquired Strasbourg (June 2023;
€75m, 99.97%): predominantly **young loanees** (Andrey Santos 19, Ângelo 18,
Anselmino 20, Páez 18, Penders 19, Sarr 19, Fofana 23) plus permanent sales
(Amougou €14.5m, Sarr €14.0m) and veteran free moves (Chilwell). The
direction is one-way development traffic: Chelsea places teenagers in
Ligue 1 minutes, then recalls or crystallises fees. Full table:
`network.csv`; visual: `network_fig.png`.

## 2. Trajectory vs matched control (nearest-neighbour: Nantes)

Control selected by 2019-22 squad-value + league-position distance among
non-MCO Ligue 1 clubs (lightweight synthetic-control design; full donor-pool
weighting out of scope, stated).

| squad value (EUR m) | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| **Strasbourg** | 123.6 | 157.7 | 307.8 | **466.1** |
| Nantes (control) | 119.3 | 125.9 | 114.6 | 122.6 |

Post-acquisition divergence: **+277% vs +3%** (2022→2025). Average squad age
fell 26.0 → 22.0 (Nantes: ~26, flat) — the youngest squad profile in our
Ligue 1 universe, unambiguously a deliberate portfolio-asset strategy.
Figure: `trajectory_fig.png`.

## 3. Reconciling with H2's econometric null

H2 found no MCO valuation premium (β = +0.089, p = 0.18). The case study
explains rather than contradicts the null:

- **Timing**: the analysis panel ends 2023-24 — exactly one post-acquisition
  Strasbourg season. The +277% divergence happens in 2024-26, outside the
  estimation window. Synergies are real but too recent to be identified by
  within-club panel variation through 2024.
- **Mechanism**: the observable synergy is *asset-value* accumulation
  (young-player market values), not immediate revenue/valuation premium —
  the H2 spec's outcome. Squad value is the leading indicator; revenue and
  club valuations lag.
- **Selection caveat**: one treated club, chosen by the owner precisely for
  this strategy — the case study is existence evidence, not a causal
  average effect.

## 4. Limitations

Transfermarkt market values are crowd estimates (consistent cross-section,
imperfect levels); loan wage-sharing terms are unobservable; DNCG financials
for Strasbourg remain the open item that would let the revenue channel be
tested directly.

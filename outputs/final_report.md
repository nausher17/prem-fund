# Multi-Club Investment Fund
## Quantitative valuation and portfolio simulation of Premier League and Ligue 1 football clubs

*Final report — 2026-07-19. Written for a reader with finance training and no
football knowledge. Every table regenerates from the phase entry points in
this repository; all randomness is seeded; results are cross-validated in R.*

---

## Executive summary

We treat top-division English and French football clubs as illiquid
alternative assets and ask whether the market for them is efficient. Using a
hand-built panel of 358 club-seasons (2015-16..2023-24, ~80 engineered
features), statutory financials OCR-extracted from ~300 scanned Companies
House filings, and three independent valuation lenses, we find:

1. **A large, quantifiable "trophy-asset premium".** Discounted cash flows
   explain only about a third of what the market believes elite clubs are
   worth: Forbes valuations sit at a **median 2.98×** our DCF (Manchester
   United gap: £3.8bn). Clubs are consumption/prestige assets with a cash-flow
   floor, not cash-flow machines.
2. **On-pitch performance predicts revenue growth (H4 — supported).** A one
   point-per-game improvement predicts ~20pp higher next-season revenue growth
   (β = 0.204, p = 0.048, club and season fixed effects, SEs clustered by
   club), confirmed out-of-sample (LASSO OOS R² = 0.54 vs −0.01 naive).
3. **Defensible nulls where nulls belong.** No multi-club-ownership premium
   (H2), no evidence markets misprice the *convexity* of Champions-League
   qualification (H3) — although the linear qualification effect is clearly
   priced (+14.9% squad value, p = 0.0015). No promotion overvaluation among
   surviving promoted clubs (H1; survivorship caveat documented).
4. **A club portfolio clears the 15% IRR objective on paper.** A long-only
   max-Sharpe portfolio built from valuation gaps has 23.5% expected return
   (Sharpe 1.78); a seven-year mark-to-model backtest of the same value
   signal returns 14.4%/yr (Sharpe 0.77) vs 12.5% equal-weight — with the
   short leg failing, an honest asymmetry we report rather than hide.

## 1. Data (Phase 1)

- **Results/standings**: football-data.co.uk match CSVs; league tables
  recomputed from results incl. the 2023-24 PL point deductions. Champions
  reproduce reality 18/18 seasons; an independent R pipeline reproduces all
  398 club-season standings exactly.
- **Market values, transfers, minutes**: Transfermarkt (throttled ≥3.5s,
  cached, robots.txt-checked). Player minutes reconcile with matches played
  within 1% for all 358 club-seasons.
- **Financial statements**: every Companies House filing in-window is a
  scanned PDF; we built a macOS-Vision OCR pipeline with prior-year
  comparative anchoring, unit/scale gates, and cited-benchmark validation
  (Everton/Liverpool/Brighton/Chelsea all within 2%). PL revenue coverage:
  **180/180** club-seasons; wage-to-revenue 155/180 with a documented
  measure caveat.
- **Source integrity**: FBref (blocks all non-browser clients) and Understat
  (robots.txt disallows crawling) were dropped/rejected rather than
  circumvented; French club financials (DNCG) remain the open item.
- **Ownership registry**: 17 MCO entries verified with citations; the ≥25%
  threshold excluded PSG (QSI's Braga stake is 21.67%) — the definition
  applied against convenience.

## 2. Valuation engine (Phase 2)

- **DCF over an empirical Markov chain.** League-position states transition
  per frequencies estimated from the panel (top-4 persistence 0.64;
  bottom-band relegation risk 0.24; Championship absorption 0.94). State
  revenue multipliers are calibrated from within-club data — the panel says
  UCL participation lifts revenue 17% at the median, overriding our 30%
  prior. WACC = gilt + 5% ERP + 3% illiquidity ± tier adjustments.
  Loss-makers keep negative DCFs (Everton: −£73m against a ~£400m sale —
  the trophy premium in one line).
- **Comparable transactions**: nine verified deals (0.85× distressed Villa
  to 5.2× trophy Chelsea; core median 2.19× EV/Revenue), applied by status
  band with uncertainty exposed.
- **UCL real option**: additive binomial lattice on position volatility.
  Convexity premium is 1-4% of value for boundary clubs — quantifying *why*
  H3's null is the expected outcome.

## 3. Hypothesis tests (Phase 3)

Panel specs: club FE + season effects, clustered SEs, lagged predictors,
ex-COI robustness rows. ML: LASSO/RF/GBM + ensemble, train ≤2020-21 / test
2021-22+, train-only winsorisation. Verdicts in the executive summary;
notable extras: squad *age* adds nothing to revenue-growth prediction once
performance is controlled, and squad-value growth is unpredictable
out-of-sample with a documented post-2021 structural break in transfer
spending. R reproduces the headline coefficient exactly.

## 4. Portfolio construction (Phase 4)

Expected returns from valuation-gap convergence (5-year assumption) plus
league drift; Ledoit-Wolf covariance (shrinkage 0.526 — a 9-observation
panel demands it); SLSQP optimisation (long-only, 20% cap; 130/30 variant);
parametric + historical VaR/CVaR; 5M-path Monte Carlo in C++ (4.8× NumPy,
quantiles identical; the benchmark write-up documents that naive C++ *loses*
to vectorised NumPy and why). Median 5-year outcome 2.83× (23.1% IRR),
P(loss) ≈ 0 under model assumptions — which inherit every caveat above.

## 5. BlueCo case study (Phase 5)

**COI restated: the author is a Chelsea supporter.** Sixteen
Chelsea↔Strasbourg player movements since June 2023, overwhelmingly teenage
loanees. Strasbourg's squad value nearly quadrupled post-acquisition
(€123.6m → €466.1m, 2022→2025) while the matched non-MCO control (Nantes)
stayed flat; average squad age fell 26.0 → 22.0. This *explains* H2's null:
the synergy is real but recent, and it accrues first as player-asset value,
which panel-window club valuations don't yet capture.

## 6. Limitations

Mark-to-model returns (clubs do not trade annually); Transfermarkt values
are crowd estimates; OCR-extracted financials carry residual measurement
error (gated, flagged, never guessed); French revenue pending DNCG; one
treated club in the case study; H1 survivorship; short panel (9 seasons)
behind a shrunk covariance. Every limitation appears where the affected
number is reported, not only here.

## Appendices

Phase artifacts: `outputs/phase1..5/*`, archives `outputs/phaseN.tar.gz`,
dashboard `streamlit run src/phase6_dashboard/app.py`, validation reports
under `data/validation/` and `r/`.

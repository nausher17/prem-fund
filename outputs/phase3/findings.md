# Phase 3 — Hypothesis test findings (2026-07-19)

All specifications: club fixed effects, season effects, SEs clustered by club,
lagged predictors where causality requires. Full tables:
`econometrics_results.csv`, `ml_scores.csv`, `ml_feature_importances.csv`.
R cross-validation reproduces the H4 headline coefficient exactly
(`r/phase3_validation.R`).

## H1 — Promotion overvaluation: NOT SUPPORTED (with a caveat that matters)

Newly promoted clubs' squad values *rise* in their following season
(β = +0.178 on forward value growth, p = 0.070) rather than correcting
downward. **Survivorship caveat**: the outcome requires the club to still be
in a panel league at t+1, so relegated promoted clubs — precisely the cases
H1 predicts — drop out. The estimate is therefore an upper bound, and the
honest verdict is "no evidence of overvaluation among surviving promoted
clubs", not "promoted clubs are undervalued". A Championship-value extension
would tighten this.

## H2 — MCO premium: NULL (as expected)

β = +0.089 log-points on squad value (p = 0.18); integrated-scope-only
β = +0.057 (p = 0.43). Within-club identification means always-MCO clubs
(Man City, Watford) contribute nothing; the estimate rests on ~15 clubs that
*switched* status in-window, most post-2021 — low power, short post-periods.
The Phase 5 case study asks whether synergies are real but too slow/small to
capitalise into market values this quickly. Ex-COI robustness: unchanged.

## H3 — UCL optionality mispricing: LINEAR EFFECT PRICED, NO CONVEXITY — NULL

The market prices qualification itself clearly: prior-season UCL
participation carries +14.9% squad value (p = 0.0015). But the *option*
component — the interaction of near-the-money position volatility with
qualification — is a precise zero (β = −0.016, p = 0.60). Phase 2's lattice
work explains why this null is expected: the convexity premium is only
~1–4% of club value for boundary clubs, well inside valuation noise.
Defensible null, mechanism quantified.

## H4 — Performance predicts revenue growth: SUPPORTED (strongest result)

Lagged points-per-game predicts revenue growth (β = 0.204, p = 0.048 with
value control; β = 0.165, p = 0.065 in the parsimonious spec). Minutes-
weighted age adds nothing conditional on performance (p = 0.74) — the "squad
demographics" half of H4 does not survive controls; the performance half
does. Out-of-sample confirmation: revenue growth is predictable (LASSO
OOS R² = 0.54 vs −0.01 naive), driven by promotion status and macro
conditions.

Note vs the prior build's p ≈ 0.002: our sample is PL-only (n = 136;
French DNCG revenue pending) and uses two-way fixed effects — the smaller,
stricter design weakens significance; the direction and economic size hold.

## ML suite (temporal split: train ≤2020-21, test 2021-22+)

| target | best model | OOS R² | verdict |
|---|---|---|---|
| revenue growth | LASSO | **0.54** | predictable — H4's OOS mirror |
| squad value growth | (tree models) | ≤ −0.27 | unpredictable; LASSO fails
catastrophically (−65) because post-2021 transfer spending sits far outside
the training range — a structural break (Chelsea/Newcastle spending regime),
reported as a finding |
| wage-to-revenue | RF | 0.07 | essentially unpredictable |

Growth targets winsorised at train p1/p99 (documented; no leakage).

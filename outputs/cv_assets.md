# CV assets

## One-line bullet (Quantitative Projects section)

> Built an end-to-end quantitative fund model treating 40+ European football
> clubs as alternative assets — Python ETL over 350+ OCR-parsed statutory
> filings, scenario-weighted DCF on an empirical Markov chain, fixed-effects
> panel econometrics (cross-validated in R), LASSO/ensemble ML with temporal
> splits, and Markowitz optimisation with a multithreaded C++ Monte Carlo
> engine (4.8× NumPy) — clearing a 15% IRR objective with a 23.5% expected-
> return max-Sharpe portfolio.

## Interview talking points

1. **The trophy-asset premium (headline finding).** "My DCF explains about a
   third of Forbes' valuations — median gap 2.98×. Instead of tuning the
   model to match, I quantified the gap and framed it as the price of
   scarcity and prestige. Everton was my favourite case: negative operating
   DCF, sold for ~£400m."
2. **Honest nulls beat inflated positives.** "Two of my four hypotheses came
   back null and I shipped them as findings: no MCO premium — but my
   Chelsea-Strasbourg case study shows *why* (synergies accrue as player
   asset value, after my panel window; +277% squad value vs a matched
   control) — and no mispricing of UCL-qualification convexity, where my own
   options model shows the premium is only 1-4% of club value: too small to
   detect, and I can show that arithmetic."
3. **The C++ benchmark that refused to flatter me.** "Naive C++ *lost* to
   vectorised NumPy — the bottleneck was RNG throughput, not loops. I
   documented that, swapped in xoshiro256++/Box-Muller for parity, and won
   4.8× only through multithreaded path-splitting. The honest benchmark is a
   better story than a fake speedup."
4. **Data engineering under hostile conditions.** "Every UK statutory filing
   I needed was a scanned PDF. I built an OCR extraction pipeline whose
   integrity anchor is each filing's audited prior-year column — 180/180
   club-season revenue coverage, all within 2% of cited public figures —
   and when FBref blocked scrapers and Understat's robots.txt said no, I
   redesigned around them instead of spoofing."

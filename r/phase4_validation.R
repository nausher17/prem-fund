#!/usr/bin/env Rscript
# Phase 4 cross-validation (base R): portfolio arithmetic re-derived from
# the written artifacts.

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
root <- normalizePath(file.path(dirname(script_path), ".."))
out <- file.path(root, "outputs", "phase4")

fail <- function(msg) { cat("FAIL:", msg, "\n"); quit(status = 1) }
note <- function(msg) cat("  [OK]", msg, "\n")

w <- read.csv(file.path(out, "weights.csv"), row.names = 1)
er <- read.csv(file.path(out, "expected_returns.csv"), row.names = 1)
cov <- as.matrix(read.csv(file.path(out, "covariance.csv")))
stats <- read.csv(file.path(out, "portfolio_stats.csv"), row.names = 1)

if (abs(sum(w$max_sharpe) - 1) > 1e-6) fail("max_sharpe weights do not sum to 1")
if (any(w$max_sharpe < -1e-9) || any(w$max_sharpe > 0.2 + 1e-6))
  fail("max_sharpe violates long-only 20% cap")
note("weight constraints verified")

for (p in c("max_sharpe", "min_var")) {
  ww <- w[[p]]
  ret <- sum(ww * er$exp_return)
  vol <- sqrt(drop(t(ww) %*% cov %*% ww))
  if (abs(ret - stats[p, "exp_return"]) > 1e-3) fail(paste(p, "return mismatch"))
  if (abs(vol - stats[p, "vol"]) > 1e-3) fail(paste(p, "vol mismatch"))
  note(sprintf("%s: return %.4f vol %.4f reproduce", p, ret, vol))
}

bt <- read.csv(file.path(out, "backtest_annual.csv"))
sm <- read.csv(file.path(out, "backtest_summary.csv"))
geo <- exp(mean(bt$long_only)) - 1
if (abs(geo - sm$ann_return[sm$strategy == "long_only"]) > 1e-3)
  fail("backtest long_only annual return mismatch")
note(sprintf("backtest long-only annualised return reproduces (%.4f)", geo))

cat("Phase 4 R validation passed.\n")

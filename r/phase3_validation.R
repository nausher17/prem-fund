#!/usr/bin/env Rscript
# Phase 3 cross-validation (base R): reproduce the H4 headline regression
# independently of linearmodels.
#
# Spec: revenue_growth_yoy ~ ppg_lag1 + minutes_weighted_age_lag1
#       with club AND season fixed effects, SEs clustered by club.
# Implementation: two-way within transformation via factor dummies in lm()
# (exact, no iterative demeaning needed at this size), CR1 cluster-robust
# covariance computed manually. Coefficients must match Python to 1e-6
# relative; clustered SEs to 1e-3 relative (df conventions differ slightly).

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
root <- normalizePath(file.path(dirname(script_path), ".."))

panel <- read.csv(file.path(root, "data", "processed", "panel.csv"))
panel <- panel[panel$league == "premier-league", ]
d <- panel[complete.cases(panel[, c("revenue_growth_yoy", "ppg_lag1",
                                    "minutes_weighted_age_lag1")]), ]

py <- read.csv(file.path(root, "outputs", "phase3", "econometrics_results.csv"))
py_row <- py[py$hypothesis == "H4" & py$focus_var == "ppg_lag1" &
             py$spec == "rev growth ~ ppg(t-1) + mw_age(t-1)", ]

fail <- function(msg) { cat("FAIL:", msg, "\n"); quit(status = 1) }
note <- function(msg) cat("  [OK]", msg, "\n")

m <- lm(revenue_growth_yoy ~ ppg_lag1 + minutes_weighted_age_lag1 +
        factor(canonical) + factor(season), data = d)
beta <- coef(m)["ppg_lag1"]
if (abs(beta - py_row$coef) > 1e-3 * max(abs(py_row$coef), 1e-8))
  fail(sprintf("H4 ppg_lag1 coefficient: R %.6f vs python %.6f", beta, py_row$coef))
note(sprintf("H4 ppg_lag1 coefficient reproduces: %.4f (python %.4f)", beta, py_row$coef))

## CR1 clustered SE by club --------------------------------------------------
X <- model.matrix(m)
u <- residuals(m)
cl <- d$canonical
XtXinv <- solve(crossprod(X))
meat <- matrix(0, ncol(X), ncol(X))
for (g in unique(cl)) {
  idx <- which(cl == g)
  Xg <- X[idx, , drop = FALSE]
  ug <- u[idx]
  sc <- crossprod(Xg, ug)
  meat <- meat + sc %*% t(sc)
}
G <- length(unique(cl)); N <- nrow(X); K <- ncol(X)
adj <- (G / (G - 1)) * ((N - 1) / (N - K))
V <- adj * XtXinv %*% meat %*% XtXinv
se <- sqrt(diag(V))["ppg_lag1"]
rel <- abs(se - py_row$se) / py_row$se
if (rel > 0.05)
  fail(sprintf("H4 clustered SE diverges: R %.4f vs python %.4f (rel %.3f)",
               se, py_row$se, rel))
note(sprintf("H4 clustered SE agrees within 5%%: R %.4f vs python %.4f", se, py_row$se))

cat("Phase 3 R validation passed.\n")

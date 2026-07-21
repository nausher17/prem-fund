#!/usr/bin/env Rscript
# Phase 2 cross-validation (base R): independently re-derive headline
# valuation quantities from the written outputs and flag divergence.
#
# Checks:
#   1. Transition matrix: rows sum to 1; recompute the 5-year UCL
#      probability for each start state by matrix power and compare with
#      p_ucl_5y in valuations.csv.
#   2. DCF re-computation for one club (Tottenham Hotspur) from first
#      principles using the same inputs written to valuations.csv +
#      transition_matrix.csv + assumptions embedded in the row (wacc, growth,
#      margin), tolerance 0.5%.
#   3. Comps: value == multiple * revenue for every club.

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
root <- normalizePath(file.path(dirname(script_path), ".."))
outdir <- file.path(root, "outputs", "phase2")

fail <- function(msg) { cat("FAIL:", msg, "\n"); quit(status = 1) }
note <- function(msg) cat("  [OK]", msg, "\n")

P <- as.matrix(read.csv(file.path(outdir, "transition_matrix.csv"), row.names = 1))
v <- read.csv(file.path(outdir, "valuations.csv"))

## 1. stochastic matrix + 5y UCL probabilities ------------------------------
if (max(abs(rowSums(P) - 1)) > 1e-6) fail("transition matrix rows do not sum to 1")
note("transition matrix is stochastic")

P5 <- P %*% P %*% P %*% P %*% P
states <- rownames(P)
for (i in seq_len(nrow(v))) {
  s <- v$state[i]
  s_chain <- ifelse(s == "REL", "CHAMP", s)
  p_ucl <- P5[match(s_chain, states), match("UCL", states)]
  if (abs(p_ucl - v$p_ucl_5y[i]) > 0.002)
    fail(sprintf("p_ucl_5y mismatch for %s: R %.4f vs py %.4f",
                 v$club[i], p_ucl, v$p_ucl_5y[i]))
}
note("5-year UCL probabilities reproduce for all clubs")

## 2. DCF re-computation: Tottenham -----------------------------------------
mult <- c(UCL = 1.17, EUR = 1.008, MID = 1.0, LOW = 0.959, CHAMP = 0.55)
# state multipliers as calibrated (printed by dcf.py; UCL/EUR/LOW re-derived
# in Python from the panel — here we verify the DCF arithmetic given them)
row <- v[v$club == "Tottenham Hotspur", ]
R0 <- row$revenue_fy24_gbp_m * 1e6
margin <- row$ebitda_margin; g <- row$growth; wacc <- row$wacc
H <- 10; gT <- 0.02; conv <- 0.60
s_idx <- match(row$state, states)
m_vec <- mult[states]; m_vec[is.na(m_vec)] <- 1
Pk <- diag(length(states)); pv <- 0; fcfH <- 0
for (t in 1:H) {
  Pk <- Pk %*% P
  exp_mult <- sum(Pk[s_idx, ] * m_vec)
  Rt <- R0 / mult[[row$state]] * exp_mult * (1 + g)^t
  fcf <- Rt * margin * conv
  pv <- pv + fcf / (1 + wacc)^t
  if (t == H) fcfH <- fcf
}
term <- if (fcfH > 0) fcfH * (1 + gT) / (wacc - gT) / (1 + wacc)^H else 0
dcf_r <- (pv + term) / 1e6
if (abs(dcf_r - row$dcf_value_gbp_m) / row$dcf_value_gbp_m > 0.005)
  fail(sprintf("Tottenham DCF: R %.1f vs py %.1f", dcf_r, row$dcf_value_gbp_m))
note(sprintf("Tottenham DCF reproduces in R (%.1f vs %.1f GBP m)",
             dcf_r, row$dcf_value_gbp_m))

## 3. comps arithmetic --------------------------------------------------------
err <- abs(v$comps_value_gbp_m - v$comps_multiple * v$revenue_fy24_gbp_m)
if (max(err) > 1) fail("comps value != multiple * revenue somewhere")
note("comps arithmetic verified for all clubs")

cat("Phase 2 R validation passed.\n")

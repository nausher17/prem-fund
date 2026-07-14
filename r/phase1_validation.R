#!/usr/bin/env Rscript
# Phase 1 cross-validation in R (base R only, no package dependencies).
#
# Independently recomputes headline numbers from the least-processed inputs
# available (match-level results) and compares them against the Python
# pipeline's outputs. Exits non-zero on any divergence.
#
# Checks:
#   1. Points / W-D-L / GF-GA per club-season recomputed from matches.csv
#      must equal league_tables.csv (pre-deduction points_raw).
#   2. Champions per league-season must match the public record.
#   3. panel.csv internal consistency: ppg == points/mp, plausible ranges
#      for minutes-weighted age and squad values.
#
# Output: data/validation/phase1_r_checks.csv

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
root <- normalizePath(file.path(dirname(script_path), ".."))
processed <- file.path(root, "data", "processed")
outdir <- file.path(root, "data", "validation")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

fail <- function(msg) { cat("FAIL:", msg, "\n"); quit(status = 1) }
results <- data.frame(check = character(), status = character())
note <- function(check, status) {
  results <<- rbind(results, data.frame(check = check, status = status))
  cat(sprintf("  [%s] %s\n", status, check))
}

matches <- read.csv(file.path(processed, "matches.csv"))
tables  <- read.csv(file.path(processed, "league_tables.csv"))

## 1. Recompute standings from match results ---------------------------------
home <- data.frame(league = matches$league, season = matches$season,
                   team = matches$HomeTeam,
                   pts = ifelse(matches$FTHG > matches$FTAG, 3,
                          ifelse(matches$FTHG == matches$FTAG, 1, 0)),
                   w = as.integer(matches$FTHG > matches$FTAG),
                   d = as.integer(matches$FTHG == matches$FTAG),
                   gf = matches$FTHG, ga = matches$FTAG)
away <- data.frame(league = matches$league, season = matches$season,
                   team = matches$AwayTeam,
                   pts = ifelse(matches$FTAG > matches$FTHG, 3,
                          ifelse(matches$FTAG == matches$FTHG, 1, 0)),
                   w = as.integer(matches$FTAG > matches$FTHG),
                   d = as.integer(matches$FTAG == matches$FTHG),
                   gf = matches$FTAG, ga = matches$FTHG)
long <- rbind(home, away)
recomp <- aggregate(cbind(pts, w, d, gf, ga) ~ league + season + team,
                    data = long, FUN = sum)

cmp <- merge(recomp, tables,
             by.x = c("league", "season", "team"),
             by.y = c("league", "season", "team"))
if (nrow(cmp) != nrow(tables)) fail("row mismatch merging recomputed standings")
bad <- cmp[cmp$pts != cmp$points_raw | cmp$w.x != cmp$w.y |
           cmp$gf.x != cmp$gf.y | cmp$ga.x != cmp$ga.y, ]
if (nrow(bad) > 0) { print(bad); fail("recomputed standings diverge from Python") }
note(sprintf("standings recomputation (%d club-seasons)", nrow(cmp)), "OK")

## 2. Champions vs public record ----------------------------------------------
oracle <- rbind(
  data.frame(league = "premier-league",
             season = c("2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
                        "2020-2021","2021-2022","2022-2023","2023-2024"),
             champion = c("Leicester","Chelsea","Man City","Man City","Liverpool",
                          "Man City","Man City","Man City","Man City")),
  data.frame(league = "ligue-1",
             season = c("2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
                        "2020-2021","2021-2022","2022-2023","2023-2024"),
             champion = c("Paris SG","Monaco","Paris SG","Paris SG","Paris SG",
                          "Lille","Paris SG","Paris SG","Paris SG")))
champs <- tables[tables$position == 1, c("league", "season", "team")]
chk <- merge(oracle, champs, by = c("league", "season"))
wrong <- chk[chk$champion != chk$team, ]
if (nrow(wrong) > 0) { print(wrong); fail("champions diverge from public record") }
note("champions vs public record (18 league-seasons)", "OK")

## 3. Panel internal consistency ----------------------------------------------
panel_path <- file.path(processed, "panel.csv")
if (file.exists(panel_path)) {
  panel <- read.csv(panel_path)
  if (max(abs(panel$ppg - panel$points / panel$mp)) > 1e-9)
    fail("panel ppg != points/mp")
  note("panel ppg consistency", "OK")
  mwa <- panel$minutes_weighted_age[!is.na(panel$minutes_weighted_age)]
  if (length(mwa) > 0 && (min(mwa) < 18 || max(mwa) > 35))
    fail("minutes-weighted age outside [18, 35]")
  note("minutes-weighted age plausible range", "OK")
  sv <- panel$total_market_value_eur_m[!is.na(panel$total_market_value_eur_m)]
  if (min(sv) < 10 || max(sv) > 2500) fail("squad values outside [10, 2500] EUR m")
  note("squad value plausible range", "OK")
} else {
  note("panel.csv not built yet - panel checks skipped", "SKIP")
}

write.csv(results, file.path(outdir, "phase1_r_checks.csv"), row.names = FALSE)
cat("R validation passed. Results ->", file.path(outdir, "phase1_r_checks.csv"), "\n")

#!/usr/bin/env bash
# Build and verify outputs/phase1.tar.gz (run from repo root).
set -euo pipefail
cd "$(dirname "$0")/.."
tar -czf outputs/phase1.tar.gz \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.DS_Store' \
  README.md requirements.txt \
  src/phase1_data tests r \
  data/processed/league_tables.csv data/processed/matches.csv \
  data/processed/tm_club_seasons.csv data/processed/tm_transfers.csv \
  data/processed/tm_player_seasons.csv data/processed/macro_monthly.csv \
  data/processed/panel.csv data/processed/panel.parquet \
  data/validation
echo "--- verify ---"
tar -tzf outputs/phase1.tar.gz | head -20
echo "..."
tar -tzf outputs/phase1.tar.gz | wc -l

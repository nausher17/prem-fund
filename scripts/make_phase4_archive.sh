#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tar -czf outputs/phase4.tar.gz --exclude='__pycache__' \
  src/phase4_portfolio cpp/mc_engine.cpp r/phase4_validation.R \
  tests/test_phase4.py outputs/phase4
tar -tzf outputs/phase4.tar.gz | wc -l

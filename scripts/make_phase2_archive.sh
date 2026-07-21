#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tar -czf outputs/phase2.tar.gz --exclude='__pycache__' \
  src/phase2_valuation r/phase2_validation.R tests/test_phase2.py \
  outputs/phase2
tar -tzf outputs/phase2.tar.gz | wc -l

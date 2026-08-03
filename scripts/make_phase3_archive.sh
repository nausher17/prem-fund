#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tar -czf outputs/phase3.tar.gz --exclude='__pycache__' \
  src/phase3_hypotheses r/phase3_validation.R tests/test_phase3.py outputs/phase3
tar -tzf outputs/phase3.tar.gz | wc -l

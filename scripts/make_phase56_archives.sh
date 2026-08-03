#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tar -czf outputs/phase5.tar.gz --exclude='__pycache__' src/phase5_blueco outputs/phase5
tar -czf outputs/phase6.tar.gz --exclude='__pycache__' src/phase6_dashboard outputs/final_report.md outputs/cv_assets.md
tar -tzf outputs/phase5.tar.gz | wc -l; tar -tzf outputs/phase6.tar.gz | wc -l

#!/usr/bin/env bash
# Rebuilds PachoAnalytics.csv from the provided match data and checks it is byte-identical.
# Usage: ./reproduce.sh            (expects data/tactical/BAR-ATH_tactical_atd.{txt,xml})
#        MTAH_DATA_DIR=/path ./reproduce.sh   (folder that contains tactical/)
set -euo pipefail
cd "$(dirname "$0")"
DATA="${MTAH_DATA_DIR:-data}"
python3 pipeline/predict.py \
  --atd-raw "$DATA/tactical/BAR-ATH_tactical_atd.txt" \
  --atd-meta "$DATA/tactical/BAR-ATH_tactical_atd.xml" \
  --out PachoAnalytics.csv
if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
want=$(grep ' PachoAnalytics.csv$' expected/SHA256SUMS | cut -d' ' -f1)
got=$($SHA PachoAnalytics.csv | cut -d' ' -f1)
if [ "$want" = "$got" ]; then echo "IDENTICAL: PachoAnalytics.csv matches the submitted file ($got)"
else echo "DIFFERENT: got $got, expected $want"; exit 1; fi

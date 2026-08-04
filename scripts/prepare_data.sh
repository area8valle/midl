#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-synthetic_cohort.npz}"
SAMPLES="${SAMPLES:-512}"

python -m midl.apps.simulate \
    --gin_file configs/experiment/main.gin \
    --samples "${SAMPLES}" \
    --output "${OUTPUT}"

echo "wrote ${OUTPUT}"
echo "OAI and MOST are access-restricted; obtain them from the URLs in the README Datasets panel"

#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
CHECKPOINT="${2:-runs/midl/best.pt}"

python -m midl.apps.score \
    --gin_file "configs/experiment/${EXPERIMENT}.gin" \
    --checkpoint "${CHECKPOINT}" "${@:3}"

#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
NPROC="${NPROC:-1}"

torchrun --standalone --nproc_per_node="${NPROC}" \
    -m midl.apps.fit --gin_file "configs/experiment/${EXPERIMENT}.gin" "${@:2}"

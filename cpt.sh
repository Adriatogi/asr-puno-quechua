#!/usr/bin/env bash
#docker compose run asr-puno-quechua
set -euo pipefail
mkdir -p logs
bash training/scripts/run_cpt.sh 4 "qxp_scripted,qxp_spontaneous" 2>&1 | tee logs/cpt.log

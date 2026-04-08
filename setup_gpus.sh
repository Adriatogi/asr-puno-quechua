#!/usr/bin/env bash
# Run on the HOST before starting the Docker container.
# Detects which container GPU indices correspond to your SLURM-allocated GPUs
# and updates CUDA_VISIBLE_DEVICES in .env.
#
# Usage: bash setup_gpus.sh

set -euo pipefail

# Fixed UUID→index map for all 8 GPUs on mkt1 (update if you move to a different machine).
declare -A CONTAINER_MAP=(
    ["GPU-03941855-b2a5-7aec-4cd9-44d931cf525d"]=0
    ["GPU-26bbdd7e-9587-9eda-aae5-c5745d30696e"]=1
    ["GPU-d34f0bb3-4f3d-d5f1-b9bf-54358d13e62b"]=2
    ["GPU-34c7e317-cbae-952e-41b2-f383e5b42c85"]=3
    ["GPU-9dce805e-f0d9-abca-f8cd-065b8151f0e9"]=4
    ["GPU-6cc4d59c-f23e-ccba-70a4-cfe7aae1b698"]=5
    ["GPU-c003cd07-8524-616f-ba45-e29e951d47fb"]=6
    ["GPU-a6fb33b5-e4f7-0da4-936b-51e4ae1f61ff"]=7
)

INDICES=""
while IFS=', ' read -r _ uuid; do
    uuid="${uuid// /}"
    if [[ -n "${CONTAINER_MAP[$uuid]+_}" ]]; then
        INDICES="${INDICES:+$INDICES,}${CONTAINER_MAP[$uuid]}"
    else
        echo "WARNING: unknown GPU UUID $uuid — is this a different machine?"
    fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

if [[ -z "$INDICES" ]]; then
    echo "ERROR: no matching GPUs found. Re-run the UUID detection step."
    exit 1
fi

echo "Allocated GPUs → container indices: $INDICES"

if grep -q "^DOCKER_CUDA_VISIBLE_DEVICES=" .env; then
    sed -i "s/^DOCKER_CUDA_VISIBLE_DEVICES=.*/DOCKER_CUDA_VISIBLE_DEVICES=$INDICES/" .env
else
    echo "DOCKER_CUDA_VISIBLE_DEVICES=$INDICES" >> .env
fi

echo "Updated .env — now start the container with: docker compose run asr-puno-quechua"

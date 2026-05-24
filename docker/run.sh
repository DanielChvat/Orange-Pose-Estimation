#!/usr/bin/env bash
# Convenience launcher.
#
# Default (no args) -> serves the web UI on http://localhost:8765
# Examples:
#   docker/run.sh                              # web UI
#   docker/run.sh bash                         # interactive shell
#   docker/run.sh bash run_pipeline.sh         # run pipeline end-to-end (headless)
#   docker/run.sh python scripts/get_masks.py --help
#
# HuggingFace auth (needed for SAM3 weights):
#   1. One-shot, this run only:
#        HF_TOKEN=hf_xxxxxxxx docker/run.sh
#   2. Persistent (recommended):
#        huggingface-cli login       # once on the host
#        docker/run.sh               # token is mounted in
#
# Mounts:
#   $(pwd)                -> /workspace              (project, read/write)
#   ~/.cache/pip          -> .cache/pip              (faster reinstalls)
#   ~/.cache/huggingface  -> .cache/huggingface      (HF token + model weights)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-orange-pose:latest}"

# Forward the web UI port (8765) by default.
PORTS=(-p "${WEB_PORT:-8765}:8765")

GPU_FLAGS=(--gpus all)

mkdir -p "$HOME/.cache/pip"
mkdir -p "$HOME/.cache/huggingface"

# Pass HF_TOKEN through if set in the host env (overrides any cached login).
HF_FLAGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_FLAGS+=(-e "HF_TOKEN=${HF_TOKEN}")
    echo "[run.sh] Using HF_TOKEN from environment."
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    echo "[run.sh] Using cached huggingface-cli login from ~/.cache/huggingface/token."
else
    echo "[run.sh] WARNING: no HuggingFace credentials found. SAM3 model download will fail."
    echo "[run.sh]   Fix: HF_TOKEN=hf_xxx docker/run.sh   OR   huggingface-cli login (once)"
fi

if [[ $# -eq 0 ]]; then
    echo "[run.sh] Web UI: http://localhost:${WEB_PORT:-8765}"
fi

exec docker run \
    --rm -it \
    "${GPU_FLAGS[@]}" \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/.container-home \
    -e USER="${USER:-orange}" \
    -e NUMBA_CACHE_DIR=/tmp/numba_cache \
    -e HF_HOME=/workspace/.container-home/.cache/huggingface \
    -e TRANSFORMERS_CACHE=/workspace/.container-home/.cache/huggingface \
    "${HF_FLAGS[@]}" \
    -v "$REPO_ROOT":/workspace \
    -v "$HOME/.cache/pip":/workspace/.container-home/.cache/pip \
    -v "$HOME/.cache/huggingface":/workspace/.container-home/.cache/huggingface \
    -w /workspace \
    "${PORTS[@]}" \
    "$IMAGE_TAG" \
    "$@"

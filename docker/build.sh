#!/usr/bin/env bash
# Build the orange-pose Docker image. Run from project root or anywhere — the
# script cd's to the project root so the build context is correct.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="${IMAGE_TAG:-orange-pose:latest}"

echo "[INFO] Building $IMAGE_TAG"
echo "[INFO] Project root: $REPO_ROOT"
echo "[INFO] Initial heavy layers (CUDA base + apt + torch + extensions) will"
echo "[INFO] take 20-40 minutes the first time. Subsequent builds reuse cache."

# Use BuildKit if available (better cache, parallel layer fetch). Falls back
# to the legacy builder if `docker buildx` isn't installed.
if docker buildx version >/dev/null 2>&1; then
    docker buildx build \
        --load \
        -t "$IMAGE_TAG" \
        -f docker/Dockerfile \
        --progress=plain \
        "$@" \
        .
else
    echo "[INFO] docker buildx not found, falling back to legacy builder."
    echo "[INFO] For better cache + parallelism, install docker-buildx (Arch: sudo pacman -S docker-buildx)."
    DOCKER_BUILDKIT=1 docker build \
        -t "$IMAGE_TAG" \
        -f docker/Dockerfile \
        "$@" \
        .
fi

echo "[DONE] Image $IMAGE_TAG built."
echo "[INFO] Try: docker run --gpus all --rm -it -v \"\$(pwd)\":/workspace -w /workspace $IMAGE_TAG bash"

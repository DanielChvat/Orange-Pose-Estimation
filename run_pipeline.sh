#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] Pipeline failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

CONFIG="${CONFIG:-configs/gaussian_pipeline.json}"
PYTHON="${PYTHON:-python}"

config_value() {
  "$PYTHON" - "$CONFIG" "$1" "$2" <<'PY'
import json
import sys

path, section, key = sys.argv[1:4]
with open(path) as f:
    data = json.load(f)
value = data.get(section, {}).get(key, "")
print("" if value is None else value)
PY
}

VIDEO="${VIDEO:-$(config_value pipeline video)}"
PROMPT="${PROMPT:-$(config_value pipeline prompt)}"
ITERATIONS="${ITERATIONS:-$(config_value pipeline iterations)}"
SAM_FPS="${SAM_FPS:-$(config_value masks fps)}"
COLMAP_RECONSTRUCTION_FPS="${COLMAP_RECONSTRUCTION_FPS:-$(config_value colmap reconstruction_fps)}"
SAM_FULL_IMAGE_PASS="${SAM_FULL_IMAGE_PASS:-$(config_value masks full_image_pass)}"
SAM_DETAIL_PASS="${SAM_DETAIL_PASS:-$(config_value masks detail_pass)}"
SAM_TILE_CONTEXT_MARGIN="${SAM_TILE_CONTEXT_MARGIN:-$(config_value masks tile_context_margin)}"
GAUSSIANS="out/3dgs/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

if [[ -z "$VIDEO" ]]; then
  echo "[ERROR] No video selected. Upload one in the web UI or run: VIDEO=/path/to/video.mp4 ./run_pipeline.sh"
  exit 2
fi

if [[ -z "$PROMPT" ]]; then
  echo "[ERROR] No prompt selected. Set PROMPT='object description' or use the web UI prompt box."
  exit 2
fi

echo "[INFO] Config: $CONFIG"
echo "[INFO] Video: $VIDEO"
echo "[INFO] Prompt: $PROMPT"
echo "[INFO] Iterations: $ITERATIONS"
echo "[INFO] SAM FPS: $SAM_FPS"
echo "[INFO] COLMAP reconstruction FPS: $COLMAP_RECONSTRUCTION_FPS"
echo "[INFO] SAM whole-frame pass: $SAM_FULL_IMAGE_PASS"
echo "[INFO] SAM detail pass: $SAM_DETAIL_PASS"
echo "[INFO] SAM tile context margin: $SAM_TILE_CONTEXT_MARGIN"

echo "[0/7] Cleaning stale outputs"
"$PYTHON" scripts/clean_outputs.py --apply
rm -rf \
  out/frames \
  out/masks \
  out/colmap \
  out/3dgs \
  out/gaussian_mask_votes \
  out/gaussian_object_tracks \
  out/gaussian_ownership \
  out/object_spheres \
  vis/gaussian_splat_overlay
mkdir -p out vis

echo "[1/7] SAM masks"
"$PYTHON" scripts/get_masks.py \
  --config "$CONFIG" \
  --video "$VIDEO" \
  --prompt "$PROMPT" \
  --fps "$SAM_FPS" \
  --full-image-pass "$SAM_FULL_IMAGE_PASS" \
  --detail-pass "$SAM_DETAIL_PASS" \
  --tile-context-margin "$SAM_TILE_CONTEXT_MARGIN"

ACTIVE_MASK_DIR="out/masks"
echo "[2/7] COLMAP dataset for 3DGS"
"$PYTHON" scripts/run_colmap.py \
  --config "$CONFIG" \
  --video "$VIDEO" \
  --reconstruction-fps "$COLMAP_RECONSTRUCTION_FPS" \
  --overwrite

echo "[INFO] Downstream mask dir: $ACTIVE_MASK_DIR"

echo "[3/7] 3DGS"
"$PYTHON" scripts/run_3dgs.py \
  --config "$CONFIG" \
  --iterations "$ITERATIONS"

echo "[4/7] Voting SAM evidence onto Gaussians"
"$PYTHON" scripts/vote_gaussian_masks.py --config "$CONFIG" --gaussians "$GAUSSIANS" --mask-dir "$ACTIVE_MASK_DIR" --overwrite

echo "[5/7] Assigning IDs to foreground splat blobs"
"$PYTHON" scripts/track_gaussian_objects.py --config "$CONFIG" --gaussians "$GAUSSIANS" --mask-dir "$ACTIVE_MASK_DIR" --overwrite

echo "[6/7] Fitting OBB + inscribed sphere per object"
"$PYTHON" scripts/fit_object_spheres.py --config "$CONFIG" --overwrite

echo "[7/7] Building browser viewer"
"$PYTHON" util/create_gaussian_overlay_viewer.py --config "$CONFIG" --overwrite

echo "[DONE] Object tracks: out/gaussian_object_tracks"
echo "[DONE] Object spheres: out/object_spheres/{spheres.json,spheres.glb}"
echo "[DONE] Viewer: vis/gaussian_splat_overlay/index.html"

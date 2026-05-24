# Docker for Orange Pose Estimation

A self-contained image with everything the pipeline needs: CUDA 12.8 dev, gcc-11,
Python 3.12, PyTorch 2.10+cu128, SAM3, speedy-splat (with CUDA extensions),
nvdiffrast, FlexiCubes, Mitsuba 3, gsplat, COLMAP, ffmpeg, Open3D, and the
rest. Built because the host (Arch Linux + gcc 15 + CUDA 13.2 driver) can't
build nvdiffrast / FlexiCubes — gcc>14 is unsupported by nvcc 12.x.

## Prerequisites (one-time host setup)

On Arch Linux:

```bash
# Start Docker daemon
sudo systemctl enable --now docker.service

# Add your user to the docker group so you don't need sudo for every command
sudo usermod -aG docker "$USER"
# log out + back in (or `newgrp docker`) for the group change to apply

# GPU passthrough — required to use --gpus all
yay -S nvidia-container-toolkit       # or pacman if it's in your repos
sudo systemctl restart docker.service

# (Optional but recommended) modern BuildKit-based builder
sudo pacman -S docker-buildx
```

Verify with:
```bash
docker info | grep -i nvidia        # should mention an nvidia runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## One-time build

```bash
docker/build.sh
```

Takes 20–40 min on first run, mostly compiling the CUDA extensions
(`diff-gaussian-rasterization`, `simple-knn`, `nvdiffrast`). Subsequent builds
reuse the layer cache and are seconds.

If you change `Dockerfile` content above the `COPY .` line (e.g. a new pip
install) the heavy layers rebuild. Changes to `scripts/`, `configs/`, `util/`,
etc. only invalidate the final tiny layer — no recompile needed.

## Running

The wrapper script mounts the project at `/workspace`, exposes the web UI
ports (8765, 8766), and gives the container GPU access:

```bash
# Interactive shell inside container, in /workspace:
docker/run.sh

# Run the full pipeline end-to-end:
docker/run.sh bash run_pipeline.sh

# Launch the web UI (browse to http://localhost:8765):
docker/run.sh python util/pipeline_web.py --port 8765

# One-off script:
docker/run.sh python scripts/get_masks.py --config configs/gaussian_pipeline.json --video raw_videos/uploads/my.mp4
```

All outputs land in `out/`, `vis/`, `logs/` on the host (volume-mounted), so
they persist after the container exits.

## Environment variables

| Var          | Default               | Meaning                                |
|--------------|-----------------------|----------------------------------------|
| `IMAGE_TAG`  | `orange-pose:latest`  | Tag to build / run                     |
| `WEB_PORT`   | `8765`                | Host port for `pipeline_web.py`        |
| `OVERLAY_PORT` | `8766`              | Host port for `serve_gaussian_overlay.py` |

## What's installed

Pinned to match the host's working versions so CUDA-built extensions share the
PyTorch ABI:

- Python 3.12 (Ubuntu 22.04 from `deadsnakes` PPA)
- `torch==2.10.0` + `torchvision==0.25.0`, both `+cu128`
- `nvdiffrast` (built from `third_party/nvdiffrast`)
- `sam3==0.1.3` (editable from `sam3/`)
- `speedy-splat`'s `diff-gaussian-rasterization` + `simple-knn` (editable)
- `FlexiCubes` (pure-Python, on `PYTHONPATH`)
- `gsplat==1.5.3`, `mitsuba==3.8.0`, `drjit==1.3.1`
- `open3d==0.19.0`, `opencv-python==4.11.0.86`, `trimesh`, `pymeshfix`, `pymeshlab`
- System: `colmap`, `ffmpeg`, `libgl1`, `libosmesa6`

## Troubleshooting

**Build OOMs.** Some CUDA extension compiles use 6+ GB of RAM. If your machine
has <16 GB, reduce parallelism:
```bash
MAX_JOBS=2 docker/build.sh
```

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`.**
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):
```bash
# Arch (AUR):
yay -S nvidia-container-toolkit
sudo systemctl restart docker
```

**Web UI not reachable.** The wrapper forwards 8765/8766; if you start the
server inside the container on a different port, also forward it:
```bash
WEB_PORT=9000 docker/run.sh python util/pipeline_web.py --port 8765
# or just:
docker run --gpus all -p 9000:9000 ... orange-pose python util/pipeline_web.py --port 9000
```

**Rebuild after editing the Dockerfile.** Same `docker/build.sh` — buildkit
caches layers, so only what changed (and what depends on it) rebuilds.

# syntax=docker/dockerfile:1.7
#
# Orange Pose Estimation - simple orange-segmentation pipeline image.
#
# Pipeline: SAM3 -> COLMAP -> 3DGS (speedy-splat) -> vote -> track ->
# infer ownership -> fit OBB + inscribed sphere per object.
#
# Layer ordering is *deliberate* to keep cache hits high:
#   1. apt + python venv      (rare changes)
#   2. PyTorch                (rare changes — pinned versions)
#   3. CUDA extensions        (speedy-splat: ~3-5 min to compile)
#   4. Heavy ML pip deps      (most likely to churn; placed AFTER CUDA
#                              extensions so adding/removing a small pip
#                              package doesn't trigger a CUDA rebuild)
#   5. SAM3 editable install  (cheap)
#   6. Project source         (cheap; final COPY .)
#
# BuildKit cache mounts keep pip's wheel cache + ccache between rebuilds.
# Even if a layer hash changes, the actual compile-from-source step is
# dramatically faster because object files are reused.
#
# Build:
#   docker build -t orange-pose -f docker/Dockerfile .
#
# Inspect cache reuse:
#   docker build --progress=plain -t orange-pose -f docker/Dockerfile .

ARG CUDA_VERSION=12.8.0
ARG CUDNN_VERSION=cudnn
FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_VERSION}-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.local/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# ----------------------------------------------------------------------------
# System packages: build toolchain, COLMAP, ffmpeg, GL libs, Python 3.12.
# Cached unless this RUN line changes.
# ----------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        wget \
        gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        cmake \
        git \
        git-lfs \
        ninja-build \
        pkg-config \
        ffmpeg \
        libgl1 \
        libosmesa6 \
        libglu1-mesa \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libtbb12 \
        libspatialindex-c6 \
        libavformat-dev \
        libavcodec-dev \
        libswscale-dev \
        libavutil-dev \
        libavdevice-dev \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3.12-tk

# Python venv at /opt/venv so all subsequent pip installs go in there
# (avoids "Cannot uninstall blinker 1.4: distutils installed project"
# class of conflicts with apt-shipped packages).
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python - && \
    python -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# ----------------------------------------------------------------------------
# COLMAP via conda-forge (CUDA build). The Ubuntu apt `colmap` package is
# compiled WITHOUT CUDA SIFT, so feature_extractor/sequential_matcher fall
# back to CPU even with --SiftExtraction.use_gpu 1. conda-forge ships
# `colmap=*=cuda_*` builds that link against the proper CUDA toolkit.
#
# We use micromamba (a single ~5 MB binary) to install colmap into a
# dedicated prefix at /opt/colmap so it doesn't tangle with the Python venv.
# Only the colmap binary's parent dir lands on PATH.
# ----------------------------------------------------------------------------
# Extract just `bin/micromamba` from the tarball into /usr/local, so it
# lands at /usr/local/bin/micromamba (already on PATH).
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local bin/micromamba && \
    chmod +x /usr/local/bin/micromamba && \
    micromamba --version

# Install the CUDA build of COLMAP into a dedicated prefix. We pass
# MAMBA_ROOT_PREFIX as an env var (more portable than --root-prefix, which
# is positional-sensitive). The pkgs cache lives in the BuildKit cache
# mount and never bloats the image layer, so no clean step is needed.
ENV MAMBA_ROOT_PREFIX=/root/micromamba
RUN --mount=type=cache,target=/root/micromamba/pkgs \
    micromamba create -y -p /opt/colmap \
        -c conda-forge \
        "colmap=*=cuda_*"

# Expose ONLY the colmap binary on PATH (don't add /opt/colmap/bin wholesale,
# because that dir contains conda's `python`, `pip`, etc. which would shadow
# our /opt/venv/bin/python and break every pip-installed module like cv2.
# Conda binaries have RPATH set to find their own libs, so no LD_LIBRARY_PATH
# pollution is needed either.
RUN ln -sf /opt/colmap/bin/colmap /usr/local/bin/colmap

# ----------------------------------------------------------------------------
# PyTorch 2.10 + CUDA 12.8. Pinned versions = stable layer hash.
# Cache mount keeps the ~3 GB of wheels around between rebuilds.
# ----------------------------------------------------------------------------
# setuptools<81 -- newer versions removed `pkg_resources`, which SAM3 still
# imports (sam3/sam3/utils/* references it at top level). Pin until SAM3
# migrates to importlib.metadata.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip "setuptools<81" wheel && \
    pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
        torchvision==0.25.0

# ----------------------------------------------------------------------------
# CUDA extensions — placed EARLY so the slow compile (~3-5 min) is rarely
# invalidated. The only inputs are the two submodule directories; we copy
# *only those* (not the whole speedy-splat tree or whole project) so a code
# edit elsewhere can't bust this cache.
#
# --no-build-isolation: setup.py imports torch at build time to locate the
# CUDA headers via torch.utils.cpp_extension. Build isolation hides the torch
# we just installed.
#
# Both pip's wheel cache and a torch_extensions cache are mounted so even
# if this layer IS invalidated, only the truly-changed sources recompile —
# unchanged .cu / .cpp files reuse their cached .o.
# ----------------------------------------------------------------------------
COPY third_party/speedy-splat/submodules/diff-gaussian-rasterization \
     /workspace/third_party/speedy-splat/submodules/diff-gaussian-rasterization
COPY third_party/speedy-splat/submodules/simple-knn \
     /workspace/third_party/speedy-splat/submodules/simple-knn
COPY third_party/gaussian-splatting/submodules/fused-ssim \
     /workspace/third_party/gaussian-splatting/submodules/fused-ssim

# Use ccache via the symlink dir at /usr/lib/ccache (`gcc`, `g++`, `cc`, `c++`
# all symlink to /usr/bin/ccache there). Putting that dir first on PATH means
# every gcc/g++ call transparently routes through ccache without nvcc or
# torch's cpp_extension.py ever knowing. Setting CC="ccache gcc" the obvious
# way fails because nvcc passes `-ccbin "ccache gcc"` as a *single arg* and
# then can't find a binary literally named "ccache gcc".
ENV PATH=/usr/lib/ccache:${PATH} \
    CCACHE_DIR=/root/.cache/ccache \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/torch_extensions \
    pip install --no-build-isolation \
        /workspace/third_party/speedy-splat/submodules/diff-gaussian-rasterization && \
    pip install --no-build-isolation \
        /workspace/third_party/speedy-splat/submodules/simple-knn && \
    pip install --no-build-isolation \
        /workspace/third_party/gaussian-splatting/submodules/fused-ssim

# ----------------------------------------------------------------------------
# Heavy Python deps. Placed AFTER the CUDA compile so a pip-list edit
# (adding open-clip-torch, psutil, etc.) doesn't trigger a CUDA rebuild.
# Pip's wheel cache is preserved across builds via the cache mount.
# ----------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        numpy==1.26.4 \
        scipy \
        scikit-image \
        scikit-learn \
        matplotlib \
        imageio \
        pillow \
        opencv-python==4.11.0.86 \
        open3d==0.19.0 \
        plyfile \
        trimesh \
        rtree \
        embreex \
        networkx \
        tqdm \
        timm \
        einops \
        torchmetrics \
        open-clip-torch \
        ftfy \
        iopath \
        psutil \
        pycocotools \
        pandas \
        h5py \
        gitpython \
        fastapi \
        python-multipart \
        gsplat==1.5.3

# ----------------------------------------------------------------------------
# SAM3 editable install. Pure Python; very fast; uses sam3 dir as the input.
# We COPY only sam3/ here so changes outside sam3/ don't bust this layer.
# ----------------------------------------------------------------------------
COPY sam3 /workspace/sam3
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e /workspace/sam3

# ----------------------------------------------------------------------------
# Project source last (cheap to rebuild on every iteration).
# ----------------------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

# ----------------------------------------------------------------------------
# Runtime entrypoint - default to serving the web UI on port 8765 so
# `docker/run.sh` (no args) brings up the pipeline browser interface.
# Override with anything else for one-off shells / scripts, e.g.
#   docker/run.sh bash
#   docker/run.sh python scripts/get_masks.py --help
# ----------------------------------------------------------------------------
EXPOSE 8765
CMD ["python", "util/pipeline_web.py", "--host", "0.0.0.0", "--port", "8765"]

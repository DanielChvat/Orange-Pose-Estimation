import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a COLMAP dataset for 3DGS from extracted video frames.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-dir", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--reconstruction-fps", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    return namespace_from_config(
        cli.config,
        "colmap",
        {
            "frame_dir": cli.frame_dir,
            "video_path": cli.video,
            "reconstruction_fps": cli.reconstruction_fps,
            "out_dir": cli.out_dir,
            "overwrite": cli.overwrite,
        },
    )


def run(cmd):
    print("[RUN]", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    subprocess.run(cmd, check=True, env=env)


def detect_gpu_flags(colmap_bin):
    """COLMAP's GPU-selection flag has changed name across versions:
        3.7  -> --FeatureExtraction.use_gpu / --FeatureMatching.use_gpu
        3.8+ -> --SiftExtraction.use_gpu   / --SiftMatching.use_gpu
        3.10+ conda-forge builds may remove the flag entirely (auto-detect).
    We discover the right names by parsing `colmap feature_extractor -h` and
    return (extract_flag, match_flag) where each is None if not supported.
    """
    def help_text(subcmd):
        try:
            proc = subprocess.run(
                [colmap_bin, subcmd, "--help"],
                capture_output=True, text=True, timeout=20,
            )
        except Exception:
            return ""
        return (proc.stdout or "") + (proc.stderr or "")

    extract_h = help_text("feature_extractor")
    match_h = help_text("sequential_matcher")

    extract_flag = None
    for candidate in ("--SiftExtraction.use_gpu", "--FeatureExtraction.use_gpu"):
        if candidate in extract_h:
            extract_flag = candidate
            break
    match_flag = None
    for candidate in ("--SiftMatching.use_gpu", "--FeatureMatching.use_gpu"):
        if candidate in match_h:
            match_flag = candidate
            break
    return extract_flag, match_flag


def run_with_gpu_fallback(cmd, gpu_flag, use_gpu, cleanup_paths=()):
    try:
        run(cmd)
        return bool(use_gpu)
    except subprocess.CalledProcessError:
        if not use_gpu:
            raise
        print(f"[WARN] COLMAP GPU step failed; retrying with {gpu_flag} 0")
        for path in cleanup_paths:
            path = Path(path)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        fallback = list(cmd)
        try:
            idx = fallback.index(gpu_flag)
            fallback[idx + 1] = "0"
        except (ValueError, IndexError):
            fallback.extend([gpu_flag, "0"])
        run(fallback)
        return False


def copy_frames(frame_dir, input_dir):
    os.makedirs(input_dir, exist_ok=True)
    frames = sorted(p for p in Path(frame_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise RuntimeError(f"No images found in {frame_dir}")
    for src in frames:
        shutil.copy2(src, Path(input_dir) / src.name)
    return frames


def load_frame_manifest(frame_dir):
    path = Path(frame_dir) / "frames_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return {
        Path(record.get("file_name", "")).name: record
        for record in data.get("frames", [])
        if record.get("file_name")
    }


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_reconstruction_frames(video_path, input_dir, fps):
    if not video_path or float(fps or 0) <= 0:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for reconstruction frames: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Could not read FPS from video: {video_path}")

    interval = max(1, int(round(video_fps / float(fps))))
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    existing = {
        file_digest(path)
        for path in Path(input_dir).iterdir()
        if path.suffix.lower() in IMAGE_EXTS
    }

    saved = 0
    frame_idx = 0
    records = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            if rotation in rotations:
                frame = cv2.rotate(frame, rotations[rotation])
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                digest = hashlib.sha256(encoded.tobytes()).hexdigest()
                if digest not in existing:
                    out_path = Path(input_dir) / f"recon_{frame_idx:06d}.jpg"
                    with open(out_path, "wb") as f:
                        f.write(encoded.tobytes())
                    existing.add(digest)
                    records.append({
                        "file_name": out_path.name,
                        "kind": "reconstruction",
                        "source_frame_idx": int(frame_idx),
                        "time_sec": float(frame_idx / video_fps),
                    })
                    saved += 1
        frame_idx += 1

    cap.release()
    print(
        f"[INFO] Added {saved} reconstruction-only frames "
        f"from {total_frames} source frames at {fps:g} FPS"
    )
    return records


def normalize_sparse_layout(out_dir):
    sparse_dir = Path(out_dir) / "sparse"
    target = sparse_dir / "0"
    target.mkdir(parents=True, exist_ok=True)
    for path in sparse_dir.iterdir():
        if path.name == "0":
            continue
        shutil.move(str(path), str(target / path.name))


def count_registered_images(model_dir):
    images_bin = Path(model_dir) / "images.bin"
    if not images_bin.exists():
        return 0
    try:
        with open(images_bin, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    except Exception as exc:
        print(f"[WARN] Could not parse {images_bin}: {exc}")
        return 0


def sparse_model_score(model_dir):
    model_dir = Path(model_dir)
    image_count = count_registered_images(model_dir)
    points_size = (model_dir / "points3D.bin").stat().st_size if (model_dir / "points3D.bin").exists() else 0
    images_size = (model_dir / "images.bin").stat().st_size if (model_dir / "images.bin").exists() else 0
    return image_count, points_size, images_size


def select_best_sparse_model(sparse_dir, frame_count, min_registered_ratio=0.15, min_registered_images=8):
    candidates = sorted(p for p in Path(sparse_dir).iterdir() if p.is_dir())
    if not candidates:
        raise RuntimeError("COLMAP mapper did not produce a sparse model")

    scored = [(sparse_model_score(candidate), candidate) for candidate in candidates]
    for score, candidate in scored:
        print(
            f"[INFO] COLMAP component {candidate.name}: "
            f"{score[0]} registered images, points3D.bin {score[1]} bytes"
        )

    score, model_dir = max(scored, key=lambda item: item[0])
    min_required = min(frame_count, max(min_registered_images, int(round(frame_count * min_registered_ratio))))
    if score[0] < min_required:
        raise RuntimeError(
            f"Best COLMAP component only registered {score[0]}/{frame_count} images "
            f"(need at least {min_required}). This reconstruction is too weak for 3DGS."
        )

    print(f"[INFO] Selected COLMAP component {model_dir.name} for undistortion")
    return model_dir


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"{out_dir} exists. Use --overwrite.")
    input_dir = out_dir / "input"
    distorted_sparse = out_dir / "distorted" / "sparse"
    distorted_sparse.mkdir(parents=True, exist_ok=True)
    frames = copy_frames(args.frame_dir, input_dir)
    source_manifest = load_frame_manifest(args.frame_dir)
    copied_records = []
    for idx, src in enumerate(frames):
        source_record = dict(source_manifest.get(src.name, {}))
        source_record.setdefault("source_frame_idx", int(idx))
        source_record.setdefault("time_sec", float(idx))
        source_record["file_name"] = src.name
        source_record["kind"] = source_record.get("kind", "sam")
        copied_records.append(source_record)
    print(f"[INFO] Copied {len(frames)} SAM frames to {input_dir}")
    reconstruction_records = extract_reconstruction_frames(
        getattr(args, "video_path", None),
        input_dir,
        getattr(args, "reconstruction_fps", 0.0),
    )
    input_images = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    manifest_records = {record["file_name"]: record for record in copied_records + reconstruction_records}
    for idx, path in enumerate(input_images):
        manifest_records.setdefault(
            path.name,
            {
                "file_name": path.name,
                "kind": "unknown",
                "source_frame_idx": int(idx),
                "time_sec": float(idx),
            },
        )
    ordered_records = sorted(
        manifest_records.values(),
        key=lambda record: (float(record.get("time_sec", record.get("source_frame_idx", 0))), record["file_name"]),
    )
    (out_dir / "frame_manifest.json").write_text(json.dumps({
        "video_path": str(getattr(args, "video_path", "") or ""),
        "input_dir": str(input_dir),
        "frames": ordered_records,
    }, indent=2))
    print(f"[INFO] COLMAP will use {len(input_images)} total images")

    database = out_dir / "distorted" / "database.db"
    use_gpu = bool(args.use_gpu)
    feature_gpu = "1" if use_gpu else "0"

    # COLMAP's GPU flag was renamed several times across versions.
    # Discover what THIS build accepts before invoking.
    extract_gpu_flag, match_gpu_flag = detect_gpu_flags(args.colmap)
    if extract_gpu_flag is None:
        print("[INFO] COLMAP feature_extractor has no explicit --use_gpu flag; "
              "relying on auto-detection (CUDA-enabled builds use GPU automatically).")
    else:
        print(f"[INFO] COLMAP feature_extractor uses {extract_gpu_flag} for GPU selection")
    if match_gpu_flag is None:
        print("[INFO] COLMAP matcher has no explicit --use_gpu flag; relying on auto-detection.")
    else:
        print(f"[INFO] COLMAP matcher uses {match_gpu_flag} for GPU selection")

    extract_cmd = [
        args.colmap,
        "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1" if args.single_camera else "0",
        "--ImageReader.camera_model", args.camera_model,
        "--SiftExtraction.max_image_size", str(args.max_image_size),
    ]
    if extract_gpu_flag is not None:
        extract_cmd.extend([extract_gpu_flag, feature_gpu])
        use_gpu = run_with_gpu_fallback(
            extract_cmd, extract_gpu_flag, use_gpu, cleanup_paths=[database]
        )
    else:
        run(extract_cmd)

    matcher_cmd = "exhaustive_matcher" if args.matcher == "exhaustive" else "sequential_matcher"
    matcher_gpu = "1" if use_gpu else "0"
    matcher_args = [
        args.colmap,
        matcher_cmd,
        "--database_path", str(database),
    ]
    if match_gpu_flag is not None:
        matcher_args.extend([match_gpu_flag, matcher_gpu])
    if args.matcher == "sequential":
        matcher_args.extend([
            "--SequentialMatching.overlap",
            str(int(getattr(args, "sequential_overlap", 12))),
        ])
        # Loop detection requires a pre-trained vocab tree file (~250 MB);
        # without one COLMAP segfaults. Only enable if a path is configured
        # AND the file exists. For short single-pass videos like ours,
        # overlap-based sequential matching is sufficient without loop closure.
        vocab_tree = getattr(args, "vocab_tree_path", None)
        if vocab_tree and Path(vocab_tree).exists():
            matcher_args.extend([
                "--SequentialMatching.loop_detection", "1",
                "--SequentialMatching.vocab_tree_path", str(vocab_tree),
            ])
        else:
            matcher_args.extend(["--SequentialMatching.loop_detection", "0"])
    if match_gpu_flag is not None:
        run_with_gpu_fallback(matcher_args, match_gpu_flag, use_gpu)
    else:
        run(matcher_args)

    run([
        args.colmap,
        "mapper",
        "--database_path", str(database),
        "--image_path", str(input_dir),
        "--output_path", str(distorted_sparse),
        "--Mapper.ba_global_function_tolerance", "0.000001",
    ])

    model_dir = select_best_sparse_model(
        distorted_sparse,
        len(input_images),
        min_registered_ratio=float(getattr(args, "min_registered_ratio", 0.15)),
        min_registered_images=int(getattr(args, "min_registered_images", 8)),
    )

    run([
        args.colmap,
        "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(model_dir),
        "--output_path", str(out_dir),
        "--output_type", "COLMAP",
    ])
    normalize_sparse_layout(out_dir)
    print(f"[DONE] Wrote COLMAP/3DGS dataset to {out_dir}")
    print(f"[INFO] Train with: python scripts/run_3dgs.py --config configs/gaussian_pipeline.json --source {out_dir} --model out/3dgs --iterations 7000")


if __name__ == "__main__":
    main()

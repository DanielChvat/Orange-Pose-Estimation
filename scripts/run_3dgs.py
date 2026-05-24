import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config
from pipeline_progress import progress_path, write_progress


TRAINING_PROGRESS_RE = re.compile(r"Training progress:\s+\d+%.*?\|\s*([0-9]+)/([0-9]+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GraphDeco Gaussian Splatting trainer on an exported COLMAP scene.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--skip-import-check", action="store_true")
    cli = parser.parse_args()
    return namespace_from_config(
        cli.config,
        "splat_training",
        {
            "source": cli.source,
            "model": cli.model,
            "iterations": cli.iterations,
            "skip_import_check": cli.skip_import_check,
        },
    )


def check_imports(python):
    missing = []
    for module in ["diff_gaussian_rasterization", "simple_knn._C", "fused_ssim"]:
        code = f"import torch\nimport {module}"
        result = subprocess.run([python, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            missing.append(module)
    return missing


def run_training(cmd, cwd):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_report_bucket = -1
    total_iters = None
    final_iter = 0
    emit_progress = progress_path() is not None
    recent_lines = deque(maxlen=240)

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        recent_lines.append(line)
        progress = TRAINING_PROGRESS_RE.search(line)
        if emit_progress and progress:
            current = int(progress.group(1))
            total = int(progress.group(2))
            total_iters = total
            final_iter = max(final_iter, current)
            write_progress("Training 3DGS", current, total, "iter")
            bucket_size = max(total // 20, 1)
            bucket = current // bucket_size
            if bucket != last_report_bucket:
                print(f"[INFO] Training 3DGS {current}/{total} ({100.0 * current / max(total, 1):.1f}%)", flush=True)
                last_report_bucket = bucket
            continue
        if emit_progress and line.startswith("Training progress:"):
            continue
        if line.strip():
            print(line, flush=True)

    returncode = process.wait()
    if emit_progress and total_iters is not None:
        write_progress("Training 3DGS", final_iter if returncode else total_iters, total_iters, "iter", done=returncode == 0)
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd, output="\n".join(recent_lines))


def resolve_data_device(value):
    if value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def is_cuda_memory_failure(exc):
    output = (getattr(exc, "output", None) or "").casefold()
    needles = [
        "cuda out of memory",
        "torch.outofmemoryerror",
        "cublas_status_alloc_failed",
        "cusolver_status_alloc_failed",
        "out of memory",
    ]
    return any(needle in output for needle in needles)


def trainer_supports(repo, option):
    repo = Path(repo)
    if option == "--disable_viewer":
        return option in (repo / "train.py").read_text(errors="ignore")
    if option == "--antialiasing":
        args_file = repo / "arguments" / "__init__.py"
        return args_file.exists() and "antialiasing" in args_file.read_text(errors="ignore")
    return True


def main():
    args = parse_args()
    python = sys.executable
    repo = os.path.abspath(args.repo)
    source = os.path.abspath(args.source)
    model = os.path.abspath(args.model)
    data_device = resolve_data_device(args.data_device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not os.path.exists(os.path.join(repo, "train.py")):
        raise FileNotFoundError(f"Could not find GraphDeco train.py under {repo}")
    sparse_dir = os.path.join(source, "sparse", "0")
    has_text_colmap = os.path.exists(os.path.join(sparse_dir, "cameras.txt"))
    has_binary_colmap = os.path.exists(os.path.join(sparse_dir, "cameras.bin"))
    if not (has_text_colmap or has_binary_colmap):
        raise FileNotFoundError(
            f"{source} does not look like a COLMAP scene. "
            "Expected sparse/0/cameras.txt or sparse/0/cameras.bin."
        )

    if not args.skip_import_check:
        missing = check_imports(python)
        if missing:
            raise RuntimeError(
                "3DGS CUDA extensions are not installed: "
                + ", ".join(missing)
                + "\nThese should have been built into the docker image. "
                + "Rebuild with:\n  docker build -t orange-pose:latest -f docker/Dockerfile ."
            )

    densify_until_iter = args.densify_until_iter
    if densify_until_iter is None:
        densify_until_iter = max(int(args.iterations) // 2, 0)

    cmd = [
        python,
        "train.py",
        "-s", source,
        "-m", model,
        "-r", str(args.resolution),
        "--data_device", data_device,
        "--iterations", str(args.iterations),
        "--save_iterations", str(args.iterations),
        "--test_iterations", "-1",
        "--densify_until_iter", str(densify_until_iter),
        "--densify_grad_threshold", str(args.densify_grad_threshold),
        "--opacity_reset_interval", str(args.opacity_reset_interval),
    ]
    if args.disable_viewer and trainer_supports(repo, "--disable_viewer"):
        cmd.append("--disable_viewer")
    elif args.disable_viewer:
        print(f"[INFO] Trainer at {repo} does not support --disable_viewer; skipping it")
    if args.antialiasing and trainer_supports(repo, "--antialiasing"):
        cmd.append("--antialiasing")
    elif args.antialiasing:
        print(f"[INFO] Trainer at {repo} does not support --antialiasing; skipping it")
    print(f"[INFO] 3DGS image data device: {data_device}")
    print("[RUN]", " ".join(cmd))
    try:
        run_training(cmd, cwd=repo)
    except subprocess.CalledProcessError as exc:
        if data_device != "cuda" or not is_cuda_memory_failure(exc):
            raise
        print("[WARN] 3DGS hit CUDA memory pressure; retrying with --data_device cpu")
        if os.path.exists(model):
            shutil.rmtree(model)
        cpu_cmd = list(cmd)
        device_idx = cpu_cmd.index("--data_device") + 1
        cpu_cmd[device_idx] = "cpu"
        print("[RUN]", " ".join(cpu_cmd))
        run_training(cpu_cmd, cwd=repo)


if __name__ == "__main__":
    main()

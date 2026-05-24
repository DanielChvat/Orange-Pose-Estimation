import argparse
import functools
import hashlib
import http.server
import io
import json
import mimetypes
import os
import re
import signal
import shutil
import socketserver
import subprocess
import threading
import time
import warnings
from urllib.parse import parse_qs, unquote, urlparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


STATIC_DIR = Path(__file__).resolve().parent / "pipeline_web_static"


def load_config(path, root):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(root) / config_path
    with open(config_path) as f:
        return json.load(f)


def default_settings(config_path, root):
    config = load_config(config_path, root)
    masks = config.get("masks", {})
    colmap = config.get("colmap", {})
    return {
        "iterations": int(config.get("pipeline", {}).get("iterations", 7000) or 7000),
        "sam_fps": float(masks.get("fps", 4.0) or 4.0),
        "reconstruction_fps": float(colmap.get("reconstruction_fps", 6.0) or 0.0),
        "full_image_pass": bool(masks.get("full_image_pass", True)),
        "detail_pass": bool(masks.get("detail_pass", True)),
        "tile_context_margin": int(masks.get("tile_context_margin", -1)),
    }


def coerce_int(value, fallback, minimum=1):
    try:
        return max(minimum, int(float(value)))
    except (TypeError, ValueError):
        return fallback


def coerce_float(value, fallback, minimum=0.001):
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return fallback


def coerce_bool(value, fallback):
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def read_video_frame(path, frame_index):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        raise ValueError(f"Could not read frame count from video: {path}")

    frame_index = max(0, min(int(frame_index), total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index}")

    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation in rotations:
        frame = cv2.rotate(frame, rotations[rotation])
    return frame_index, frame


class PipelineState:
    STEPS = [
        "[0/7] Cleaning stale outputs",
        "[1/7] SAM masks",
        "[2/7] COLMAP dataset for 3DGS",
        "[3/7] 3DGS",
        "[4/7] Voting SAM evidence onto Gaussians",
        "[5/7] Assigning IDs to foreground splat blobs",
        "[6/7] Fitting OBB + inscribed sphere per object",
        "[7/7] Building browser viewer",
    ]

    def __init__(self, root, python, config):
        self.root = Path(root).resolve()
        self.python = python
        self.config = config
        self.lock = threading.Lock()
        self.process = None
        self.log_path = self.root / "logs" / "pipeline_web.log"
        self.progress_path = self.root / "logs" / "pipeline_progress.json"
        self.last_video = None
        self.last_prompt = None
        self.started_at = None
        self.returncode = None
        self.last_prompts = None
        self.last_test_prompts = None
        self.last_settings = default_settings(config, self.root)

    def running(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def effective_settings(self, settings=None):
        defaults = default_settings(self.config, self.root)
        settings = settings or {}
        return {
            "iterations": coerce_int(settings.get("iterations"), defaults["iterations"]),
            "sam_fps": coerce_float(settings.get("sam_fps"), defaults["sam_fps"]),
            "reconstruction_fps": coerce_float(
                settings.get("reconstruction_fps"),
                defaults["reconstruction_fps"],
                minimum=0.0,
            ),
            "full_image_pass": coerce_bool(settings.get("full_image_pass"), defaults["full_image_pass"]),
            "detail_pass": coerce_bool(settings.get("detail_pass"), defaults["detail_pass"]),
            "tile_context_margin": coerce_int(
                settings.get("tile_context_margin"),
                defaults["tile_context_margin"],
                minimum=-1,
            ),
        }

    def start(self, video_path, prompts, settings=None):
        prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        prompt = " | ".join(prompts)
        settings = self.effective_settings(settings)
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("Pipeline is already running")

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.progress_path.exists():
                self.progress_path.unlink()
            log = open(self.log_path, "w", buffering=1)
            env = os.environ.copy()
            env.update({
                "CONFIG": self.config,
                "VIDEO": str(video_path),
                "PROMPT": prompt,
                "ITERATIONS": str(settings["iterations"]),
                "SAM_FPS": str(settings["sam_fps"]),
                "COLMAP_RECONSTRUCTION_FPS": str(settings["reconstruction_fps"]),
                "SAM_FULL_IMAGE_PASS": str(settings["full_image_pass"]).lower(),
                "SAM_DETAIL_PASS": str(settings["detail_pass"]).lower(),
                "SAM_TILE_CONTEXT_MARGIN": str(settings["tile_context_margin"]),
                "PYTHON": self.python,
                "PYTHONUNBUFFERED": "1",
                "PIPELINE_PROGRESS": str(self.progress_path),
            })
            # stdbuf -oL/-eL forces line-buffered stdout/stderr in the subprocess
            # so the bash script's `echo "[2/7] ..."` markers reach the log file
            # as they happen, not in 8 KB block flushes. Without this the UI
            # can't tell which step is running during long stages like COLMAP
            # and falls back to "Starting pipeline" + everything-queued.
            self.process = subprocess.Popen(
                ["stdbuf", "-oL", "-eL", "bash", "run_pipeline.sh"],
                cwd=self.root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.last_video = str(video_path)
            self.last_prompt = prompt
            self.last_prompts = prompts
            self.last_settings = settings
            self.started_at = time.time()
            self.returncode = None
            threading.Thread(target=self._watch, args=(self.process, log), daemon=True).start()

    def stop(self):
        stopped = False
        with self.lock:
            process = self.process

        pgid = None
        if process is not None and process.poll() is None:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
                stopped = True
            except ProcessLookupError:
                pgid = None

        config_pattern = re.escape(str(self.config))
        patterns = [
            r"bash run_pipeline\.sh",
            rf"scripts/get_masks\.py --config {config_pattern}",
            rf"scripts/run_colmap\.py --config {config_pattern}",
            rf"scripts/run_3dgs\.py --config {config_pattern}",
            rf"scripts/vote_gaussian_masks\.py --config {config_pattern}",
            rf"scripts/track_gaussian_objects\.py --config {config_pattern}",
            rf"scripts/infer_gaussian_ownership\.py --config {config_pattern}",
            rf"scripts/fit_object_spheres\.py --config {config_pattern}",
            rf"util/create_gaussian_overlay_viewer\.py --config {config_pattern}",
        ]
        for pattern in patterns:
            result = subprocess.run(
                ["pkill", "-TERM", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            stopped = stopped or result.returncode == 0

        if process is not None:
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                for pattern in patterns:
                    subprocess.run(
                        ["pkill", "-KILL", "-f", pattern],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

        try:
            self.progress_path.unlink()
        except FileNotFoundError:
            pass

        return stopped

    def _watch(self, process, log):
        returncode = process.wait()
        log.close()
        with self.lock:
            if self.process is process:
                self.returncode = returncode

    def snapshot(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            has_process = self.process is not None
            returncode = None if self.process is None or running else self.process.returncode
            if self.returncode is not None:
                returncode = self.returncode
            log = ""
            if self.log_path.exists():
                text = self.log_path.read_text(errors="replace")
                log = text[-20000:]
            progress = self.read_progress()
            external_running = False
            if not running and not has_process and progress and not progress.get("done"):
                updated_at = float(progress.get("updated_at") or 0.0)
                external_running = time.time() - updated_at < 120.0
                running = external_running
            if not running and not has_process:
                progress = None
            current_step = self.current_step(log) if has_process or external_running else None
            error = self.error_summary(log) if has_process else None
            label = self.status_label(running, returncode, current_step, error)
            return {
                "running": running,
                "returncode": returncode,
                "last_video": self.last_video,
                "last_prompt": self.last_prompt,
                "last_prompts": self.last_prompts or split_prompts(self.last_prompt or "orange fruit"),
                "settings": self.last_settings or self.effective_settings(),
                "started_at": self.started_at,
                "current_step": current_step,
                "progress": progress,
                "error": error,
                "label": label,
                "log": log,
                "steps": self.STEPS,
            }

    def read_progress(self):
        if not self.progress_path.exists():
            return None
        try:
            with open(self.progress_path) as f:
                return json.load(f)
        except Exception:
            return None

    def current_step(self, log):
        current = None
        for line in log.splitlines():
            stripped = line.strip()
            for step in self.STEPS:
                if stripped.startswith(step):
                    current = step
            if stripped.startswith("[DONE]"):
                current = "[DONE]"
        return current

    def error_summary(self, log):
        for line in reversed(log.splitlines()):
            stripped = line.strip()
            if stripped.startswith("[ERROR]"):
                return stripped
            if "Traceback (most recent call last)" in stripped:
                return "Python traceback; see log below."
            if "Error" in stripped or "Exception" in stripped:
                return stripped[-500:]
        return None

    def status_label(self, running, returncode, current_step, error):
        if running:
            return current_step or "Starting pipeline"
        if returncode is None:
            return "Idle"
        if returncode == 0:
            return "Done"
        return f"Failed (exit {returncode})" + (f": {error}" if error else "")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_existing_upload(upload_dir, digest_hex, ignore_path):
    for path in upload_dir.iterdir():
        if not path.is_file() or path == ignore_path or path.name.startswith(".upload_"):
            continue
        try:
            if file_sha256(path) == digest_hex:
                return path
        except OSError:
            continue
    return None


def store_uploaded_video(upload_dir, video_item):
    upload_dir.mkdir(parents=True, exist_ok=True)
    original = Path(video_item.filename or "upload.mp4").name
    suffix = (Path(original).suffix or ".mp4").lower()
    digest = hashlib.sha256()
    tmp_path = upload_dir / f".upload_{time.time_ns()}{suffix}"
    with open(tmp_path, "wb") as f:
        while True:
            chunk = video_item.file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            f.write(chunk)

    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original).stem)[:48].strip("._-")
    digest_hex = digest.hexdigest()
    existing_upload = find_existing_upload(upload_dir, digest_hex, tmp_path)
    if existing_upload is not None:
        tmp_path.unlink()
        return existing_upload, digest_hex, original

    name_parts = [digest_hex[:20]]
    if safe_stem:
        name_parts.append(safe_stem)
    video_path = upload_dir / ("_".join(name_parts) + suffix)
    if video_path.exists():
        tmp_path.unlink()
    else:
        tmp_path.replace(video_path)
    return video_path, digest_hex, original


def resolve_existing_upload(root, value):
    if not value:
        return None
    upload_dir = (Path(root) / "raw_videos" / "uploads").resolve()
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    path = path.resolve()
    if not str(path).startswith(str(upload_dir)) or not path.exists():
        return None
    return path


def split_prompts(prompt):
    parts = re.split(r"\s*(?:[;|,]|\band\b)\s*", str(prompt or ""), flags=re.IGNORECASE)
    prompts = []
    seen = set()
    for part in parts:
        value = part.strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(value)
    return prompts


def mask_records_from_npz(path):
    try:
        data = np.load(path, allow_pickle=True)
        records = data["objects"].tolist() if "objects" in data else []
    except Exception:
        return []
    clean = []
    for record in records:
        if not isinstance(record, dict):
            continue
        bbox = record.get("bbox", [0, 0, 0, 0])
        centroid = record.get("centroid", [0.0, 0.0])
        score = record.get("score")
        clean.append({
            "local_id": int(record.get("local_id", 0) or 0),
            "prompt": str(record.get("prompt", "")),
            "prompt_idx": int(record.get("prompt_idx", -1) or -1),
            "score": None if score is None else float(score),
            "area": int(record.get("area", 0) or 0),
            "bbox": [int(v) for v in bbox],
            "centroid": [float(v) for v in centroid],
            "color": color_for_id(int(record.get("local_id", 0) or 0)),
        })
    return clean


def prompt_counts(records):
    counts = {}
    for record in records:
        prompt = record.get("prompt") or "object"
        counts[prompt] = counts.get(prompt, 0) + 1
    return [{"prompt": prompt, "count": count} for prompt, count in sorted(counts.items())]


def prompt_count_map(records):
    return {item["prompt"]: item["count"] for item in prompt_counts(records)}


def prompts_from_log(log):
    for line in reversed(log.splitlines()):
        marker = "[INFO] SAM prompts:"
        if marker in line:
            return [part.strip() for part in line.split(marker, 1)[1].split(",") if part.strip()]
    return []


def find_frame_image(frame_dir, stem):
    for suffix in (".jpg", ".jpeg", ".png"):
        path = frame_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def sam_source_dirs(root, source):
    if source == "test":
        base = Path(root) / "out" / "sam_prompt_test"
        return base / "frames", base / "masks"
    return Path(root) / "out" / "frames", Path(root) / "out" / "masks"


def sam_source_from_path(path):
    params = parse_qs(urlparse(path).query)
    source = (params.get("source") or ["pipeline"])[0]
    return source if source in {"pipeline", "test"} else "pipeline"


def color_for_id(idx):
    rng = np.random.default_rng(int(idx) * 7919 + 17)
    return [int(v) for v in rng.integers(50, 255, size=3)]


def parse_safe_frame_and_id(path):
    parsed = urlparse(path)
    params = parse_qs(parsed.query)
    frame = (params.get("frame") or [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", frame or ""):
        raise ValueError("Bad frame name")
    local_id = int((params.get("id") or ["0"])[0])
    if local_id <= 0:
        raise ValueError("Bad object id")
    return frame, local_id


class PipelineHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
        ".ply": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path in {"/", "/index.html"}:
            self.write_page()
            return
        if request_path.startswith("/assets/"):
            self.serve_pipeline_asset()
            return
        if request_path.startswith("/viewer/"):
            self.serve_viewer()
            return
        if request_path == "/status":
            self.write_status()
            return
        if request_path == "/status.json":
            self.write_status_json()
            return
        if request_path == "/sam-status.json":
            self.write_sam_status_json()
            return
        if request_path.startswith("/video-frame"):
            self.write_video_frame()
            return
        if request_path.startswith("/sam-frame-preview"):
            self.write_sam_frame_preview()
            return
        if request_path.startswith("/sam-preview"):
            self.write_sam_preview()
            return
        if request_path.startswith("/sam-cutout"):
            self.write_sam_cutout()
            return
        self.send_error(404, "Not found")

    def do_HEAD(self):
        request_path = urlparse(self.path).path
        if request_path.startswith("/assets/"):
            self.serve_pipeline_asset(head_only=True)
            return
        if request_path.startswith("/viewer/"):
            self.serve_viewer()
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        if self.path == "/inspect-video":
            self.inspect_video_upload()
            return
        if self.path == "/stop":
            self.stop_pipeline()
            return
        if self.path == "/test-sam":
            self.test_sam_prompt_frame()
            return
        if self.path != "/run":
            self.send_error(404, "Not found")
            return
        self.start_pipeline_from_upload()

    def serve_pipeline_asset(self, head_only=False):
        parsed = urlparse(self.path)
        rel = unquote(parsed.path.removeprefix("/assets/"))
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            self.send_error(400, "Bad asset path")
            return
        path = STATIC_DIR / rel_path
        if not path.exists() or not path.is_file():
            self.send_error(404, "Asset not found")
            return
        data = b"" if head_only else path.read_bytes()
        self.send_response(200)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def write_html(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_page(self):
        state = self.server.state.snapshot()
        prompt_values = state.get("last_prompts") or ["orange fruit"]
        settings = state.get("settings") or self.server.state.effective_settings()
        initial = {
            "state": state,
            "prompts": prompt_values,
            "settings": settings,
        }
        initial_json = json.dumps(initial).replace("</", "<\\/")
        body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gaussian Object Pipeline</title>
  <link rel="stylesheet" href="/assets/pipeline.css">
</head>
<body>
<main>
  <header class="appHeader">
    <div>
      <h1>Gaussian Object Pipeline</h1>
      <p class="muted">Upload a video, tune prompts, inspect SAM masks, and run the local 3DGS object pipeline.</p>
    </div>
    <button id="settingsButton" class="iconButton" type="button" aria-label="Pipeline settings">Settings</button>
  </header>

  <div class="grid">
    <form id="runForm" class="panel" method="post" action="/run" enctype="multipart/form-data">
      <input id="existingVideo" name="existing_video" type="hidden">
      <input id="iterationsHidden" name="iterations" type="hidden">
      <input id="samFpsHidden" name="sam_fps" type="hidden">
      <input id="reconstructionFpsHidden" name="reconstruction_fps" type="hidden">
      <input id="fullImagePassHidden" name="full_image_pass" type="hidden">
      <input id="detailPassHidden" name="detail_pass" type="hidden">
      <input id="tileContextMarginHidden" name="tile_context_margin" type="hidden">

      <label>Video
        <input id="videoInput" name="video" type="file" accept="video/*">
      </label>
      <div id="videoInfo" class="fieldHint">Choose a video to inspect frame count and SAM sampling.</div>

      <label>Object detections</label>
      <div id="objectList" class="objectList"></div>
      <button id="addObject" class="secondary" type="button">Add New Object Detection</button>

      <section id="promptTest" class="promptTest" hidden>
        <strong>Test Prompt</strong>
        <div class="rangeRow">
          <input id="testFrameSlider" type="range" min="0" max="0" value="0">
          <input id="testFrameIndex" type="number" min="0" value="0">
        </div>
        <div class="previewShell"><img id="testFramePreview" alt="Selected video frame"></div>
        <div class="row">
          <button id="testSamButton" class="secondary" type="button">Test Prompts On This Frame</button>
          <span id="testSamStatus" class="fieldHint"></span>
        </div>
      </section>

      <button id="runButton" type="submit">Run Pipeline</button>
    </form>

    <section class="statusPanel panel">
      <div class="row statusActions">
        <strong id="stateText">Idle</strong>
        <button id="stopButton" class="danger" type="button" disabled>Stop Pipeline</button>
        <a class="button" href="/viewer/index.html" target="_blank" rel="noopener">Open Viewer</a>
        <a class="button" href="/status">Raw Log</a>
      </div>
      <p id="errorText"></p>
      <div class="pipelineRun">
        <div class="pipelineRunHeader">
          <div>
            <h2>Pipeline Run</h2>
            <p class="muted">A compact timeline of the active reconstruction stage.</p>
          </div>
          <span id="overallPill" class="overallPill">Overall</span>
        </div>
        <div id="progressWrap">
          <div id="progressMeta"><span id="progressDesc">No active progress</span><span id="progressCount"></span></div>
          <div id="progressOuter"><div id="progressInner"></div></div>
        </div>
        <div id="stageGraph" class="stageGraph" aria-label="Pipeline stage timeline"></div>
      </div>
    </section>
  </div>

  <div class="tabs">
    <button id="logTabButton" class="tabButton active" type="button">Pipeline Log</button>
    <button id="samTabButton" class="tabButton" type="button">SAM Masks</button>
  </div>

  <section id="logPanel" class="viewPanel">
    <pre id="logText"></pre>
  </section>

  <section id="samPanel" class="viewPanel panel" hidden>
    <div class="samToolbar">
      <label>Mask Source
        <select id="samSourceSelect">
          <option value="pipeline">Raw SAM masks</option>
          <option value="test">Prompt test masks</option>
        </select>
      </label>
      <div class="samFrameControls">
        <button id="samPrevFrame" class="secondary" type="button">Prev</button>
        <button id="samNextFrame" class="secondary" type="button">Next</button>
        <input id="samFrameSlider" type="range" min="0" max="0" value="0">
        <input id="samFrameIndex" type="number" min="0" max="0" value="0">
        <button id="samLiveFrame" class="secondary" type="button">Live</button>
      </div>
      <span id="samFrameLabel" class="fieldHint">No frames yet</span>
    </div>
    <div class="samLayout">
      <div>
        <div class="samImageBox">
          <img id="samBasePreview" class="samBasePreview" alt="Raw SAM frame">
          <img id="samPreview" class="samOverlayPreview" alt="SAM mask overlay">
        </div>
        <p id="samSummary" class="muted">SAM output will appear once masks are being written.</p>
      </div>
      <div>
        <h3>Frame Cutouts</h3>
        <div id="samBins" class="binColumn"></div>
      </div>
    </div>
  </section>
</main>

<dialog id="settingsDialog">
  <div class="dialogHeader">
    <strong>Pipeline Settings</strong>
    <button id="closeSettings" class="secondary" type="button">Close</button>
  </div>
  <div class="dialogBody">
    <label>3DGS iterations
      <input id="iterationsInput" type="number" min="100" step="100">
    </label>
    <label>SAM target FPS / sampling window
      <input id="samFpsInput" type="number" min="0.1" step="0.1">
      <span id="samEstimate" class="fieldHint"></span>
    </label>
    <label>3DGS reconstruction FPS
      <input id="reconstructionFpsInput" type="number" min="0" step="0.5">
      <span class="fieldHint">Extra video frames for COLMAP/3DGS only. SAM still runs at the SAM target FPS above.</span>
    </label>
    <label>
      <span><input id="fullImagePassInput" type="checkbox"> Whole-frame SAM pass</span>
      <span class="fieldHint">Uses full-image context for prompt grounding before tiled recall.</span>
    </label>
    <label>
      <span><input id="detailPassInput" type="checkbox"> Padded tile recall pass</span>
      <span class="fieldHint">Adds higher-detail tiled detections while keeping only each tile core.</span>
    </label>
    <label>Padded tile context margin
      <input id="tileContextMarginInput" type="number" min="-1" step="1">
      <span class="fieldHint">Use -1 for automatic padding from the tile overlap.</span>
    </label>
  </div>
</dialog>

<script id="pipelineInitial" type="application/json">{initial_json}</script>
<script src="/assets/pipeline.js" defer></script>
</body>
</html>"""
        self.write_html(body)

    def write_json(self, value):
        data = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_status_json(self):
        self.write_json(self.server.state.snapshot())

    def write_status(self):
        state = self.server.state.snapshot()
        text = state["log"]
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def sam_status_payload(self, source="pipeline", prompts=None, log_text="", selected_frame=None):
        root = self.server.state.root
        _, mask_dir = sam_source_dirs(root, source)
        files = sorted(mask_dir.glob("*.npz")) if mask_dir.exists() else []
        frames = []
        selected_records = []
        selected_stem = None
        total_records = []
        for path in files:
            records = mask_records_from_npz(path)
            item = {
                "frame": path.stem,
                "object_count": len(records),
                "counts": prompt_counts(records),
            }
            frames.append(item)
            total_records.extend(records)
            if selected_frame and path.stem == selected_frame:
                selected_records = records
                selected_stem = path.stem
        if files and selected_stem is None:
            selected_stem = files[-1].stem
            selected_records = mask_records_from_npz(files[-1])
        prompts = list(prompts or [])
        if not prompts:
            prompts = prompts_from_log(log_text)
        if not prompts:
            prompts = sorted(prompt_count_map(total_records))
        return {
            "source": source,
            "prompts": prompts,
            "processed_frames": len(files),
            "latest_frame": selected_stem,
            "frames": frames,
            "recent_frames": frames[-60:],
            "latest_objects": selected_records,
            "total_counts": prompt_count_map(total_records),
        }

    def write_sam_status_json(self):
        source = sam_source_from_path(self.path)
        params = parse_qs(urlparse(self.path).query)
        selected_frame = (params.get("frame") or [None])[0]
        if selected_frame is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", selected_frame or ""):
            self.send_error(400, "Bad frame name")
            return
        with self.server.state.lock:
            if source == "test":
                prompts = list(self.server.state.last_test_prompts or [])
            else:
                prompts = list(self.server.state.last_prompts or [])
            log_text = self.server.state.log_path.read_text(errors="replace") if self.server.state.log_path.exists() else ""
        payload = self.sam_status_payload(
            source=source,
            prompts=prompts,
            log_text=log_text,
            selected_frame=selected_frame,
        )
        self.write_json(payload)

    def write_sam_preview(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        frame = (params.get("frame") or [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", frame or ""):
            self.send_error(400, "Bad frame name")
            return

        root = self.server.state.root
        frame_dir, mask_dir = sam_source_dirs(root, sam_source_from_path(self.path))
        mask_path = mask_dir / f"{frame}.npz"
        frame_path = find_frame_image(frame_dir, frame)
        if frame_path is None or not mask_path.exists():
            self.send_error(404, "SAM preview frame not found")
            return

        try:
            image = Image.open(frame_path).convert("RGBA")
            data = np.load(mask_path, allow_pickle=True)
            mask = data["instance_mask"].astype(np.int32)
        except Exception as exc:
            self.send_error(500, f"Could not render SAM preview: {exc}")
            return

        if mask.shape[0] != image.height or mask.shape[1] != image.width:
            mask_img = Image.fromarray(mask.astype(np.int32), mode="I").resize(
                image.size,
                Image.Resampling.NEAREST,
            )
            mask = np.array(mask_img, dtype=np.int32)

        overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        for local_id in np.unique(mask):
            if local_id <= 0:
                continue
            color = color_for_id(local_id)
            overlay[mask == local_id] = [color[0], color[1], color[2], 105]

        preview = Image.alpha_composite(image, Image.fromarray(overlay, mode="RGBA"))
        draw = ImageDraw.Draw(preview)
        records = mask_records_from_npz(mask_path)
        for record in records:
            cx, cy = record.get("centroid", [0, 0])
            color = record.get("color", color_for_id(record.get("local_id", 0)))
            label = record.get("prompt", "") or "object"
            x, y = int(cx), int(cy)
            label_width = max(36, 6 * len(label))
            draw.rectangle((x + 4, y - 16, x + 28 + label_width, y + 4), fill=(0, 0, 0, 155))
            draw.rectangle((x + 8, y - 12, x + 19, y - 1), fill=tuple(color) + (230,), outline=(255, 255, 255, 210))
            draw.text((x + 24, y - 14), label, fill=(255, 255, 255, 235))

        max_width = 900
        if preview.width > max_width:
            scale = max_width / preview.width
            preview = preview.resize((max_width, int(preview.height * scale)), Image.Resampling.LANCZOS)

        out = io.BytesIO()
        preview.convert("RGB").save(out, format="JPEG", quality=88)
        payload = out.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def write_sam_frame_preview(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        frame = (params.get("frame") or [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", frame or ""):
            self.send_error(400, "Bad frame name")
            return

        root = self.server.state.root
        frame_dir, _ = sam_source_dirs(root, sam_source_from_path(self.path))
        frame_path = find_frame_image(frame_dir, frame)
        if frame_path is None:
            self.send_error(404, "SAM frame not found")
            return

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as exc:
            self.send_error(500, f"Could not render SAM frame: {exc}")
            return

        max_width = 900
        if image.width > max_width:
            scale = max_width / image.width
            image = image.resize((max_width, int(image.height * scale)), Image.Resampling.LANCZOS)

        out = io.BytesIO()
        image.save(out, format="JPEG", quality=88)
        payload = out.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def write_sam_cutout(self):
        try:
            frame, local_id = parse_safe_frame_and_id(self.path)
        except (TypeError, ValueError):
            self.send_error(400, "Bad SAM cutout request")
            return

        root = self.server.state.root
        frame_dir, mask_dir = sam_source_dirs(root, sam_source_from_path(self.path))
        mask_path = mask_dir / f"{frame}.npz"
        frame_path = find_frame_image(frame_dir, frame)
        if frame_path is None or not mask_path.exists():
            self.send_error(404, "SAM cutout not found")
            return

        try:
            image = Image.open(frame_path).convert("RGBA")
            data = np.load(mask_path, allow_pickle=True)
            mask = data["instance_mask"].astype(np.int32)
        except Exception as exc:
            self.send_error(500, f"Could not load SAM cutout: {exc}")
            return

        if mask.shape[0] != image.height or mask.shape[1] != image.width:
            mask_img = Image.fromarray(mask.astype(np.int32), mode="I").resize(
                image.size,
                Image.Resampling.NEAREST,
            )
            mask = np.array(mask_img, dtype=np.int32)

        obj_mask = mask == local_id
        if not obj_mask.any():
            self.send_error(404, "Object id not found in frame")
            return

        ys, xs = np.nonzero(obj_mask)
        pad = 10
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(image.width, int(xs.max()) + pad + 1)
        y1 = min(image.height, int(ys.max()) + pad + 1)

        crop = image.crop((x0, y0, x1, y1))
        alpha = (obj_mask[y0:y1, x0:x1].astype(np.uint8) * 255)
        alpha = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=0.35))
        crop.putalpha(alpha)

        max_side = 180
        if max(crop.size) > max_side:
            scale = max_side / max(crop.size)
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)

        out = io.BytesIO()
        crop.save(out, format="PNG")
        payload = out.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_viewer(self):
        # The viewer is produced by util/create_gaussian_overlay_viewer.py at
        # the end of the pipeline and lives at vis/gaussian_splat_overlay/.
        # That dir contains its own index.html plus the per-layer PLY/GLB
        # files it references with plain relative paths.
        viewer_root = self.server.state.root / "vis" / "gaussian_splat_overlay"
        parsed = urlparse(self.path)
        relative = unquote(parsed.path[len("/viewer/"):]) or "index.html"
        target = (viewer_root / relative).resolve()
        if not str(target).startswith(str(viewer_root.resolve())) or not target.exists():
            self.send_error(404,
                "Viewer not available. Run the pipeline to completion so "
                "out/object_spheres/index.html is generated.")
            return
        self.path = "/" + relative
        old_directory = self.directory
        self.directory = str(viewer_root)
        try:
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        finally:
            self.directory = old_directory

    def parse_multipart_form(self):
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )

    def write_video_frame(self):
        params = parse_qs(urlparse(self.path).query)
        video_path = resolve_existing_upload(self.server.state.root, (params.get("path") or [""])[0])
        if video_path is None:
            self.send_error(400, "Video must be uploaded before frames can be previewed")
            return

        try:
            frame_index = int((params.get("frame") or ["0"])[0])
            _, frame = read_video_frame(video_path, frame_index)
        except Exception as exc:
            self.send_error(500, str(exc))
            return

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            self.send_error(500, "Could not encode frame preview")
            return
        payload = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def inspect_video_upload(self):
        form = self.parse_multipart_form()
        if "video" not in form:
            self.send_error(400, "Expected video field")
            return
        upload_dir = self.server.state.root / "raw_videos" / "uploads"
        try:
            video_path, digest_hex, original = store_uploaded_video(upload_dir, form["video"])
            info = probe_video(video_path)
        except Exception as exc:
            self.send_error(500, str(exc))
            return
        payload = {
            "path": str(video_path),
            "name": original,
            "sha256": digest_hex,
            **info,
        }
        self.write_json(payload)

    def form_prompts(self, form):
        prompt_items = form.getlist("prompts") if "prompts" in form else form.getlist("prompt")
        prompts = [str(item.value if hasattr(item, "value") else item).strip() for item in prompt_items]
        return [prompt for prompt in prompts if prompt]

    def form_settings(self, form):
        settings = {}
        for key in ("iterations", "sam_fps", "reconstruction_fps", "full_image_pass", "detail_pass", "tile_context_margin"):
            if key in form:
                settings[key] = str(form[key].value).strip()
        return settings

    def test_sam_prompt_frame(self):
        if self.server.state.running():
            self.send_error(409, "Wait for the current pipeline run to finish before testing prompts")
            return

        form = self.parse_multipart_form()
        prompts = self.form_prompts(form)
        if not prompts:
            self.send_error(400, "Add at least one object detection prompt")
            return

        existing_video = form["existing_video"].value if "existing_video" in form else ""
        video_path = resolve_existing_upload(self.server.state.root, existing_video)
        if video_path is None:
            if "video" not in form:
                self.send_error(400, "Expected video field")
                return
            upload_dir = self.server.state.root / "raw_videos" / "uploads"
            video_path, _, _ = store_uploaded_video(upload_dir, form["video"])

        try:
            frame_index = int(form["frame_index"].value if "frame_index" in form else 0)
            _, frame = read_video_frame(video_path, frame_index)
        except Exception as exc:
            self.send_error(500, str(exc))
            return

        settings = self.server.state.effective_settings(self.form_settings(form))
        test_root = self.server.state.root / "out" / "sam_prompt_test"
        frame_dir = test_root / "frames"
        mask_dir = test_root / "masks"
        if test_root.exists():
            shutil.rmtree(test_root)
        frame_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        frame_path = frame_dir / "frame_0000.jpg"
        cv2.imwrite(str(frame_path), frame)

        with self.server.state.lock:
            self.server.state.last_test_prompts = list(prompts)

        cmd = [
            self.server.state.python,
            "scripts/get_masks.py",
            "--config",
            self.server.state.config,
            "--out-root",
            str(test_root),
            "--no-extract",
            "--full-image-pass",
            str(settings["full_image_pass"]).lower(),
            "--detail-pass",
            str(settings["detail_pass"]).lower(),
            "--tile-context-margin",
            str(settings["tile_context_margin"]),
        ]
        for prompt in prompts:
            cmd.extend(["--prompt", prompt])
        try:
            result = subprocess.run(
                cmd,
                cwd=self.server.state.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            self.send_error(504, "SAM prompt test timed out")
            return

        payload = self.sam_status_payload(source="test", prompts=prompts)
        payload["frame_index"] = int(frame_index)
        payload["ok"] = result.returncode == 0
        payload["log"] = result.stdout[-12000:]
        if result.returncode != 0:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            data = json.dumps(payload).encode("utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.write_json(payload)

    def stop_pipeline(self):
        stopped = self.server.state.stop()
        self.write_json({"stopped": stopped})

    def start_pipeline_from_upload(self):
        if self.server.state.running():
            self.send_error(409, "Pipeline is already running")
            return
        form = self.parse_multipart_form()

        prompts = self.form_prompts(form)
        if not prompts:
            self.send_error(400, "Add at least one object detection prompt")
            return

        existing_video = form["existing_video"].value if "existing_video" in form else ""
        video_path = resolve_existing_upload(self.server.state.root, existing_video)
        if video_path is None:
            if "video" not in form:
                self.send_error(400, "Expected video field")
                return
            upload_dir = self.server.state.root / "raw_videos" / "uploads"
            video_path, _, _ = store_uploaded_video(upload_dir, form["video"])

        settings = self.form_settings(form)

        try:
            self.server.state.start(video_path, prompts, settings=settings)
        except Exception as exc:
            self.send_error(500, str(exc))
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description="Local upload-and-run web app for the Gaussian object pipeline.")
    parser.add_argument("--config", type=str, default="configs/gaussian_pipeline.json")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    handler = functools.partial(PipelineHandler, directory=str(root))
    with ReusableThreadingTCPServer((args.host, args.port), handler) as httpd:
        httpd.state = PipelineState(root, args.python, args.config)
        print(f"[INFO] Pipeline web app: http://{args.host}:{args.port}/")
        print("[INFO] Press Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

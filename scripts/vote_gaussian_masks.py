import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config
from pipeline_progress import progress_iter


SH_C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser(description="Vote SAM mask evidence directly onto 3D Gaussian splats.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--gaussians", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    return namespace_from_config(
        cli.config,
        "voting",
        {
            "gaussians": cli.gaussians,
            "mask_dir": cli.mask_dir,
            "out_dir": cli.out_dir,
            "overwrite": cli.overwrite,
        },
    )


def add_3dgs_to_path():
    # Speedy-Splat ships the same `scene.colmap_loader` module as vanilla 3DGS;
    # we fall back to the old path if a user kept the vanilla repo around.
    for candidate in ("third_party/speedy-splat", "third_party/gaussian-splatting"):
        repo = Path(candidate).resolve()
        if repo.exists():
            sys.path.insert(0, str(repo))
            return


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    x = np.clip(x, 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


def parse_rgb(text):
    values = [float(part) for part in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected r,g,b color, got {text}")
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def load_colmap_model(path):
    add_3dgs_to_path()
    from scene.colmap_loader import (
        qvec2rotmat,
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )

    path = Path(path)
    cameras_path_bin = path / "cameras.bin"
    images_path_bin = path / "images.bin"
    cameras_path_txt = path / "cameras.txt"
    images_path_txt = path / "images.txt"
    if cameras_path_bin.exists() and images_path_bin.exists():
        cameras = read_intrinsics_binary(str(cameras_path_bin))
        images = read_extrinsics_binary(str(images_path_bin))
    elif cameras_path_txt.exists() and images_path_txt.exists():
        cameras = read_intrinsics_text(str(cameras_path_txt))
        images = read_extrinsics_text(str(images_path_txt))
    else:
        raise FileNotFoundError(f"Could not find COLMAP cameras/images under {path}")
    records = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras[image.camera_id]
        records.append((image.name, camera, qvec2rotmat(image.qvec), np.asarray(image.tvec, dtype=np.float64)))
    return records


def camera_params(camera):
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera.model in {"SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise ValueError(f"Unsupported COLMAP camera model for voting: {camera.model}")
    return int(camera.width), int(camera.height), fx, fy, cx, cy


def load_instance_mask(path, key):
    data = np.load(path, allow_pickle=True)
    for candidate in [key, "instance_mask", "local_instance_mask", "mask"]:
        if candidate in data:
            mask = np.asarray(data[candidate])
            if mask.ndim == 3:
                mask = mask.max(axis=0)
            return mask.astype(np.int32)
    raise KeyError(f"No mask key like {key} in {path}")


def load_mask_records(path):
    data = np.load(path, allow_pickle=True)
    if "objects" not in data:
        return {}
    records = {}
    for item in data["objects"].tolist():
        if not isinstance(item, dict):
            continue
        local_id = int(item.get("local_id", len(records) + 1))
        records[local_id] = item
    return records


def collect_prompt_slots(mask_dir):
    prompt_to_slot = {}
    for path in sorted(Path(mask_dir).glob("*.npz")):
        for record in load_mask_records(path).values():
            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                prompt = f"prompt_{record.get('prompt_idx', len(prompt_to_slot))}"
            if prompt not in prompt_to_slot:
                prompt_to_slot[prompt] = len(prompt_to_slot)
    return prompt_to_slot


def load_union_mask(mask_dir, image_name, key, width, height):
    path = Path(mask_dir) / f"{Path(image_name).stem}.npz"
    if not path.exists():
        return None
    mask = load_instance_mask(path, key) > 0
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def otsu_threshold(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[(values > 0.0) & (values <= 1.0)]
    if len(values) < 32:
        return 0.5
    hist, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    prob = hist.astype(np.float64)
    total = prob.sum()
    if total <= 0:
        return 0.5
    prob /= total
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    score = np.zeros_like(denom)
    valid = denom > 1e-12
    score[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]
    return float(centers[int(np.argmax(score))])


def project_points(xyz, rotation, translation, camera):
    width, height, fx, fy, cx, cy = camera_params(camera)
    cam = xyz @ rotation.T + translation[None, :]
    z = cam[:, 2]
    valid_z = z > 1e-6
    u = fx * (cam[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (cam[:, 1] / np.maximum(z, 1e-6)) + cy
    valid = valid_z & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    return u, v, z, valid, width, height


def visible_indices(u, v, z, valid, width, height, margin_ratio):
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return idx, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    ui = np.floor(u[idx]).astype(np.int32)
    vi = np.floor(v[idx]).astype(np.int32)
    pix = vi * width + ui
    nearest = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(nearest, pix, z[idx].astype(np.float32))
    finite_depth = z[idx][np.isfinite(z[idx])]
    if len(finite_depth) == 0:
        margin = 0.0
    else:
        margin = max(float(np.median(finite_depth) * margin_ratio), 1e-4)
    keep = z[idx] <= nearest[pix] + margin
    return idx[keep], ui[keep], vi[keep]


def set_graphdeco_color(vertex, indices, rgb):
    if len(indices) == 0:
        return
    for channel, value in enumerate(rgb):
        vertex[f"f_dc_{channel}"][indices] = (float(value) - 0.5) / SH_C0
    for name in vertex.dtype.names or ():
        if name.startswith("f_rest_"):
            vertex[name][indices] = 0.0


def colorize_scene(vertex, selected, fg_rgb, bg_rgb, fg_opacity, bg_opacity, dim_background):
    output = vertex.copy()
    all_idx = np.arange(len(output))
    bg_idx = all_idx[~selected]
    fg_idx = all_idx[selected]
    if dim_background:
        set_graphdeco_color(output, bg_idx, bg_rgb)
    set_graphdeco_color(output, fg_idx, fg_rgb)
    if "opacity" in output.dtype.names:
        if dim_background:
            output["opacity"][bg_idx] = np.minimum(output["opacity"][bg_idx], logit(bg_opacity))
        output["opacity"][fg_idx] = np.maximum(output["opacity"][fg_idx], logit(fg_opacity))
    return output


def write_ply(path, template_ply, vertex):
    PlyData([PlyElement.describe(vertex, "vertex")], text=template_ply.text).write(path)


def gaussian_max_scale(vertex):
    names = vertex.dtype.names or ()
    if not {"scale_0", "scale_1", "scale_2"}.issubset(names):
        return np.ones(len(vertex), dtype=np.float32)
    return np.exp(
        np.column_stack([
            np.asarray(vertex["scale_0"], dtype=np.float64),
            np.asarray(vertex["scale_1"], dtype=np.float64),
            np.asarray(vertex["scale_2"], dtype=np.float64),
        ])
    ).max(axis=1).astype(np.float32)


def selected_components(xyz, selected, radius, radius_multiplier):
    idx = np.flatnonzero(selected)
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int32), 0.0
    pts = xyz[idx]
    tree = cKDTree(pts)
    if radius == "auto":
        if len(pts) == 1:
            eps = 0.01
        else:
            dists, _ = tree.query(pts, k=2)
            nn = dists[:, 1]
            nn = nn[np.isfinite(nn) & (nn > 0)]
            eps = float(np.median(nn) * radius_multiplier) if len(nn) else 0.01
    else:
        eps = float(radius)
    eps = max(eps, 1e-5)

    labels = np.full(len(idx), -1, dtype=np.int32)
    component_id = 0
    for start in range(len(idx)):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = component_id
        while stack:
            node = stack.pop()
            for nbr in tree.query_ball_point(pts[node], eps):
                if labels[nbr] < 0:
                    labels[nbr] = component_id
                    stack.append(nbr)
        component_id += 1
    return labels, eps


def clean_selection(vertex, xyz, selected, score, args):
    if args.no_clean_components or not selected.any():
        return selected, {
            "num_selected_raw": int(selected.sum()),
            "num_selected_clean": int(selected.sum()),
            "num_components_raw": 0,
            "num_components_kept": 0,
            "component_radius": 0.0,
        }, np.full(len(selected), -1, dtype=np.int32)

    scale = gaussian_max_scale(vertex)
    selected_idx = np.flatnonzero(selected)
    scale_limit = float(np.percentile(scale[selected_idx], args.max_selected_scale_percentile))
    scale_ok = scale <= scale_limit
    candidate = selected & scale_ok
    labels, eps = selected_components(xyz, candidate, args.component_radius, args.component_radius_multiplier)
    candidate_idx = np.flatnonzero(candidate)
    global_labels = np.full(len(selected), -1, dtype=np.int32)
    global_labels[candidate_idx] = labels

    if len(labels) == 0:
        return candidate, {
            "num_selected_raw": int(selected.sum()),
            "num_selected_clean": int(candidate.sum()),
            "num_components_raw": 0,
            "num_components_kept": 0,
            "component_radius": float(eps),
            "scale_limit": scale_limit,
        }, global_labels

    counts = np.bincount(labels)
    if args.min_component_splats == "auto":
        min_count = max(8, int(math.ceil(0.001 * max(len(candidate_idx), 1))))
    else:
        min_count = int(args.min_component_splats)
    keep_component = counts >= min_count
    cleaned = np.zeros_like(selected)
    cleaned[candidate_idx] = keep_component[labels]

    kept_labels = np.flatnonzero(keep_component)
    stats = {
        "num_selected_raw": int(selected.sum()),
        "num_selected_after_scale": int(candidate.sum()),
        "num_selected_clean": int(cleaned.sum()),
        "num_components_raw": int(len(counts)),
        "num_components_kept": int(len(kept_labels)),
        "component_radius": float(eps),
        "min_component_splats": int(min_count),
        "scale_limit": scale_limit,
        "mean_clean_score": float(score[cleaned].mean()) if cleaned.any() else 0.0,
        "median_clean_score": float(np.median(score[cleaned])) if cleaned.any() else 0.0,
    }
    return cleaned, stats, global_labels


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        for child in out_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(args.gaussians)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    cameras = load_colmap_model(args.colmap_model)
    hits = np.zeros(len(xyz), dtype=np.int32)
    seen = np.zeros(len(xyz), dtype=np.int32)
    prompt_to_slot = collect_prompt_slots(args.mask_dir)
    prompt_hits = np.zeros((len(prompt_to_slot), len(xyz)), dtype=np.uint16)
    mask_pixels = 0
    used_frames = 0

    # Per-frame observations: for every (frame, splat) pair where the splat
    # projected inside a non-background SAM mask, record (splat_idx, mask_id).
    # Stored CSR-style so the tracker can iterate frames cheaply.
    obs_splat_chunks = []
    obs_mask_chunks = []
    obs_frame_offsets = [0]
    obs_frame_names = []

    for image_name, camera, rotation, translation in progress_iter(cameras, desc="Voting SAM masks onto Gaussians"):
        width, height, *_ = camera_params(camera)
        mask_path = Path(args.mask_dir) / f"{Path(image_name).stem}.npz"
        if not mask_path.exists():
            continue
        mask = load_instance_mask(mask_path, args.mask_key)
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
        mask_records = load_mask_records(mask_path)
        union_mask = mask > 0
        used_frames += 1
        mask_pixels += int(union_mask.sum())
        u, v, z, valid, width, height = project_points(xyz, rotation, translation, camera)
        vis_idx, ui, vi = visible_indices(u, v, z, valid, width, height, args.depth_margin_ratio)
        if len(vis_idx) == 0:
            obs_frame_offsets.append(obs_frame_offsets[-1])
            obs_frame_names.append(str(image_name))
            continue
        local_ids = mask[vi, ui]
        in_mask = local_ids > 0
        seen[vis_idx] += 1
        hits[vis_idx[in_mask]] += 1
        hit_splats = vis_idx[in_mask].astype(np.int32)
        hit_masks = local_ids[in_mask].astype(np.int32)
        obs_splat_chunks.append(hit_splats)
        obs_mask_chunks.append(hit_masks)
        obs_frame_offsets.append(obs_frame_offsets[-1] + len(hit_splats))
        obs_frame_names.append(str(image_name))
        if len(prompt_to_slot):
            for local_id in np.unique(local_ids[in_mask]):
                record = mask_records.get(int(local_id), {})
                prompt = str(record.get("prompt", "")).strip()
                if not prompt:
                    prompt = f"prompt_{record.get('prompt_idx', len(prompt_to_slot))}"
                slot = prompt_to_slot.get(prompt)
                if slot is None:
                    continue
                prompt_hits[slot, vis_idx[local_ids == local_id]] += 1

    obs_splat_idx = (
        np.concatenate(obs_splat_chunks) if obs_splat_chunks else np.zeros(0, dtype=np.int32)
    )
    obs_mask_id = (
        np.concatenate(obs_mask_chunks) if obs_mask_chunks else np.zeros(0, dtype=np.int32)
    )
    obs_frame_offsets_arr = np.asarray(obs_frame_offsets, dtype=np.int64)
    obs_frame_names_arr = np.asarray(obs_frame_names, dtype=object)

    score = np.divide(hits, np.maximum(seen, 1), dtype=np.float32)
    eligible = seen >= args.min_visible
    if args.score_threshold == "auto":
        threshold = otsu_threshold(score[eligible])
    else:
        threshold = float(args.score_threshold)
    selected_raw = eligible & (score >= threshold) & (hits > 0)
    prompt_thresholds = []
    if len(prompt_to_slot):
        prompt_selected = np.zeros(len(xyz), dtype=bool)
        denom = np.maximum(seen, 1).astype(np.float32)
        for slot in range(prompt_hits.shape[0]):
            prompt_score = prompt_hits[slot].astype(np.float32) / denom
            prompt_eligible = eligible & (prompt_hits[slot] > 0)
            values = prompt_score[prompt_eligible]
            if args.score_threshold == "auto":
                prompt_threshold = otsu_threshold(values)
            else:
                prompt_threshold = threshold
            prompt_thresholds.append(float(prompt_threshold))
            prompt_selected |= prompt_eligible & (prompt_score >= prompt_threshold)
        if prompt_selected.any():
            selected_raw = prompt_selected
    selected, clean_stats, component_labels = clean_selection(vertex, xyz, selected_raw, score, args)

    fg_rgb = parse_rgb(args.foreground_color)
    bg_rgb = parse_rgb(args.background_color)
    highlighted = colorize_scene(
        vertex,
        selected,
        fg_rgb,
        bg_rgb,
        args.foreground_opacity,
        args.background_opacity,
        args.dim_background,
    )

    selected_vertex = vertex[selected].copy()
    set_graphdeco_color(selected_vertex, np.arange(len(selected_vertex)), fg_rgb)
    if "opacity" in selected_vertex.dtype.names:
        selected_vertex["opacity"][:] = np.maximum(selected_vertex["opacity"], logit(args.foreground_opacity))

    write_ply(out_dir / "objectness_scene_gaussians.ply", ply, highlighted)
    write_ply(out_dir / "object_only_gaussians.ply", ply, selected_vertex)
    raw_vertex = vertex[selected_raw].copy()
    set_graphdeco_color(raw_vertex, np.arange(len(raw_vertex)), fg_rgb)
    if "opacity" in raw_vertex.dtype.names:
        raw_vertex["opacity"][:] = np.maximum(raw_vertex["opacity"], logit(args.foreground_opacity))
    write_ply(out_dir / "object_only_raw_gaussians.ply", ply, raw_vertex)
    np.savez_compressed(
        out_dir / "votes.npz",
        hits=hits,
        seen=seen,
        score=score,
        selected=selected,
        selected_raw=selected_raw,
        component_labels=component_labels,
        xyz=xyz.astype(np.float32),
        prompt_hits=prompt_hits,
        prompt_names=np.asarray([prompt for prompt, _slot in sorted(prompt_to_slot.items(), key=lambda item: item[1])], dtype=object),
        prompt_thresholds=np.asarray(prompt_thresholds, dtype=np.float32),
        obs_splat_idx=obs_splat_idx,
        obs_mask_id=obs_mask_id,
        obs_frame_offsets=obs_frame_offsets_arr,
        obs_frame_names=obs_frame_names_arr,
    )
    stats = {
        "num_gaussians": int(len(xyz)),
        "num_selected": int(selected.sum()),
        "num_selected_raw": int(selected_raw.sum()),
        "selected_fraction": float(selected.mean()),
        "score_threshold": float(threshold),
        "prompt_score_thresholds": {
            prompt: float(prompt_thresholds[slot])
            for prompt, slot in prompt_to_slot.items()
            if slot < len(prompt_thresholds)
        },
        "used_frames": int(used_frames),
        "mask_pixels": int(mask_pixels),
        "mean_selected_score": float(score[selected].mean()) if selected.any() else 0.0,
        "median_selected_score": float(np.median(score[selected])) if selected.any() else 0.0,
        "dim_background": bool(args.dim_background),
        "cleanup": clean_stats,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"[DONE] Wrote voted Gaussian masks to {out_dir}")


if __name__ == "__main__":
    main()

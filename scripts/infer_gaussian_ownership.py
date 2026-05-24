import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config
from pipeline_progress import progress_iter
from track_gaussian_objects import colorize_gaussian_tracks, gaussian_rgb_from_vertex, load_mask_records, mask_path
from vote_gaussian_masks import camera_params, load_colmap_model, load_instance_mask, project_points, visible_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment full-scene Gaussians into object-owned layers from SAM mask evidence."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--gaussians", type=str, default=None)
    parser.add_argument("--tracks-npz", type=str, default=None)
    parser.add_argument("--tracks-json", type=str, default=None)
    parser.add_argument("--votes", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    return namespace_from_config(
        cli.config,
        "ownership",
        {
            "gaussians": cli.gaussians,
            "tracks_npz": cli.tracks_npz,
            "tracks_json": cli.tracks_json,
            "votes": cli.votes,
            "mask_dir": cli.mask_dir,
            "out_dir": cli.out_dir,
            "overwrite": cli.overwrite,
        },
    )


def write_ply(path, template_ply, vertex):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=template_ply.text).write(path)


def load_prompt_owner(votes, num_gaussians):
    if "prompt_hits" not in votes or "prompt_names" not in votes or "seen" not in votes:
        return None, []
    prompt_hits = np.asarray(votes["prompt_hits"])
    prompt_names = [str(item) for item in np.asarray(votes["prompt_names"], dtype=object).tolist()]
    if prompt_hits.ndim != 2 or prompt_hits.shape[1] != num_gaussians or prompt_hits.shape[0] != len(prompt_names):
        return None, prompt_names
    denom = np.maximum(np.asarray(votes["seen"], dtype=np.float32), 1.0)[None, :]
    prompt_score = prompt_hits.astype(np.float32) / denom
    owner = prompt_score.argmax(axis=0).astype(np.int16)
    best_hits = prompt_hits.max(axis=0)
    best_score = prompt_score.max(axis=0)
    owner[(best_hits <= 0) | (best_score <= 0.0)] = -1
    return owner, prompt_names


def load_object_metadata(path, labels):
    objects = {}
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        for item in data.get("objects", []):
            oid = int(item["id"])
            prompts = {str(prompt) for prompt in item.get("source_prompts", []) if prompt}
            if item.get("class_label"):
                prompts.add(str(item["class_label"]))
            objects[oid] = {
                "source_prompts": sorted(prompts),
                "seed_splats": int(np.count_nonzero(labels == oid)),
            }
    for oid in sorted(int(v) for v in np.unique(labels) if v > 0):
        objects.setdefault(
            oid,
            {
                "source_prompts": [],
                "seed_splats": int(np.count_nonzero(labels == oid)),
            },
        )
    return objects


def observation_prompts(obs):
    prompts = set()
    prompt = str(obs.get("prompt", "")).strip()
    if prompt:
        prompts.add(prompt)
    for value in obs.get("source_prompts", []) or []:
        value = str(value).strip()
        if value:
            prompts.add(value)
    return prompts


def prompt_slot_for_record(record, prompt_to_slot):
    prompt = str(record.get("prompt", "")).strip()
    if prompt and prompt in prompt_to_slot:
        return prompt_to_slot[prompt]
    for prompt in record.get("source_prompts", []) or []:
        prompt = str(prompt).strip()
        if prompt in prompt_to_slot:
            return prompt_to_slot[prompt]
    return None


def collect_full_mask_observations(args, xyz, candidate_mask, prompt_owner=None, prompt_names=None):
    cameras = [
        item
        for item in load_colmap_model(args.colmap_model)
        if mask_path(args.mask_dir, item[0]).exists()
    ]
    prompt_names = [] if prompt_names is None else list(prompt_names)
    prompt_to_slot = {str(prompt): idx for idx, prompt in enumerate(prompt_names)}
    observations = []
    for image_name, camera, rotation, translation in progress_iter(cameras, desc="Collecting full-mask Gaussian observations"):
        path = mask_path(args.mask_dir, image_name)
        if not path.exists():
            continue
        width, height, *_ = camera_params(camera)
        mask = load_instance_mask(path, args.mask_key)
        if mask.shape != (height, width):
            import cv2

            mask = cv2.resize(mask.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
        mask_records = load_mask_records(path)
        u, v, z, valid, width, height = project_points(xyz, rotation, translation, camera)
        visible, ui, vi = visible_indices(u, v, z, valid, width, height, margin_ratio=0.015)
        if len(visible) == 0:
            continue
        keep_candidate = candidate_mask[visible]
        visible = visible[keep_candidate]
        ui = ui[keep_candidate]
        vi = vi[keep_candidate]
        if len(visible) == 0:
            continue
        instance_ids = mask[vi, ui]
        for local_id in np.unique(instance_ids):
            if local_id <= 0:
                continue
            group = visible[instance_ids == local_id].astype(np.int64)
            if len(group) < int(args.min_observation_splats):
                continue
            record = mask_records.get(int(local_id), {})
            owner_slot = prompt_slot_for_record(record, prompt_to_slot)
            if prompt_owner is not None and owner_slot is not None:
                group = group[prompt_owner[group] == owner_slot]
                if len(group) < int(args.min_observation_splats):
                    continue
            observations.append({
                "frame": image_name,
                "local_id": int(local_id),
                "splats": group.astype(np.int64),
                "prompt": record.get("prompt", ""),
                "source_prompts": record.get("source_prompts", []),
                "prompt_idx": record.get("prompt_idx"),
            })
    return observations


def prompt_compatible(obs_prompts, obj_prompts):
    if not obs_prompts or not obj_prompts:
        return True
    return bool(obs_prompts & obj_prompts)


def pca_geometry(points):
    center = np.median(points, axis=0)
    if len(points) < 3:
        return center, np.ones(3, dtype=np.float64) * 0.01, np.eye(3)
    centered = points - center[None, :]
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    local = centered @ vecs
    axes = np.percentile(np.abs(local), 88, axis=0)
    axes = np.maximum(axes, 1e-4)
    return center, axes, vecs


def object_geometry_stats(xyz, labels):
    stats = {}
    for oid in sorted(int(v) for v in np.unique(labels) if v > 0):
        indices = np.flatnonzero(labels == oid)
        pts = xyz[indices]
        center, axes, rotation = pca_geometry(pts)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        stats[oid] = {
            "indices": indices,
            "center": center,
            "axes": axes,
            "rotation": rotation,
            "radius": max(float(np.percentile(dist, 90)), 1e-4),
            "count": int(len(indices)),
        }
    return stats


def nearby_candidate_mask(xyz, stats, multiplier):
    candidate = np.zeros(len(xyz), dtype=bool)
    multiplier = float(multiplier)
    if multiplier <= 0:
        return candidate
    for stat in stats.values():
        radius = max(float(stat["radius"]) * multiplier, 1e-5)
        dist = np.linalg.norm(xyz - stat["center"][None, :], axis=1)
        candidate |= dist <= radius
    return candidate


def assign_observation_to_object(obs, labels, object_prompts, args):
    splats = np.asarray(obs["splats"], dtype=np.int64)
    local_labels = labels[splats]
    ids, counts = np.unique(local_labels[local_labels > 0], return_counts=True)
    if len(ids) == 0:
        return None
    obs_prompts = observation_prompts(obs)
    candidates = []
    for oid, count in zip(ids, counts):
        oid = int(oid)
        if oid not in object_prompts:
            continue
        if not prompt_compatible(obs_prompts, object_prompts[oid]):
            continue
        candidates.append((oid, int(count)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_oid, best_count = candidates[0]
    second_count = candidates[1][1] if len(candidates) > 1 else 0
    ratio = best_count / max(float(len(splats)), 1.0)
    margin = (best_count - second_count) / max(float(best_count), 1.0)
    strong = best_count >= int(args.strong_mask_seed_splats)
    if best_count < int(args.min_mask_seed_splats):
        return None
    if ratio < float(args.min_mask_seed_ratio) and not strong:
        return None
    if margin < float(args.min_mask_seed_margin) and not strong:
        return None
    return best_oid


def build_mask_evidence(observations, labels, object_ids, candidate_index, object_prompts, args):
    object_to_row = {oid: row for row, oid in enumerate(object_ids)}
    num_candidates = int(np.count_nonzero(candidate_index >= 0))
    positive = np.zeros((len(object_ids), num_candidates), dtype=np.uint16)
    assigned_observations = {oid: 0 for oid in object_ids}
    skipped_ambiguous = 0

    for obs in progress_iter(observations, desc="Building segmented Gaussian evidence"):
        target = assign_observation_to_object(obs, labels, object_prompts, args)
        if target is None:
            skipped_ambiguous += 1
            continue
        row = object_to_row[target]
        local = candidate_index[np.asarray(obs["splats"], dtype=np.int64)]
        local = local[local >= 0]
        if len(local) == 0:
            continue
        positive[row, np.unique(local)] += 1
        assigned_observations[target] += 1
    return positive, assigned_observations, skipped_ambiguous


def add_seed_prior(logits, labels, candidate_indices, object_ids, weight):
    if weight <= 0:
        return
    for row, oid in enumerate(object_ids):
        local = np.flatnonzero(labels[candidate_indices] == oid)
        if len(local):
            logits[row, local] += float(weight)


def add_prompt_prior(logits, candidate_indices, object_ids, object_prompts, prompt_owner, prompt_names, args):
    if prompt_owner is None or not prompt_names or float(args.prompt_weight) <= 0:
        return
    prompt_to_slot = {prompt: idx for idx, prompt in enumerate(prompt_names)}
    candidate_prompt_owner = prompt_owner[candidate_indices]
    for row, oid in enumerate(object_ids):
        slots = [prompt_to_slot[prompt] for prompt in object_prompts.get(oid, set()) if prompt in prompt_to_slot]
        if not slots:
            continue
        match = np.isin(candidate_prompt_owner, np.asarray(slots, dtype=np.int16))
        logits[row, match] += float(args.prompt_weight)
        if float(args.prompt_mismatch_penalty) > 0:
            mismatch = (candidate_prompt_owner >= 0) & ~match
            logits[row, mismatch] -= float(args.prompt_mismatch_penalty)


def add_geometry_prior(logits, xyz, candidate_indices, object_ids, stats, args):
    weight = float(args.geometry_weight)
    if weight <= 0:
        return
    pts = xyz[candidate_indices]
    for row, oid in enumerate(object_ids):
        stat = stats.get(oid)
        if not stat:
            continue
        radius = max(float(stat["radius"]), 1e-5)
        dist = np.linalg.norm(pts - stat["center"][None, :], axis=1)
        sigma = radius * float(args.geometry_sigma_multiplier)
        prior = np.exp(-0.5 * (dist / max(sigma, 1e-6)) ** 2)
        prior[dist > radius * float(args.geometry_cutoff_multiplier)] = 0.0
        logits[row] += weight * prior.astype(np.float32)


def smooth_logits(logits, xyz, rgb, candidate_indices, args):
    iterations = int(args.smooth_iterations)
    neighbors = int(args.smooth_neighbors)
    if iterations <= 0 or neighbors <= 0 or logits.shape[1] <= 2:
        return logits
    pts = xyz[candidate_indices]
    colors = rgb[candidate_indices]
    neighbors = min(neighbors + 1, len(pts))
    tree = cKDTree(pts)
    dists, nbrs = tree.query(pts, k=neighbors)
    if nbrs.ndim == 1:
        return logits
    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]
    finite = dists[np.isfinite(dists) & (dists > 0)]
    sigma_x = float(np.median(finite)) * float(args.smooth_distance_multiplier) if len(finite) else 1.0
    sigma_x = max(sigma_x, 1e-6)
    color_delta = colors[:, None, :] - colors[nbrs]
    color_dist = np.linalg.norm(color_delta, axis=2)
    weights = np.exp(-0.5 * (dists / sigma_x) ** 2)
    weights *= np.exp(-0.5 * (color_dist / max(float(args.smooth_color_sigma), 1e-6)) ** 2)
    weights = weights.astype(np.float32)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, np.maximum(weights_sum, 1e-8), dtype=np.float32)

    base = logits.copy()
    out = logits.copy()
    smooth_weight = float(args.smooth_weight)
    for _ in range(iterations):
        stable = out - out.max(axis=0, keepdims=True)
        probs = np.exp(stable)
        probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-8)
        smoothed = np.zeros_like(out)
        for n in range(nbrs.shape[1]):
            smoothed += probs[:, nbrs[:, n]] * weights[:, n][None, :]
        out = base + smooth_weight * smoothed
    return out


def bbox_iou(b1, b2):
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def refine_obb_rotation_with_sam(obb_info, obj_pts, sam_views, angle_steps=36, aspect_weight=1.0, iou_weight=0.25, extent_percentile=99.0, padding=0.15):
    if not obb_info or not sam_views:
        return obb_info
    R0 = np.asarray(obb_info["rotation"], dtype=np.float64)
    center = np.asarray(obb_info["center"], dtype=np.float64)
    axis_1_0 = R0[:, 0]
    axis_2_0 = R0[:, 1]
    axis_3 = R0[:, 2]
    corner_signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    best_angle = 0.0
    best_score = -1.0
    best_aspect = 0.0
    best_iou = 0.0
    best_half_extents = np.asarray(obb_info["half_extents"], dtype=np.float64)
    centered = obj_pts - center
    for step in range(int(angle_steps)):
        angle_deg = step * (180.0 / int(angle_steps))
        angle = math.radians(angle_deg)
        cosA, sinA = math.cos(angle), math.sin(angle)
        axis_1 = cosA * axis_1_0 - sinA * axis_2_0
        axis_2 = sinA * axis_1_0 + cosA * axis_2_0
        R = np.column_stack([axis_1, axis_2, axis_3])
        local = centered @ R
        half_extents = np.percentile(np.abs(local), float(extent_percentile), axis=0).astype(np.float64)
        half_extents = half_extents * (1.0 + float(padding))
        half_extents = np.maximum(half_extents, 1e-4)
        corners_local = corner_signs * half_extents[None, :]
        corners_world = corners_local @ R.T + center[None, :]
        sum_aspect = 0.0
        sum_iou = 0.0
        n_used = 0
        for view in sam_views:
            cam_pts = corners_world @ view["rotation"].T + view["translation"][None, :]
            if not np.all(cam_pts[:, 2] > 1e-3):
                continue
            u = view["fx"] * cam_pts[:, 0] / cam_pts[:, 2] + view["cx"]
            v = view["fy"] * cam_pts[:, 1] / cam_pts[:, 2] + view["cy"]
            proj_w = float(u.max() - u.min())
            proj_h = float(v.max() - v.min())
            sam_w = float(view["sam_bbox"][2] - view["sam_bbox"][0])
            sam_h = float(view["sam_bbox"][3] - view["sam_bbox"][1])
            if proj_h < 1 or sam_h < 1 or proj_w < 1 or sam_w < 1:
                continue
            log_aspect_diff = abs(math.log(proj_w / proj_h) - math.log(sam_w / sam_h))
            aspect_score = math.exp(-log_aspect_diff)
            sum_aspect += aspect_score
            sum_iou += bbox_iou([float(u.min()), float(v.min()), float(u.max()), float(v.max())], view["sam_bbox"])
            n_used += 1
        if n_used == 0:
            continue
        avg_aspect = sum_aspect / n_used
        avg_iou = sum_iou / n_used
        score = aspect_weight * avg_aspect + iou_weight * avg_iou
        if score > best_score:
            best_score = score
            best_angle = angle_deg
            best_aspect = avg_aspect
            best_iou = avg_iou
            best_half_extents = half_extents
    if best_score < 0:
        return obb_info
    angle = math.radians(best_angle)
    cosA, sinA = math.cos(angle), math.sin(angle)
    axis_1 = cosA * axis_1_0 - sinA * axis_2_0
    axis_2 = sinA * axis_1_0 + cosA * axis_2_0
    R_final = np.column_stack([axis_1, axis_2, axis_3])
    obb_info = dict(obb_info)
    obb_info["rotation"] = R_final.tolist()
    obb_info["half_extents"] = best_half_extents.tolist()
    obb_info["sam_align_angle_deg"] = float(best_angle)
    obb_info["sam_align_aspect_score"] = float(best_aspect)
    obb_info["sam_align_iou"] = float(best_iou)
    return obb_info


def collect_sam_views_for_object(args, prompts, cameras=None):
    from vote_gaussian_masks import load_mask_records as load_mask_records_full
    if cameras is None:
        cameras = load_colmap_model(args.colmap_model)
    prompts = set(str(p) for p in prompts if p)
    views = []
    for image_name, camera, rotation, translation in cameras:
        path = mask_path(args.mask_dir, image_name)
        if not path.exists():
            continue
        records = load_mask_records_full(path)
        chosen_ids = []
        for local_id, rec in records.items():
            if not isinstance(rec, dict):
                continue
            rec_prompts = set()
            if rec.get("prompt"):
                rec_prompts.add(str(rec["prompt"]))
            for p in rec.get("source_prompts", []) or []:
                if p:
                    rec_prompts.add(str(p))
            if prompts and not (rec_prompts & prompts):
                continue
            chosen_ids.append(int(local_id))
        if not chosen_ids:
            continue
        w, h, fx, fy, cx, cy = camera_params(camera)
        sam_instance = load_instance_mask(path, args.mask_key)
        if sam_instance.shape != (h, w):
            import cv2

            sam_instance = cv2.resize(sam_instance.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        sam_mask = np.isin(sam_instance, np.asarray(chosen_ids, dtype=sam_instance.dtype))
        sam_pixels = np.flatnonzero(sam_mask.reshape(-1)).astype(np.int64)
        if len(sam_pixels) == 0:
            continue
        ys = sam_pixels // int(w)
        xs = sam_pixels - ys * int(w)
        views.append({
            "image_name": image_name,
            "rotation": np.asarray(rotation, dtype=np.float64),
            "translation": np.asarray(translation, dtype=np.float64),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "width": int(w), "height": int(h),
            "sam_bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            "sam_pixels": sam_pixels,
        })
    return views


def detect_world_up_from_cameras(cameras):
    if not cameras:
        return None
    ups = []
    for image_name, camera, R, T in cameras:
        # COLMAP camera convention: X right, Y down, Z forward.
        # World-up in world coords (assuming camera held upright) = R^T @ [0, -1, 0]
        world_up = R.T @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        n = float(np.linalg.norm(world_up))
        if n > 1e-6:
            ups.append(world_up / n)
    if not ups:
        return None
    ups = np.asarray(ups)
    mean_up = ups.mean(axis=0)
    n = float(np.linalg.norm(mean_up))
    if n < 1e-6:
        return None
    return mean_up / n


def detect_ground_normal(xyz, max_iters=300, sample_size=3, seed=42):
    n = len(xyz)
    if n < sample_size:
        return None
    bbox = xyz.max(axis=0) - xyz.min(axis=0)
    scene_scale = float(np.linalg.norm(bbox))
    threshold = max(scene_scale * 0.005, 0.01)
    rng = np.random.default_rng(int(seed))
    best_inliers = 0
    best_normal = None
    for _ in range(int(max_iters)):
        sample = rng.choice(n, sample_size, replace=False)
        p1, p2, p3 = xyz[sample[0]], xyz[sample[1]], xyz[sample[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        nn = float(np.linalg.norm(normal))
        if nn < 1e-6:
            continue
        normal = normal / nn
        d = -float(normal @ p1)
        dists = np.abs(xyz @ normal + d)
        inliers = int(np.sum(dists < threshold))
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
    if best_normal is None:
        return None
    centroid = xyz.mean(axis=0)
    above = float(np.sum((xyz - centroid) @ best_normal > 0))
    if above < n / 2:
        best_normal = -best_normal
    return best_normal


def getattr_density_pct(value):
    # Negative value encodes "use this percentile". E.g., -98.0 → use 98th percentile.
    pct = abs(float(value))
    if pct <= 1.0:
        pct = pct * 100.0
    return max(50.0, min(100.0, pct))


def find_optimal_density_ratio_per_axis(local_pts, ratios=None, n_bins=40):
    if ratios is None:
        ratios = np.linspace(0.05, 0.50, 19)
    n_axes = local_pts.shape[1]
    optimal = np.zeros(n_axes, dtype=np.float64)
    for axis in range(n_axes):
        vals_1d = local_pts[:, axis:axis + 1]
        extents_at_ratio = []
        for r in ratios:
            he = density_edge_extent(vals_1d, n_bins=n_bins, min_density_ratio=float(r))
            extents_at_ratio.append(he[0])
        extents_at_ratio = np.asarray(extents_at_ratio, dtype=np.float64)
        # Kneedle: point on the (ratio, extent) curve farthest from line connecting first and last
        if len(extents_at_ratio) < 4 or extents_at_ratio.max() - extents_at_ratio.min() < 1e-6:
            optimal[axis] = float(ratios[len(ratios) // 2])
            continue
        norm_r = (ratios - ratios[0]) / max(ratios[-1] - ratios[0], 1e-6)
        norm_e = (extents_at_ratio - extents_at_ratio.min()) / max(extents_at_ratio.max() - extents_at_ratio.min(), 1e-6)
        p_start = np.array([norm_r[0], norm_e[0]])
        p_end = np.array([norm_r[-1], norm_e[-1]])
        line = p_end - p_start
        line_norm = line / max(float(np.linalg.norm(line)), 1e-6)
        max_d = -1.0
        best_idx = len(ratios) // 2
        for i in range(len(ratios)):
            p = np.array([norm_r[i], norm_e[i]])
            v = p - p_start
            proj = float(np.dot(v, line_norm))
            proj_pt = p_start + proj * line_norm
            d = float(np.linalg.norm(p - proj_pt))
            if d > max_d:
                max_d = d
                best_idx = i
        optimal[axis] = float(ratios[best_idx])
    return optimal


def density_edge_extent(local_pts, n_bins=40, min_density_ratio=0.10, min_consecutive_empty=2):
    half_extents = np.zeros(local_pts.shape[1], dtype=np.float64)
    for axis in range(local_pts.shape[1]):
        vals = local_pts[:, axis]
        edge_distance = 0.0
        for side in (1, -1):
            side_vals = vals[(vals * side) > 0] * side
            if len(side_vals) < 4:
                continue
            max_val = float(side_vals.max())
            if max_val <= 0:
                continue
            hist, edges = np.histogram(side_vals, bins=n_bins, range=(0.0, max_val))
            peak = float(hist.max())
            if peak <= 0:
                continue
            threshold = peak * float(min_density_ratio)
            consecutive_empty = 0
            edge_bin = n_bins
            for i in range(n_bins - 1, -1, -1):
                if hist[i] >= threshold:
                    edge_bin = i + 1
                    break
                consecutive_empty += 1
                if consecutive_empty >= int(min_consecutive_empty):
                    edge_bin = i
            side_edge = float(edges[edge_bin]) if edge_bin <= n_bins else max_val
            edge_distance = max(edge_distance, side_edge)
        half_extents[axis] = edge_distance if edge_distance > 0 else float(np.abs(vals).max())
    return half_extents


def compute_sam_consistency_ratio(pts, sam_views):
    n = len(pts)
    if n == 0 or not sam_views:
        return np.ones(n, dtype=np.float64)
    n_inside = np.zeros(n, dtype=np.int64)
    n_visible = np.zeros(n, dtype=np.int64)
    for view in sam_views:
        cam_pts = pts @ view["rotation"].T + view["translation"][None, :]
        valid = cam_pts[:, 2] > 1e-3
        u = view["fx"] * cam_pts[:, 0] / np.maximum(cam_pts[:, 2], 1e-6) + view["cx"]
        v = view["fy"] * cam_pts[:, 1] / np.maximum(cam_pts[:, 2], 1e-6) + view["cy"]
        in_image = valid & (u >= 0) & (u < view["width"]) & (v >= 0) & (v < view["height"])
        n_visible[in_image] += 1
        sam_pixels = view.get("sam_pixels")
        if sam_pixels is not None:
            inside = np.zeros(n, dtype=bool)
            visible_idx = np.flatnonzero(in_image)
            if len(visible_idx):
                ui = np.clip(np.rint(u[visible_idx]).astype(np.int64), 0, int(view["width"]) - 1)
                vi = np.clip(np.rint(v[visible_idx]).astype(np.int64), 0, int(view["height"]) - 1)
                lin = vi * int(view["width"]) + ui
                inside[visible_idx] = np.isin(lin, sam_pixels, assume_unique=False)
        else:
            sam = view["sam_bbox"]
            inside = in_image & (u >= sam[0]) & (u <= sam[2]) & (v >= sam[1]) & (v <= sam[3])
        n_inside[inside] += 1
    ratio = np.where(n_visible > 0, n_inside / np.maximum(n_visible, 1), 0.0)
    return ratio


def obb_keep_mask_constrained(pts, world_up, sam_views=None, sam_min_visible=3, sam_min_consistency=0.5, dense_fraction=0.7, padding=0.10, extent_percentile=95.0, k_density=8, density_edge_ratio=0.25):
    n = len(pts)
    if n < max(20, k_density + 2):
        return np.ones(n, dtype=bool), None
    world_up = np.asarray(world_up, dtype=np.float64)
    world_up = world_up / max(float(np.linalg.norm(world_up)), 1e-6)

    tree = cKDTree(pts)
    nn_dists, _ = tree.query(pts, k=k_density + 1)
    last_nn = nn_dists[:, -1]
    last_nn = np.where(np.isfinite(last_nn), last_nn, last_nn.max())
    density = 1.0 / np.maximum(last_nn, 1e-6)
    threshold = float(np.percentile(density, max(0.0, min(100.0, (1.0 - float(dense_fraction)) * 100.0))))
    core_mask = density >= threshold
    if int(core_mask.sum()) < 10:
        core_mask = np.ones(n, dtype=bool)
    core_pts = pts[core_mask]

    center = core_pts.mean(axis=0)
    centered_core = core_pts - center
    vert = centered_core @ world_up
    horizontal_core = centered_core - np.outer(vert, world_up)
    if len(horizontal_core) >= 3:
        cov_h = np.cov(horizontal_core.T)
        if np.all(np.isfinite(cov_h)):
            eigvals, eigvecs = np.linalg.eigh(cov_h)
            order = np.argsort(eigvals)[::-1]
            axis_1 = eigvecs[:, order[0]]
        else:
            axis_1 = np.array([1.0, 0.0, 0.0])
    else:
        axis_1 = np.array([1.0, 0.0, 0.0])
    axis_1 = axis_1 - (axis_1 @ world_up) * world_up
    if float(np.linalg.norm(axis_1)) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(fallback @ world_up)) > 0.95:
            fallback = np.array([0.0, 1.0, 0.0])
        axis_1 = fallback - (fallback @ world_up) * world_up
    axis_1 = axis_1 / max(float(np.linalg.norm(axis_1)), 1e-6)
    axis_3 = world_up
    axis_2 = np.cross(axis_3, axis_1)
    axis_2 = axis_2 / max(float(np.linalg.norm(axis_2)), 1e-6)
    R = np.column_stack([axis_1, axis_2, axis_3])

    extent_source = "dense_core"
    sam_keep_count = 0
    extent_pts = core_pts
    if sam_views is not None and len(sam_views) >= int(sam_min_visible):
        consistency = compute_sam_consistency_ratio(pts, sam_views)
        sam_keep = consistency >= float(sam_min_consistency)
        sam_keep_count = int(sam_keep.sum())
        if sam_keep_count >= max(20, n // 10):
            extent_pts = pts[sam_keep]
            extent_source = "sam_consistency"
    extent_centered = extent_pts - center
    local_extent = extent_centered @ R
    if extent_source == "sam_consistency":
        if float(density_edge_ratio) < 0:
            # Negative ratio = disable density-edge entirely; use percentile of SAM-consistent splats
            pct = float(getattr_density_pct(density_edge_ratio))
            half_extents = np.percentile(np.abs(local_extent), pct, axis=0).astype(np.float64)
        elif float(density_edge_ratio) == 0:
            # Auto per-axis: kneedle on (density_ratio, extent) curve
            optimal_ratios = find_optimal_density_ratio_per_axis(local_extent, n_bins=40)
            half_extents = np.zeros(local_extent.shape[1], dtype=np.float64)
            for axis in range(local_extent.shape[1]):
                vals_1d = local_extent[:, axis:axis + 1]
                half_extents[axis] = density_edge_extent(vals_1d, n_bins=40, min_density_ratio=float(optimal_ratios[axis]))[0]
        else:
            half_extents = density_edge_extent(local_extent, n_bins=40, min_density_ratio=float(density_edge_ratio))
    else:
        half_extents = np.percentile(np.abs(local_extent), float(extent_percentile), axis=0).astype(np.float64)
    half_extents = half_extents * (1.0 + float(padding))
    half_extents = np.maximum(half_extents, 1e-4)

    all_local = (pts - center) @ R
    keep = np.all(np.abs(all_local) <= half_extents, axis=1)
    return keep, {
        "center": center.tolist(),
        "rotation": R.tolist(),
        "half_extents": half_extents.tolist(),
        "core_size": int(core_mask.sum()),
        "extent_percentile": float(extent_percentile),
        "axis_aligned_to": "world_up",
        "extent_source": extent_source,
        "sam_keep_count": int(sam_keep_count),
    }


def obb_keep_mask(pts, dense_fraction=0.5, padding=0.15, extent_percentile=97.5, k_density=8):
    n = len(pts)
    if n < max(20, k_density + 2):
        return np.ones(n, dtype=bool), None
    tree = cKDTree(pts)
    nn_dists, _ = tree.query(pts, k=k_density + 1)
    last_nn = nn_dists[:, -1]
    last_nn = np.where(np.isfinite(last_nn), last_nn, last_nn.max())
    density = 1.0 / np.maximum(last_nn, 1e-6)
    threshold = float(np.percentile(density, max(0.0, min(100.0, (1.0 - float(dense_fraction)) * 100.0))))
    core_mask = density >= threshold
    if int(core_mask.sum()) < 10:
        core_mask = np.ones(n, dtype=bool)
    core_pts = pts[core_mask]
    center = core_pts.mean(axis=0)
    centered = core_pts - center
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return np.ones(n, dtype=bool), None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    all_local = (pts - center) @ R
    half_extents = np.percentile(np.abs(all_local), float(extent_percentile), axis=0).astype(np.float64)
    half_extents = half_extents * (1.0 + float(padding))
    half_extents = np.maximum(half_extents, 1e-4)
    keep = np.all(np.abs(all_local) <= half_extents, axis=1)
    return keep, {
        "center": center.tolist(),
        "rotation": R.tolist(),
        "half_extents": half_extents.tolist(),
        "core_size": int(core_mask.sum()),
        "extent_percentile": float(extent_percentile),
    }


def enforce_spatial_coherence(labels, xyz, seed_stats, args, world_up=None, sam_views_by_oid=None):
    if not bool(getattr(args, "cleanup_components", True)):
        return labels, {}
    radius_mult = float(getattr(args, "cleanup_component_radius_multiplier", 3.0))
    min_ratio = float(getattr(args, "cleanup_min_component_ratio", 0.10))
    max_dist_ratio = float(getattr(args, "cleanup_max_distance_ratio", 1.5))
    min_splats = max(int(getattr(args, "min_object_splats", 32)), 8)
    min_comp_size = max(int(getattr(args, "cleanup_min_component_splats", 8)), 1)
    obb_enabled = bool(getattr(args, "cleanup_obb_filter", False))
    obb_dense_fraction = float(getattr(args, "cleanup_obb_dense_fraction", 0.5))
    obb_padding = float(getattr(args, "cleanup_obb_padding", 0.15))
    obb_extent_percentile = float(getattr(args, "cleanup_obb_extent_percentile", 97.5))
    obb_constrain_vertical = bool(getattr(args, "cleanup_obb_constrain_vertical", True)) and world_up is not None
    cleaned = labels.copy()
    diagnostics = {}
    for oid in sorted(int(v) for v in np.unique(labels) if v > 0):
        idx = np.flatnonzero(labels == oid)
        if len(idx) < min_splats:
            continue
        pts = xyz[idx]
        seed = seed_stats.get(int(oid))
        dropped_radius = 0
        if seed is not None and max_dist_ratio > 0:
            seed_center = seed["center"]
            seed_radius = max(float(seed["radius"]), 1e-4)
            dists = np.linalg.norm(pts - seed_center[None, :], axis=1)
            inside = dists <= seed_radius * max_dist_ratio
            dropped_radius = int(np.count_nonzero(~inside))
            cleaned[idx[~inside]] = 0
            idx = idx[inside]
            pts = pts[inside]
        if len(idx) < min_splats:
            cleaned[idx] = 0
            diagnostics[int(oid)] = {
                "dropped_radius": dropped_radius,
                "dropped_components": int(len(idx)),
                "kept": 0,
                "num_components_total": 0,
                "num_components_kept": 0,
            }
            continue

        dropped_obb = 0
        obb_info = None
        if obb_enabled and len(pts) >= 20:
            if obb_constrain_vertical:
                sam_views_for_obj = sam_views_by_oid.get(int(oid)) if sam_views_by_oid else None
                obb_keep, obb_info = obb_keep_mask_constrained(
                    pts, world_up,
                    sam_views=sam_views_for_obj,
                    dense_fraction=obb_dense_fraction,
                    padding=obb_padding,
                    extent_percentile=obb_extent_percentile,
                    density_edge_ratio=float(getattr(args, "cleanup_obb_density_edge_ratio", 0.25)),
                )
            else:
                obb_keep, obb_info = obb_keep_mask(pts, dense_fraction=obb_dense_fraction, padding=obb_padding, extent_percentile=obb_extent_percentile)
            if int(obb_keep.sum()) >= min_splats:
                dropped_obb = int(np.count_nonzero(~obb_keep))
                cleaned[idx[~obb_keep]] = 0
                idx = idx[obb_keep]
                pts = pts[obb_keep]

        tree = cKDTree(pts)
        if len(pts) > 1:
            nn_dists, _ = tree.query(pts, k=2)
            local_nn = nn_dists[:, 1]
            local_nn = local_nn[np.isfinite(local_nn) & (local_nn > 0)]
            eps = float(np.median(local_nn) * radius_mult) if len(local_nn) else 0.01
        else:
            eps = 0.01
        eps = max(eps, 1e-5)

        comp = np.full(len(idx), -1, dtype=np.int32)
        comp_id = 0
        for start in range(len(idx)):
            if comp[start] >= 0:
                continue
            stack = [start]
            comp[start] = comp_id
            while stack:
                node = stack.pop()
                for nbr in tree.query_ball_point(pts[node], eps):
                    if comp[nbr] < 0:
                        comp[nbr] = comp_id
                        stack.append(nbr)
            comp_id += 1
        counts = np.bincount(comp)
        if len(counts) == 0:
            continue
        max_count = int(counts.max())
        threshold = max(int(math.ceil(max_count * min_ratio)), min_comp_size)
        keep_components = np.flatnonzero(counts >= threshold)
        if len(keep_components) == 0:
            keep_components = np.asarray([int(np.argmax(counts))], dtype=np.int64)
        keep_mask = np.isin(comp, keep_components)
        cleaned[idx[~keep_mask]] = 0
        diagnostics[int(oid)] = {
            "dropped_radius": int(dropped_radius),
            "dropped_obb": int(dropped_obb),
            "dropped_components": int(np.count_nonzero(~keep_mask)),
            "kept": int(np.count_nonzero(keep_mask)),
            "num_components_total": int(len(counts)),
            "num_components_kept": int(len(keep_components)),
            "largest_component": int(max_count),
            "component_eps": float(eps),
            "component_threshold": int(threshold),
            "obb_info": obb_info,
        }
    return cleaned, diagnostics


def infer_labels(logits, positive, labels, candidate_indices, object_ids, args):
    best_rows = np.argmax(logits, axis=0)
    best_logits = logits[best_rows, np.arange(logits.shape[1])]
    masked = logits.copy()
    masked[best_rows, np.arange(logits.shape[1])] = -np.inf
    second_logits = np.max(masked, axis=0)
    margins = best_logits - second_logits
    best_positive = positive[best_rows, np.arange(positive.shape[1])]
    best_ids = np.asarray([object_ids[row] for row in best_rows], dtype=np.int32)
    seed_same = labels[candidate_indices] == best_ids
    keep = (
        (best_positive >= int(args.min_positive_observations)) | seed_same
    ) & (best_logits >= float(args.min_logit)) & (
        (margins >= float(args.min_logit_margin)) | seed_same
    )

    refined = np.zeros_like(labels)
    refined[candidate_indices[keep]] = best_ids[keep]

    min_splats = int(args.min_object_splats)
    for oid in list(np.unique(refined[refined > 0])):
        if int(np.count_nonzero(refined == oid)) < min_splats:
            refined[refined == oid] = 0
    return refined, best_ids, best_logits, margins, best_positive, keep


def write_summary(out_dir, labels0, labels, positive, candidate_indices, object_ids, objects, confidence, margins):
    stats = []
    for row, oid in enumerate(object_ids):
        before = int(np.count_nonzero(labels0 == oid))
        after = int(np.count_nonzero(labels == oid))
        owned = np.flatnonzero(labels[candidate_indices] == oid)
        added = int(np.count_nonzero((labels[candidate_indices] == oid) & (labels0[candidate_indices] != oid)))
        dropped = int(np.count_nonzero((labels0 == oid) & (labels != oid)))
        mean_conf = float(np.mean(confidence[owned])) if len(owned) else 0.0
        mean_margin = float(np.mean(margins[owned])) if len(owned) else 0.0
        stats.append({
            "id": int(oid),
            "source_prompts": objects.get(oid, {}).get("source_prompts", []),
            "seed_splats": before,
            "owned_splats": after,
            "added_splats": added,
            "dropped_splats": dropped,
            "assigned_mask_observations": int(np.count_nonzero(positive[row] > 0)),
            "mean_confidence": mean_conf,
            "mean_margin": mean_margin,
        })
    csv_path = Path(out_dir) / "ownership_diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()) if stats else ["id"])
        writer.writeheader()
        writer.writerows(stats)
    return stats


def write_camera_records(path, cameras):
    records = []
    for image_name, camera, R, T in cameras:
        try:
            w, h, fx, fy, cx, cy = camera_params(camera)
        except Exception:
            continue
        # Camera center in world: -R^T @ T
        position = (-R.T @ T).astype(float).tolist()
        records.append({
            "image_name": image_name,
            "rotation": [[float(R[i][j]) for j in range(3)] for i in range(3)],
            "translation": [float(T[i]) for i in range(3)],
            "position": position,
            "width": int(w),
            "height": int(h),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        })
    with open(path, "w") as f:
        json.dump({"cameras": records}, f)
    return len(records)


def write_tracks_json(path, xyz, labels, objects, diagnostics, summary_extra, cleanup_diagnostics=None):
    diagnostics_by_id = {int(item["id"]): item for item in diagnostics}
    cleanup_diagnostics = cleanup_diagnostics or {}
    records = []
    for oid in sorted(int(v) for v in np.unique(labels) if v > 0):
        pts = xyz[labels == oid]
        center, axes, rotation = pca_geometry(pts)
        prompts = objects.get(oid, {}).get("source_prompts", [])
        diag = diagnostics_by_id.get(oid, {})
        cleanup_info = cleanup_diagnostics.get(int(oid), {}) or {}
        obb_info = cleanup_info.get("obb_info") if isinstance(cleanup_info, dict) else None
        records.append({
            "id": int(oid),
            "center": center.astype(float).tolist(),
            "axes": axes.astype(float).tolist(),
            "rotation": rotation.astype(float).tolist(),
            "shape_type": "ownership_pca",
            "source_prompts": prompts,
            "num_splats": int(len(pts)),
            "ownership_mean_confidence": float(diag.get("mean_confidence", 0.0)),
            "ownership_mean_margin": float(diag.get("mean_margin", 0.0)),
            "ownership_added_splats": int(diag.get("added_splats", 0)),
            "ownership_dropped_splats": int(diag.get("dropped_splats", 0)),
            "obb": obb_info,
        })
    data = dict(summary_extra)
    data["objects"] = records
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(args.gaussians)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    rgb = gaussian_rgb_from_vertex(vertex)
    track_data = np.load(args.tracks_npz, allow_pickle=True)
    labels0 = np.asarray(track_data["labels"], dtype=np.int32)
    votes = np.load(args.votes, allow_pickle=True)
    selected_raw = np.asarray(votes["selected_raw"], dtype=bool) if "selected_raw" in votes else np.asarray(track_data["selected"], dtype=bool)
    selected = np.asarray(votes["selected"], dtype=bool) if "selected" in votes else np.asarray(track_data["selected"], dtype=bool)
    score = (
        np.asarray(votes["score"], dtype=np.float32)
        if "score" in votes
        else np.asarray(track_data["score"], dtype=np.float32)
        if "score" in track_data
        else np.zeros(len(xyz), dtype=np.float32)
    )
    prompt_owner, prompt_names = load_prompt_owner(votes, len(xyz))

    objects = load_object_metadata(args.tracks_json, labels0)
    object_ids = sorted(oid for oid in objects if np.count_nonzero(labels0 == oid) >= int(args.min_object_splats))
    if not object_ids:
        raise RuntimeError("No tracked object proposals available for ownership inference.")
    object_prompts = {oid: set(objects[oid].get("source_prompts", [])) for oid in object_ids}

    seed_stats = object_geometry_stats(xyz, labels0)
    vote_candidate = (selected_raw | selected) & (score >= float(args.min_candidate_score))
    candidate_mask = vote_candidate | (labels0 > 0)
    if bool(args.include_nearby_scene_candidates):
        candidate_mask |= nearby_candidate_mask(xyz, seed_stats, args.candidate_radius_multiplier)
    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_index = np.full(len(xyz), -1, dtype=np.int64)
    candidate_index[candidate_indices] = np.arange(len(candidate_indices), dtype=np.int64)

    observations = collect_full_mask_observations(
        args,
        xyz,
        candidate_mask,
        prompt_owner,
        prompt_names,
    )
    positive, assigned_observations, skipped_ambiguous = build_mask_evidence(
        observations,
        labels0,
        object_ids,
        candidate_index,
        object_prompts,
        args,
    )

    saturation = max(float(args.observation_saturation), 1e-6)
    logits = float(args.mask_weight) * (1.0 - np.exp(-positive.astype(np.float32) / saturation))
    add_seed_prior(logits, labels0, candidate_indices, object_ids, args.seed_weight)
    add_geometry_prior(logits, xyz, candidate_indices, object_ids, seed_stats, args)
    add_prompt_prior(logits, candidate_indices, object_ids, object_prompts, prompt_owner, prompt_names, args)
    logits = smooth_logits(logits, xyz, rgb, candidate_indices, args)

    labels, best_ids, confidence, margins, best_positive, keep = infer_labels(
        logits,
        positive,
        labels0,
        candidate_indices,
        object_ids,
        args,
    )
    labels_full = labels.copy()
    world_up = None
    world_up_source = None
    use_camera_up = bool(getattr(args, "use_camera_world_up", True))
    cameras_loaded = None
    if use_camera_up:
        cameras_loaded = load_colmap_model(args.colmap_model)
        world_up = detect_world_up_from_cameras(cameras_loaded)
        if world_up is not None:
            world_up_source = "cameras"
            print(f"[INFO] Camera-based world up: [{world_up[0]:.3f}, {world_up[1]:.3f}, {world_up[2]:.3f}]")
    if world_up is None and (bool(getattr(args, "cleanup_obb_constrain_vertical", True)) or bool(getattr(args, "save_world_up", True))):
        world_up = detect_ground_normal(xyz)
        if world_up is not None:
            world_up_source = "ground_plane"
            print(f"[INFO] Ground-plane world up: [{world_up[0]:.3f}, {world_up[1]:.3f}, {world_up[2]:.3f}]")
    sam_views_by_oid = None
    if bool(getattr(args, "cleanup_obb_filter", False)) and world_up is not None and bool(getattr(args, "cleanup_obb_use_sam_extent", True)):
        if cameras_loaded is None:
            cameras_loaded = load_colmap_model(args.colmap_model)
        sam_views_by_oid = {}
        for oid_check in sorted(int(v) for v in np.unique(labels_full) if v > 0):
            prompts_check = object_prompts.get(int(oid_check), set())
            sam_views_by_oid[int(oid_check)] = collect_sam_views_for_object(args, prompts_check, cameras=cameras_loaded)
    labels, cleanup_diagnostics = enforce_spatial_coherence(labels, xyz, seed_stats, args, world_up=world_up, sam_views_by_oid=sam_views_by_oid)

    if bool(getattr(args, "cleanup_obb_sam_align", False)):
        cameras = cameras_loaded if cameras_loaded is not None else load_colmap_model(args.colmap_model)
        for oid, info in cleanup_diagnostics.items():
            obb = info.get("obb_info") if isinstance(info, dict) else None
            if not obb:
                continue
            obj_pts = xyz[labels == int(oid)]
            if len(obj_pts) < 20:
                continue
            prompts = object_prompts.get(int(oid), set())
            sam_views = collect_sam_views_for_object(args, prompts, cameras=cameras)
            if len(sam_views) < 3:
                continue
            refined_obb = refine_obb_rotation_with_sam(obb, obj_pts, sam_views)
            info["obb_info"] = refined_obb
            if "sam_align_angle_deg" in refined_obb:
                he = refined_obb.get("half_extents", [0,0,0])
                print(f"[INFO] obj {oid}: SAM-aligned OBB by {refined_obb['sam_align_angle_deg']:.1f}deg (aspect={refined_obb.get('sam_align_aspect_score', 0):.3f}, IoU={refined_obb['sam_align_iou']:.3f}, he=({he[0]:.2f},{he[1]:.2f},{he[2]:.2f}))")
    diagnostics = write_summary(
        out_dir,
        labels0,
        labels,
        positive,
        candidate_indices,
        object_ids,
        objects,
        confidence,
        margins,
    )
    confidence_full = np.zeros(len(xyz), dtype=np.float32)
    margin_full = np.zeros(len(xyz), dtype=np.float32)
    positive_full = np.zeros(len(xyz), dtype=np.uint16)
    confidence_full[candidate_indices] = confidence.astype(np.float32)
    margin_full[candidate_indices] = margins.astype(np.float32)
    positive_full[candidate_indices] = best_positive.astype(np.uint16)

    tracked_scene = colorize_gaussian_tracks(vertex, labels, selected_only=False, opacity=args.foreground_opacity)
    tracked_objects = colorize_gaussian_tracks(vertex, labels, selected_only=True, opacity=args.foreground_opacity)
    write_ply(out_dir / "ownership_scene_gaussians.ply", ply, tracked_scene)
    write_ply(out_dir / "ownership_object_gaussians.ply", ply, tracked_objects)
    np.savez_compressed(
        out_dir / "gaussian_tracks.npz",
        labels=labels,
        labels_full=labels_full,
        seed_labels=labels0,
        selected=labels > 0,
        selected_raw=selected_raw,
        score=score,
        xyz=xyz.astype(np.float32),
        ownership_confidence=confidence_full,
        ownership_margin=margin_full,
        ownership_positive_observations=positive_full,
        candidate_indices=candidate_indices.astype(np.int64),
        candidate_best_object=best_ids.astype(np.int32),
        candidate_best_logit=confidence.astype(np.float32),
        candidate_margin=margins.astype(np.float32),
        candidate_positive_observations=best_positive.astype(np.uint16),
        candidate_kept=keep.astype(bool),
    )
    summary = {
        "num_gaussians": int(len(xyz)),
        "num_candidates": int(len(candidate_indices)),
        "num_seed_objects": int(len(object_ids)),
        "num_objects": int(np.unique(labels[labels > 0]).size),
        "num_seed_splats": int(np.count_nonzero(labels0 > 0)),
        "num_owned_splats": int(np.count_nonzero(labels > 0)),
        "num_added_splats": int(np.count_nonzero((labels > 0) & (labels0 != labels))),
        "num_dropped_seed_splats": int(np.count_nonzero((labels0 > 0) & (labels == 0))),
        "num_observations": int(len(observations)),
        "skipped_ambiguous_observations": int(skipped_ambiguous),
        "assigned_observations": {str(k): int(v) for k, v in assigned_observations.items()},
        "spatial_cleanup": {str(k): v for k, v in cleanup_diagnostics.items()},
        "world_up": world_up.tolist() if world_up is not None else None,
        "world_up_source": world_up_source,
        "diagnostics_csv": str(out_dir / "ownership_diagnostics.csv"),
    }
    if cameras_loaded is None:
        cameras_loaded = load_colmap_model(args.colmap_model)
    write_camera_records(out_dir / "cameras.json", cameras_loaded)
    write_tracks_json(out_dir / "tracks.json", xyz, labels, objects, diagnostics, summary, cleanup_diagnostics)
    print(json.dumps(summary, indent=2))
    print(f"[DONE] Wrote segmented Gaussian ownership to {out_dir}")


if __name__ == "__main__":
    main()

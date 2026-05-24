"""Cluster foreground Gaussian splats into per-object IDs.

The algorithm has two phases:

  1. Conservative spatial DBSCAN on selected (foreground) splats produces many
     small *fragments*. The eps is derived from the median nearest-neighbour
     distance, so it adapts to scene scale.
  2. Fragments are merged into objects when they are (a) physically touching
     within a small bridge gap AND (b) consistently observed in the same SAM
     mask across frames AND (c) rarely observed in different SAM masks.

Why hybrid:
  - Pure 3D clustering over-splits sparse coverage and over-merges touching
    objects.
  - Pure SAM-evidence clustering fails when SAM groups multiple small objects
    into a single mask (a known SAM3 failure mode).
  - The hybrid uses DBSCAN to enforce physical separation (so SAM merging two
    nearby oranges into one mask cannot fuse them) and uses SAM evidence only
    to *bridge thin density gaps* (so a single object with sparse coverage
    isn't shattered into multiple boxes).

Inputs (from vote_gaussian_masks.py):
  votes.npz -- hits/seen/score/selected/selected_raw plus per-frame
               observations obs_splat_idx / obs_mask_id / obs_frame_offsets.

Outputs (same schema as before so the viewer + sphere fitter don't change):
  gaussian_tracks.npz   labels (int32, 0=background), xyz, selected*, score
  tracks.json           {objects: [ {id, center, axes, rotation, obb, ...} ]}
  tracked_object_gaussians.ply  per-object-colored foreground splats
  tracked_scene_gaussians.ply   scene with foreground recolored per object
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config


SH_C0 = 0.28209479177387814


# --------------------------------------------------------------- args

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster foreground splats into object IDs.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--gaussians", type=str, default=None)
    parser.add_argument("--votes", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    return namespace_from_config(
        cli.config,
        "tracking",
        {
            "gaussians": cli.gaussians,
            "votes": cli.votes,
            "mask_dir": cli.mask_dir,
            "out_dir": cli.out_dir,
            "overwrite": cli.overwrite,
        },
    )


# --------------------------------------------------------------- helpers

def _logit(x):
    x = np.clip(x, 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    return np.asarray([r, g, b], dtype=np.float32)


def _color_for(oid):
    hue = (int(oid) * 0.618033988749895) % 1.0
    return _hsv_to_rgb(hue, 0.72, 0.98)


def _gaussian_max_scale(vertex):
    names = vertex.dtype.names or ()
    scale_fields = [n for n in names if n.startswith("scale_")]
    if not scale_fields:
        return np.zeros(len(vertex), dtype=np.float32)
    return np.exp(np.max(
        np.column_stack([np.asarray(vertex[n], dtype=np.float32) for n in scale_fields]),
        axis=1,
    ))


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# --------------------------------------------------------------- spatial DBSCAN

def _median_neighbor_distance(xyz, k=6):
    if len(xyz) < 2:
        return 0.0
    k = min(k, len(xyz))
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k)
    nn = d[:, 1:].mean(axis=1)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if len(nn) else 0.0


def _dbscan_components(xyz, eps, min_samples):
    """Flood-fill connected components on the eps-radius graph. Returns
    int32 labels in [1..K]; 0 means noise (component smaller than
    min_samples)."""
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    tree = cKDTree(xyz)
    labels = np.zeros(n, dtype=np.int32)
    cluster = 0
    for seed in range(n):
        if labels[seed] != 0:
            continue
        cluster += 1
        stack = [seed]
        members = []
        while stack:
            idx = stack.pop()
            if labels[idx] != 0:
                continue
            labels[idx] = cluster
            members.append(idx)
            for nb in tree.query_ball_point(xyz[idx], r=eps):
                if labels[nb] == 0:
                    stack.append(nb)
        if len(members) < min_samples:
            for m in members:
                labels[m] = 0
            cluster -= 1
    return labels


# --------------------------------------------------------------- SAM signatures

def _per_fragment_signatures(fragment_labels, obs_splat_idx, obs_mask_id, obs_frame_offsets, majority_fraction):
    """For each (fragment, frame): if a single SAM mask owns
    >= majority_fraction of the fragment's splats seen in that frame, record
    (mask_id, votes, visible_in_frame). Ambiguous frames are omitted."""
    num_frames = len(obs_frame_offsets) - 1
    sigs = defaultdict(dict)
    for f in range(num_frames):
        lo = int(obs_frame_offsets[f])
        hi = int(obs_frame_offsets[f + 1])
        if hi <= lo:
            continue
        splats = obs_splat_idx[lo:hi]
        masks = obs_mask_id[lo:hi]
        frags = fragment_labels[splats]
        keep = frags > 0
        if not keep.any():
            continue
        frags = frags[keep]
        masks = masks[keep]
        order = np.lexsort((masks, frags))
        frags = frags[order]
        masks = masks[order]
        boundary = np.concatenate(([True], frags[1:] != frags[:-1]))
        starts = np.flatnonzero(boundary)
        ends = np.concatenate((starts[1:], [len(frags)]))
        for s, e in zip(starts, ends):
            frag_id = int(frags[s])
            ids, counts = np.unique(masks[s:e], return_counts=True)
            best = int(np.argmax(counts))
            total = int(e - s)
            needed = max(1, int(np.ceil(majority_fraction * total)))
            if int(counts[best]) >= needed:
                sigs[frag_id][f] = (int(ids[best]), int(counts[best]), total)
    return sigs


# --------------------------------------------------------------- merge

def _fragment_info(xyz, labels):
    info = {}
    for fid in np.unique(labels):
        if fid <= 0:
            continue
        pts = xyz[labels == fid]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        radius = float(np.percentile(dist, 90.0)) if len(dist) else 0.0
        info[int(fid)] = {
            "center": center.astype(np.float64),
            "radius": radius,
            "num_splats": int(len(pts)),
            "points": pts.astype(np.float64),
        }
    return info


def _has_density_valley(pts_a, pts_b, center_a, center_b, valley_max_ratio, num_bins):
    """True if projecting pts_a + pts_b onto the A->B axis reveals a density
    dip between the two centroids deep enough to indicate two distinct modes.
    Used to veto a SAM-evidence merge of two visually-touching but physically-
    distinct objects."""
    axis = center_b - center_a
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return False
    axis = axis / norm
    proj = np.concatenate([
        (pts_a - center_a) @ axis,
        (pts_b - center_a) @ axis,
    ])
    # Clip the histogram domain to the centroid-to-centroid span; outside that
    # range, points are "outside" the boundary and unrelated to the valley
    # decision.
    inside = proj[(proj >= 0.0) & (proj <= norm)]
    if len(inside) < max(2 * num_bins, 8):
        return False
    counts, _ = np.histogram(inside, bins=num_bins, range=(0.0, norm))
    third = max(num_bins // 3, 1)
    peak_a = int(counts[:third].max())
    peak_b = int(counts[-third:].max())
    if peak_a <= 0 or peak_b <= 0:
        return False
    middle = counts[third:num_bins - third]
    if len(middle) == 0:
        return False
    valley = int(middle.min())
    return valley < valley_max_ratio * min(peak_a, peak_b)


def _merge_fragments(xyz, labels, signatures, args):
    info = _fragment_info(xyz, labels)
    frag_ids = sorted(info.keys())
    if len(frag_ids) < 2:
        return labels, 0
    pos_of = {fid: i for i, fid in enumerate(frag_ids)}
    centers = np.array([info[fid]["center"] for fid in frag_ids])
    radii = np.array([info[fid]["radius"] for fid in frag_ids])

    bridge_ratio = float(getattr(args, "bridge_gap_ratio", 0.35))
    min_co_obs = int(getattr(args, "min_co_observations", 3))
    min_same_ratio = float(getattr(args, "min_same_mask_ratio", 0.65))
    max_diff_ratio = float(getattr(args, "max_diff_mask_ratio", 0.20))
    valley_max_ratio = float(getattr(args, "valley_max_ratio", 0.45))
    valley_bins = int(getattr(args, "valley_bins", 12))
    valley_check = bool(getattr(args, "valley_check", True))

    # Generous candidate query: bridgeable if center_dist <= r_a + r_b + bridge_ratio * min(r_a, r_b).
    # Upper bound for query = 2*max_radius * (1 + bridge_ratio).
    max_radius = float(radii.max()) if len(radii) else 0.0
    search_r = max(2.0 * max_radius * (1.0 + bridge_ratio), 1e-6)
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=search_r)

    uf = UnionFind(len(frag_ids))
    merges = 0
    for i, j in pairs:
        ri, rj = radii[i], radii[j]
        center_dist = float(np.linalg.norm(centers[i] - centers[j]))
        gap = center_dist - ri - rj
        if gap > bridge_ratio * min(ri, rj):
            continue
        sig_a = signatures.get(frag_ids[i], {})
        sig_b = signatures.get(frag_ids[j], {})
        if not sig_a or not sig_b:
            continue
        co_frames = sig_a.keys() & sig_b.keys()
        if len(co_frames) < min_co_obs:
            continue
        same = sum(1 for f in co_frames if sig_a[f][0] == sig_b[f][0])
        diff = len(co_frames) - same
        if same < min_co_obs:
            continue
        if same / len(co_frames) < min_same_ratio:
            continue
        if diff / len(co_frames) > max_diff_ratio:
            continue
        if valley_check and _has_density_valley(
            info[frag_ids[i]]["points"],
            info[frag_ids[j]]["points"],
            centers[i],
            centers[j],
            valley_max_ratio,
            valley_bins,
        ):
            continue
        if uf.union(i, j):
            merges += 1

    # Relabel by representative.
    new_labels = labels.copy()
    rep_to_new = {}
    next_id = 1
    for fid in frag_ids:
        rep = uf.find(pos_of[fid])
        if rep not in rep_to_new:
            rep_to_new[rep] = next_id
            next_id += 1
        new_id = rep_to_new[rep]
        if new_id != fid:
            new_labels[labels == fid] = new_id
    return new_labels, merges


# --------------------------------------------------------------- tiny absorb

def _absorb_tiny_fragments(xyz, labels, args):
    """Fold tiny fragments into the nearest physically-overlapping larger
    fragment. Catches DBSCAN-noise nubs and bimodal-split misfires that the
    SAM-merge step couldn't rescue (e.g. when the tiny fragment doesn't have
    enough co-observations to build a confident mask signature).

    Unlike the SAM-merge step this skips the same-mask check, since tiny
    fragments rarely have enough observations to vote consistently. Safety
    comes from requiring spatial overlap: gap <= absorb_gap_ratio * combined
    radii. That keeps a genuinely-small isolated object from being eaten by a
    distant neighbour."""
    if not bool(getattr(args, "absorb_tiny_fragments", True)):
        return labels, 0
    tiny_max = int(getattr(args, "tiny_fragment_max_splats", 25))
    absorb_gap_ratio = float(getattr(args, "tiny_absorb_gap_ratio", 0.5))

    info = _fragment_info(xyz, labels)
    if not info:
        return labels, 0
    fids = sorted(info.keys())
    tiny_ids = [fid for fid in fids if info[fid]["num_splats"] <= tiny_max]
    big_ids = [fid for fid in fids if info[fid]["num_splats"] > tiny_max]
    if not tiny_ids or not big_ids:
        return labels, 0

    big_centers = np.asarray([info[fid]["center"] for fid in big_ids])
    big_radii = np.asarray([info[fid]["radius"] for fid in big_ids])
    tree = cKDTree(big_centers)

    new_labels = labels.copy()
    absorbed = 0
    for tid in tiny_ids:
        tc = info[tid]["center"]
        tr = info[tid]["radius"]
        # Search neighbours within distance = max(big_radius) + tiny_radius +
        # margin. Use the global big-radius max so we don't miss a chunky
        # neighbour whose body extends past its centre.
        search_r = float(big_radii.max() + tr + absorb_gap_ratio * (big_radii.max() + tr))
        candidates = tree.query_ball_point(tc, r=search_r)
        best = None
        best_gap = float("inf")
        for k in candidates:
            target_fid = big_ids[k]
            target_r = float(big_radii[k])
            dist = float(np.linalg.norm(tc - big_centers[k]))
            gap = dist - target_r - tr
            if gap > absorb_gap_ratio * (target_r + tr):
                continue
            if gap < best_gap:
                best = target_fid
                best_gap = gap
        if best is not None:
            new_labels[labels == tid] = best
            absorbed += 1
    return new_labels, absorbed


# --------------------------------------------------------------- bimodal split

def _split_bimodal_fragments(xyz, labels, args):
    """For each fragment, project onto its PCA principal axis and split if a
    clear density valley exists between two modes. Recovers two distinct
    objects whose splats DBSCAN merged into one fragment (the case the
    between-fragment valley veto cannot reach)."""
    if not bool(getattr(args, "bimodal_split", True)):
        return labels, 0
    valley_max = float(getattr(args, "bimodal_valley_ratio", 0.4))
    bins = int(getattr(args, "bimodal_bins", 24))
    min_axis_ratio = float(getattr(args, "bimodal_min_axis_ratio", 1.4))
    min_splats = int(getattr(args, "bimodal_min_splats", 24))
    max_iters = int(getattr(args, "bimodal_max_iters", 3))

    new_labels = labels.copy()
    splits = 0
    for _ in range(max_iters):
        changed = False
        for fid in sorted(int(v) for v in np.unique(new_labels) if v > 0):
            idx = np.flatnonzero(new_labels == fid)
            if len(idx) < min_splats:
                continue
            pts = xyz[idx]
            center = pts.mean(axis=0)
            centered = pts - center
            cov = np.cov(centered.T)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            # Sphere-ish clusters (axis_ratio ~ 1) should not be split; only
            # elongated clusters are candidates for two touching objects.
            if vals[1] < 1e-18 or (vals[0] / vals[1]) < (min_axis_ratio ** 2):
                continue
            proj = centered @ vecs[:, 0]
            counts, edges = np.histogram(proj, bins=bins)
            tail = max(bins // 4, 1)
            peak_lo = int(counts[:tail].max())
            peak_hi = int(counts[-tail:].max())
            middle = counts[tail:bins - tail]
            if len(middle) == 0 or peak_lo <= 0 or peak_hi <= 0:
                continue
            valley = int(middle.min())
            if valley / min(peak_lo, peak_hi) > valley_max:
                continue
            valley_bin_local = int(np.argmin(middle))
            split_val = float(edges[tail + valley_bin_local + 1])
            mask_b = proj >= split_val
            mask_a = ~mask_b
            half_min = max(min_splats // 2, 4)
            if mask_a.sum() < half_min or mask_b.sum() < half_min:
                continue
            new_id = int(new_labels.max()) + 1
            new_labels[idx[mask_b]] = new_id
            splits += 1
            changed = True
        if not changed:
            break
    return new_labels, splits


# --------------------------------------------------------------- OBB + sphere fit

def _algebraic_sphere_fit(pts):
    """Linear-LSQ sphere fit: minimises (||p - c||^2 - r^2)^2. Stable for
    partial-coverage point clouds (a hemisphere of points still recovers the
    full-sphere radius from the curvature)."""
    if len(pts) < 4:
        return None
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = (pts * pts).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = sol[:3]
    r_sq = float(sol[3] + (center * center).sum())
    if not np.isfinite(r_sq) or r_sq <= 0.0:
        return None
    return {"center": np.asarray(center, dtype=np.float64), "radius": float(np.sqrt(r_sq))}


def _fit_obb(points, trim_percentile, extent_percentile, extent_padding,
             sphere_grow=True, sphere_grow_max_aspect=1.8, sphere_grow_min_factor=1.05,
             sphere_grow_max_factor=3.0):
    if len(points) < 4:
        return None
    pts = np.asarray(points, dtype=np.float64)
    center = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - center[None, :], axis=1)
    limit = np.percentile(dist, np.clip(trim_percentile, 50.0, 99.5))
    trimmed = pts[dist <= max(limit, 1e-6)]
    axis_pts = trimmed if len(trimmed) >= 4 else pts
    center = np.median(axis_pts, axis=0)
    centered = axis_pts - center[None, :]
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] = -vecs[:, -1]
    local = (pts - center[None, :]) @ vecs
    pct = float(np.clip(extent_percentile, 80.0, 100.0))
    lower = 0.5 * (100.0 - pct)
    upper = 100.0 - lower
    lo = np.percentile(local, lower, axis=0)
    hi = np.percentile(local, upper, axis=0)
    axes = np.maximum((hi - lo) * 0.5 * (1.0 + max(float(extent_padding), 0.0)), 0.012)
    obb_center = center + vecs @ (0.5 * (lo + hi))

    sphere_info = None
    if sphere_grow:
        sphere = _algebraic_sphere_fit(pts)
        if sphere is not None:
            r_sphere = sphere["radius"]
            axis_max = float(axes.max())
            axis_min = float(axes.min())
            aspect = axis_max / max(axis_min, 1e-9)
            # Only sphere-grow when the cluster looks sphere-like (cubic OBB)
            # and the sphere fit is meaningfully larger than the OBB extent
            # but not absurdly larger (which would indicate a bad fit on
            # near-coplanar points).
            if (aspect <= sphere_grow_max_aspect
                    and r_sphere > axis_max * sphere_grow_min_factor
                    and r_sphere < axis_max * sphere_grow_max_factor):
                axes = np.full(3, r_sphere)
                obb_center = sphere["center"]
                vecs = np.eye(3)
                sphere_info = {
                    "applied": True,
                    "radius": float(r_sphere),
                    "obb_axis_max": axis_max,
                    "aspect": float(aspect),
                }
            else:
                sphere_info = {
                    "applied": False,
                    "radius": float(r_sphere),
                    "obb_axis_max": axis_max,
                    "aspect": float(aspect),
                }
    return {
        "center": obb_center,
        "axes": axes,
        "rotation": vecs,
        "num_fit_splats": int(len(axis_pts)),
        "sphere_fit": sphere_info,
    }


# --------------------------------------------------------------- PLY write

def _set_dc_color(vertex, idx, rgb):
    if "f_dc_0" in vertex.dtype.names:
        vertex["f_dc_0"][idx] = (float(rgb[0]) - 0.5) / SH_C0
        vertex["f_dc_1"][idx] = (float(rgb[1]) - 0.5) / SH_C0
        vertex["f_dc_2"][idx] = (float(rgb[2]) - 0.5) / SH_C0
    for name in vertex.dtype.names or ():
        if name.startswith("f_rest_"):
            vertex[name][idx] = 0.0


def _colorize(vertex, labels, opacity, selected_only):
    if selected_only:
        keep = labels > 0
        out = vertex[keep].copy()
        out_labels = labels[keep]
    else:
        out = vertex.copy()
        out_labels = labels
    for oid in np.unique(out_labels):
        if oid <= 0:
            continue
        idx = np.flatnonzero(out_labels == oid)
        _set_dc_color(out, idx, _color_for(int(oid)))
        if "opacity" in out.dtype.names:
            out["opacity"][idx] = np.maximum(out["opacity"][idx], _logit(opacity))
    return out


def _write_ply(path, template, vertex):
    PlyData([PlyElement.describe(vertex, "vertex")], text=template.text).write(path)


# --------------------------------------------------------------- main

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(args.gaussians)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    votes = np.load(args.votes, allow_pickle=True)
    selected = np.asarray(votes["selected"], dtype=bool)
    selected_raw = (
        np.asarray(votes["selected_raw"], dtype=bool)
        if "selected_raw" in votes else selected.copy()
    )
    score = (
        np.asarray(votes["score"], dtype=np.float32)
        if "score" in votes else np.zeros(len(xyz), dtype=np.float32)
    )

    if "obs_splat_idx" not in votes:
        raise SystemExit(
            "[track_gaussian_objects] votes.npz lacks per-frame observations. "
            "Re-run scripts/vote_gaussian_masks.py to regenerate it."
        )
    obs_splat_idx = np.asarray(votes["obs_splat_idx"], dtype=np.int64)
    obs_mask_id = np.asarray(votes["obs_mask_id"], dtype=np.int32)
    obs_frame_offsets = np.asarray(votes["obs_frame_offsets"], dtype=np.int64)

    prompt_owner = None
    prompt_names = None
    if "prompt_hits" in votes and "prompt_names" in votes:
        prompt_hits = np.asarray(votes["prompt_hits"])
        prompt_names = [str(n) for n in np.asarray(votes["prompt_names"], dtype=object).tolist()]
        if prompt_hits.ndim == 2 and prompt_hits.shape[1] == len(xyz) and prompt_hits.shape[0] == len(prompt_names):
            denom = np.maximum(np.asarray(votes["seen"], dtype=np.float32), 1.0)[None, :]
            prompt_score = prompt_hits.astype(np.float32) / denom
            prompt_owner = prompt_score.argmax(axis=0).astype(np.int16)
            prompt_owner[prompt_hits.max(axis=0) <= 0] = -1

    selection_source = str(getattr(args, "selection_source", "selected_raw")).strip().lower()
    sel = selected_raw if selection_source == "selected_raw" else selected
    sel_idx = np.flatnonzero(sel)
    if len(sel_idx) == 0:
        raise SystemExit("[track_gaussian_objects] no selected splats to cluster")

    # ---- Phase 1: spatial DBSCAN on selected splats -----------------
    eps_mult = float(getattr(args, "cluster_eps_multiplier", 2.5))
    min_fragment = int(getattr(args, "min_fragment_splats", 8))
    sel_xyz = xyz[sel_idx]
    median_nn = _median_neighbor_distance(sel_xyz)
    eps = max(median_nn * eps_mult, 1e-6)
    print(f"[INFO] DBSCAN  eps={eps:.5f}  (median_nn={median_nn:.5f} x mult={eps_mult})  min_fragment={min_fragment}")
    frag_sel = _dbscan_components(sel_xyz, eps, min_fragment)
    fragment_labels = np.zeros(len(xyz), dtype=np.int32)
    fragment_labels[sel_idx] = frag_sel
    n_frag_raw = int(fragment_labels.max())
    print(f"[INFO] DBSCAN  {n_frag_raw} fragments from {len(sel_idx)} selected splats")
    if n_frag_raw == 0:
        raise SystemExit(
            "[track_gaussian_objects] DBSCAN produced 0 fragments. Lower "
            "tracking.min_fragment_splats or raise tracking.cluster_eps_multiplier."
        )

    # Force-separate fragments across different prompt owners.
    if prompt_owner is not None and prompt_names and len(prompt_names) > 1:
        composite = fragment_labels.astype(np.int64) * (len(prompt_names) + 2) \
            + (prompt_owner.astype(np.int64) + 1)
        composite[fragment_labels <= 0] = 0
        unique_ids = np.unique(composite[composite > 0])
        remap = {int(old): new for new, old in enumerate(unique_ids, start=1)}
        new = np.zeros_like(fragment_labels)
        for old, n in remap.items():
            new[composite == old] = n
        fragment_labels = new
        print(f"[INFO] After prompt-aware split: {int(fragment_labels.max())} fragments")

    # ---- Phase 1b: bimodal split inside elongated fragments ---------
    fragment_labels, n_splits = _split_bimodal_fragments(xyz, fragment_labels, args)
    if n_splits:
        print(f"[INFO] Bimodal split  {n_splits} fragments split -> {int(fragment_labels.max())} fragments")

    # ---- Phase 2: SAM-evidence merge --------------------------------
    signatures = _per_fragment_signatures(
        fragment_labels, obs_splat_idx, obs_mask_id, obs_frame_offsets,
        float(getattr(args, "majority_fraction", 0.55)),
    )
    labels, n_merges = _merge_fragments(xyz, fragment_labels, signatures, args)
    print(f"[INFO] SAM merge  {int(fragment_labels.max())} -> {int(labels.max())} objects ({n_merges} merges)")

    # ---- Absorb tiny fragments into overlapping larger neighbours ---
    labels, n_absorbed = _absorb_tiny_fragments(xyz, labels, args)
    if n_absorbed:
        print(f"[INFO] Tiny absorb  {n_absorbed} small fragments folded into larger neighbours")

    # ---- Grow into DBSCAN-noise foreground splats -------------------
    if bool(getattr(args, "grow_orphans", True)):
        labeled_mask = labels > 0
        orphan_idx = np.flatnonzero(sel & ~labeled_mask)
        labeled_idx = np.flatnonzero(labeled_mask)
        n_grown = 0
        if len(orphan_idx) and len(labeled_idx):
            grow_mult = float(getattr(args, "grow_radius_multiplier", 2.0))
            grow_eps = grow_mult * eps
            tree = cKDTree(xyz[labeled_idx])
            d, nn = tree.query(xyz[orphan_idx], k=1, distance_upper_bound=grow_eps)
            keep = np.isfinite(d) & (nn < len(labeled_idx))
            if keep.any():
                labels[orphan_idx[keep]] = labels[labeled_idx[nn[keep]]]
                n_grown = int(keep.sum())
        print(f"[INFO] Grow  {n_grown} orphan foreground splats absorbed (radius={grow_mult}x eps)")

    # ---- Filter tiny objects ----------------------------------------
    min_object = int(getattr(args, "min_object_splats", 30))
    sizes = Counter(int(v) for v in labels[labels > 0])
    drop = [oid for oid, c in sizes.items() if c < min_object]
    if drop:
        labels = np.where(np.isin(labels, drop), 0, labels)
        print(f"[INFO] Dropped {len(drop)} objects below min_object_splats={min_object}")

    # Compact label space to 1..K.
    kept = sorted(int(v) for v in np.unique(labels) if v > 0)
    remap = {old: new for new, old in enumerate(kept, start=1)}
    compact = np.zeros_like(labels)
    for old, new in remap.items():
        compact[labels == old] = new
    labels = compact
    print(f"[INFO] Final object count: {int(labels.max())}")

    # ---- Per-object prompts + OBBs ----------------------------------
    def majority_prompts(idx):
        if prompt_owner is None or not prompt_names:
            return []
        owners = prompt_owner[idx]
        owners = owners[(owners >= 0) & (owners < len(prompt_names))]
        if len(owners) == 0:
            return []
        ids, counts = np.unique(owners, return_counts=True)
        order = np.argsort(counts)[::-1]
        return [
            {"prompt": prompt_names[int(ids[i])], "count": int(counts[i])}
            for i in order
        ]

    trim_pct = float(getattr(args, "obb_trim_percentile", 88.0))
    ext_pct = float(getattr(args, "obb_extent_percentile", 99.0))
    ext_pad = float(getattr(args, "obb_extent_padding", 0.08))
    sphere_grow = bool(getattr(args, "sphere_grow_obb", True))
    sphere_grow_max_aspect = float(getattr(args, "sphere_grow_max_aspect", 1.8))
    sphere_grow_min_factor = float(getattr(args, "sphere_grow_min_factor", 1.05))
    sphere_grow_max_factor = float(getattr(args, "sphere_grow_max_factor", 3.0))
    objects = []
    n_sphere_grown = 0
    for oid in range(1, int(labels.max()) + 1):
        idx = np.flatnonzero(labels == oid)
        geom = _fit_obb(
            xyz[idx], trim_pct, ext_pct, ext_pad,
            sphere_grow=sphere_grow,
            sphere_grow_max_aspect=sphere_grow_max_aspect,
            sphere_grow_min_factor=sphere_grow_min_factor,
            sphere_grow_max_factor=sphere_grow_max_factor,
        )
        if geom is None:
            c = np.median(xyz[idx], axis=0) if len(idx) else np.zeros(3)
            geom = {
                "center": c,
                "axes": np.full(3, 0.01),
                "rotation": np.eye(3),
                "num_fit_splats": int(len(idx)),
            }
        c = np.asarray(geom["center"], dtype=np.float64)
        a = np.asarray(geom["axes"], dtype=np.float64)
        r = np.asarray(geom["rotation"], dtype=np.float64)
        prompts = majority_prompts(idx)
        sphere_info = geom.get("sphere_fit")
        if sphere_info and sphere_info.get("applied"):
            n_sphere_grown += 1
        objects.append({
            "id": int(oid),
            "center": c.tolist(),
            "axes": a.tolist(),
            "rotation": r.tolist(),
            "shape_type": "sphere" if sphere_info and sphere_info.get("applied") else "spatial_component",
            "source_prompts": prompts,
            "class_label": prompts[0]["prompt"] if prompts else "",
            "num_splats": int(len(idx)),
            "num_fit_splats": int(geom["num_fit_splats"]),
            "sphere_fit": sphere_info,
            "obb": {
                "center": c.tolist(),
                "rotation": r.tolist(),
                "half_extents": a.tolist(),
            },
        })

    # ---- Outputs ----------------------------------------------------
    fg_opacity = float(getattr(args, "foreground_opacity", 0.9))
    tracked_scene = _colorize(vertex, labels, fg_opacity, selected_only=False)
    tracked_objects = _colorize(vertex, labels, fg_opacity, selected_only=True)
    _write_ply(out_dir / "tracked_scene_gaussians.ply", ply, tracked_scene)
    _write_ply(out_dir / "tracked_object_gaussians.ply", ply, tracked_objects)

    np.savez_compressed(
        out_dir / "gaussian_tracks.npz",
        labels=labels.astype(np.int32),
        selected=selected.astype(bool),
        selected_raw=selected_raw.astype(bool),
        score=score.astype(np.float32),
        xyz=xyz.astype(np.float32),
    )
    summary = {
        "mode": "sam_evidence_hybrid",
        "num_gaussians": int(len(xyz)),
        "num_selected": int(selected.sum()),
        "num_selected_raw": int(selected_raw.sum()),
        "num_tracked_splats": int((labels > 0).sum()),
        "num_objects": int(labels.max()),
        "num_fragments_raw": n_frag_raw,
        "num_bimodal_splits": int(n_splits),
        "num_sam_merges": int(n_merges),
        "num_tiny_absorbed": int(n_absorbed),
        "num_sphere_grown": int(n_sphere_grown),
        "dbscan_eps": float(eps),
        "median_nn_distance": float(median_nn),
        "objects": objects,
    }
    with open(out_dir / "tracks.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "objects"}, indent=2))
    print(f"[DONE] Wrote object tracks to {out_dir}")


if __name__ == "__main__":
    main()

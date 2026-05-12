import json
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import triton
import triton.language as tl
from scipy.spatial.transform import Rotation
from tqdm import tqdm


@triton.jit
def vote_kernel(
    pts_ptr,
    K_ptr,
    R_ptr,
    t_ptr,
    masks_ptr,
    depths_ptr,
    votes_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DEPTH_ABS_TOL: tl.constexpr,
    DEPTH_REL_TOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    px = tl.load(pts_ptr + offsets * 3 + 0, mask=mask, other=0.0)
    py = tl.load(pts_ptr + offsets * 3 + 1, mask=mask, other=0.0)
    pz = tl.load(pts_ptr + offsets * 3 + 2, mask=mask, other=0.0)

    votes = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for c in range(C):
        r00 = tl.load(R_ptr + c * 9 + 0)
        r01 = tl.load(R_ptr + c * 9 + 1)
        r02 = tl.load(R_ptr + c * 9 + 2)
        r10 = tl.load(R_ptr + c * 9 + 3)
        r11 = tl.load(R_ptr + c * 9 + 4)
        r12 = tl.load(R_ptr + c * 9 + 5)
        r20 = tl.load(R_ptr + c * 9 + 6)
        r21 = tl.load(R_ptr + c * 9 + 7)
        r22 = tl.load(R_ptr + c * 9 + 8)

        t0 = tl.load(t_ptr + c * 3 + 0)
        t1 = tl.load(t_ptr + c * 3 + 1)
        t2 = tl.load(t_ptr + c * 3 + 2)

        fx = tl.load(K_ptr + c * 9 + 0)
        fy = tl.load(K_ptr + c * 9 + 4)
        cx = tl.load(K_ptr + c * 9 + 2)
        cy = tl.load(K_ptr + c * 9 + 5)

        cx_ = r00 * px + r01 * py + r02 * pz + t0
        cy_ = r10 * px + r11 * py + r12 * pz + t1
        cz_ = r20 * px + r21 * py + r22 * pz + t2

        valid_z = cz_ > 0.0

        u = fx * cx_ / cz_ + cx
        v = fy * cy_ / cz_ + cy

        u_int = u.to(tl.int32)
        v_int = v.to(tl.int32)
        in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

        mask_idx = c * H * W + v_int * W + u_int
        mask_idx = tl.where(in_bounds & valid_z, mask_idx, 0)
        mask_val = tl.load(masks_ptr + mask_idx, mask=in_bounds & valid_z & mask, other=0)
        depth_val = tl.load(depths_ptr + mask_idx, mask=in_bounds & valid_z & mask, other=-1.0)
        depth_tol = tl.maximum(DEPTH_ABS_TOL, DEPTH_REL_TOL * depth_val)
        depth_ok = tl.abs(cz_ - depth_val) <= depth_tol

        votes += tl.where(in_bounds & valid_z & mask & (mask_val > 0) & depth_ok, 1, 0)

    tl.store(votes_ptr + offsets, votes, mask=mask)


@triton.jit
def vote_id_kernel(
    pts_ptr,
    point_ids_ptr,
    K_ptr,
    R_ptr,
    t_ptr,
    masks_ptr,
    depths_ptr,
    votes_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DEPTH_ABS_TOL: tl.constexpr,
    DEPTH_REL_TOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    px = tl.load(pts_ptr + offsets * 3 + 0, mask=mask, other=0.0)
    py = tl.load(pts_ptr + offsets * 3 + 1, mask=mask, other=0.0)
    pz = tl.load(pts_ptr + offsets * 3 + 2, mask=mask, other=0.0)
    point_id = tl.load(point_ids_ptr + offsets, mask=mask, other=0)

    votes = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for c in range(C):
        r00 = tl.load(R_ptr + c * 9 + 0)
        r01 = tl.load(R_ptr + c * 9 + 1)
        r02 = tl.load(R_ptr + c * 9 + 2)
        r10 = tl.load(R_ptr + c * 9 + 3)
        r11 = tl.load(R_ptr + c * 9 + 4)
        r12 = tl.load(R_ptr + c * 9 + 5)
        r20 = tl.load(R_ptr + c * 9 + 6)
        r21 = tl.load(R_ptr + c * 9 + 7)
        r22 = tl.load(R_ptr + c * 9 + 8)

        t0 = tl.load(t_ptr + c * 3 + 0)
        t1 = tl.load(t_ptr + c * 3 + 1)
        t2 = tl.load(t_ptr + c * 3 + 2)

        fx = tl.load(K_ptr + c * 9 + 0)
        fy = tl.load(K_ptr + c * 9 + 4)
        cx = tl.load(K_ptr + c * 9 + 2)
        cy = tl.load(K_ptr + c * 9 + 5)

        cx_ = r00 * px + r01 * py + r02 * pz + t0
        cy_ = r10 * px + r11 * py + r12 * pz + t1
        cz_ = r20 * px + r21 * py + r22 * pz + t2

        valid_z = cz_ > 0.0

        u = fx * cx_ / cz_ + cx
        v = fy * cy_ / cz_ + cy

        u_int = u.to(tl.int32)
        v_int = v.to(tl.int32)
        in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

        mask_idx = c * H * W + v_int * W + u_int
        mask_idx = tl.where(in_bounds & valid_z, mask_idx, 0)
        mask_val = tl.load(masks_ptr + mask_idx, mask=in_bounds & valid_z & mask, other=0)
        depth_val = tl.load(depths_ptr + mask_idx, mask=in_bounds & valid_z & mask, other=-1.0)
        depth_tol = tl.maximum(DEPTH_ABS_TOL, DEPTH_REL_TOL * depth_val)
        depth_ok = tl.abs(cz_ - depth_val) <= depth_tol

        votes += tl.where(
            in_bounds & valid_z & mask & (mask_val == point_id) & depth_ok,
            1,
            0,
        )

    tl.store(votes_ptr + offsets, votes, mask=mask)


def pose_processed_hw(p, fallback_hw):
    if "processed_height" in p and "processed_width" in p:
        return int(p["processed_height"]), int(p["processed_width"])
    return fallback_hw


def build_camera_matrices(poses, H, W, processed_hw_by_frame, device):
    C = len(poses)
    K = torch.zeros(C, 3, 3, dtype=torch.float32, device=device)
    R = torch.zeros(C, 3, 3, dtype=torch.float32, device=device)
    t = torch.zeros(C, 3, dtype=torch.float32, device=device)

    for i, p in enumerate(poses):
        proc_h, proc_w = pose_processed_hw(p, processed_hw_by_frame[p["filename"]])
        fx = p["focal"] * (W / proc_w)
        fy = p["focal"] * (H / proc_h)
        K[i] = torch.tensor([
            [fx, 0, W / 2],
            [0, fy, H / 2],
            [0, 0, 1],
        ], dtype=torch.float32)

        c2w = torch.zeros(4, 4, dtype=torch.float32)
        c2w[3, 3] = 1.0

        rot = Rotation.from_quat([p["qx"], p["qy"], p["qz"], p["qw"]]).as_matrix()
        c2w[:3, :3] = torch.tensor(rot, dtype=torch.float32)
        c2w[:3, 3] = torch.tensor([p["tx"], p["ty"], p["tz"]], dtype=torch.float32)

        w2c = torch.linalg.inv(c2w)
        R[i] = w2c[:3, :3]
        t[i] = w2c[:3, 3]

    return K, R, t


def resize_mask_to_points(mask, target_hw, pose):
    target_h, target_w = target_hw
    if mask.shape == (target_h, target_w):
        return mask

    if "resized_width" in pose and "resized_height" in pose:
        resized_w = int(pose["resized_width"])
        resized_h = int(pose["resized_height"])
        resized = cv2.resize(mask.astype(np.int32), (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        left = int(round(pose.get("crop_left", 0)))
        top = int(round(pose.get("crop_top", 0)))
        return resized[top:top + target_h, left:left + target_w].astype(np.int32)

    return cv2.resize(mask.astype(np.int32), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def load_instance_mask(mask_path, preferred_key="instance_mask"):
    data = np.load(mask_path, allow_pickle=True)

    for key in [preferred_key, "instance_mask", "local_instance_mask", "mask"]:
        if key in data:
            mask = data[key]
            break
    else:
        raise KeyError(
            f"{mask_path} has no mask key. Available keys: {list(data.keys())}"
        )

    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask from {mask_path}, got {mask.shape}")

    return mask.astype(np.int32)


def id_colors(object_ids):
    colors = np.zeros((len(object_ids), 4), dtype=np.uint8)

    for object_id in np.unique(object_ids):
        rng = np.random.default_rng(int(object_id) * 9973)
        colors[object_ids == object_id, :3] = rng.integers(40, 256, size=3, dtype=np.uint8)
        colors[object_ids == object_id, 3] = 255

    return colors


def load_track_centers(tracks_path):
    if not tracks_path or not os.path.exists(tracks_path):
        return {}

    with open(tracks_path) as f:
        tracks = json.load(f)

    centers = {}
    for cluster in tracks.get("clusters", []):
        if "global_id" not in cluster or "center_3d" not in cluster:
            continue
        centers[int(cluster["global_id"])] = np.asarray(cluster["center_3d"], dtype=np.float32)

    return centers


def compact_filter_points(
    pts,
    object_ids,
    track_centers,
    max_radius,
    radius_quantile,
    min_points,
):
    keep = np.zeros(len(pts), dtype=bool)
    stats = []

    for object_id in np.unique(object_ids):
        object_id = int(object_id)
        idx = np.where(object_ids == object_id)[0]
        obj_pts = pts[idx]

        finite = np.isfinite(obj_pts).all(axis=1)
        if int(finite.sum()) < min_points:
            continue

        valid_idx = idx[finite]
        obj_pts = obj_pts[finite]

        center = track_centers.get(object_id)
        if center is None:
            center = np.median(obj_pts, axis=0)

        dists = np.linalg.norm(obj_pts - center[None, :], axis=1)
        robust_radius = float(np.quantile(dists, radius_quantile))
        keep_radius = min(float(max_radius), robust_radius)

        obj_keep = dists <= keep_radius
        if int(obj_keep.sum()) < min_points:
            keep_radius = robust_radius
            obj_keep = dists <= keep_radius

        keep[valid_idx[obj_keep]] = True
        stats.append({
            "object_id": object_id,
            "before": int(len(idx)),
            "after": int(obj_keep.sum()),
            "center_3d": center.astype(float).tolist(),
            "robust_radius": robust_radius,
            "keep_radius": float(keep_radius),
        })

    return keep, stats


def split_disconnected_object_ids(pts, object_ids, voxel_size, min_points):
    """
    Split one tracked ID into multiple IDs if its 3D points form disconnected blobs.

    This catches cases where 2D tracking merged two partially visible oranges but
    the voted 3D cloud still clearly separates them.
    """
    if len(pts) == 0:
        return object_ids, []

    new_ids = object_ids.astype(np.int32).copy()
    next_id = int(new_ids.max()) + 1 if len(new_ids) else 1
    stats = []

    for object_id in sorted(int(x) for x in np.unique(object_ids)):
        idx = np.where(object_ids == object_id)[0]
        obj_pts = pts[idx]

        finite = np.isfinite(obj_pts).all(axis=1)
        if int(finite.sum()) < min_points:
            continue

        valid_idx = idx[finite]
        obj_pts = obj_pts[finite]

        origin = obj_pts.min(axis=0)
        voxels = np.floor((obj_pts - origin[None, :]) / voxel_size).astype(np.int32)

        voxel_to_points = {}
        for point_i, voxel in enumerate(voxels):
            key = tuple(int(v) for v in voxel)
            voxel_to_points.setdefault(key, []).append(point_i)

        unvisited = set(voxel_to_points)
        components = []
        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        while unvisited:
            start = unvisited.pop()
            stack = [start]
            comp_voxels = [start]

            while stack:
                vx, vy, vz = stack.pop()
                for dx, dy, dz in neighbor_offsets:
                    neighbor = (vx + dx, vy + dy, vz + dz)
                    if neighbor not in unvisited:
                        continue
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
                    comp_voxels.append(neighbor)

            comp_points = []
            for voxel in comp_voxels:
                comp_points.extend(voxel_to_points[voxel])
            components.append(np.asarray(comp_points, dtype=np.int64))

        components = [comp for comp in components if len(comp) >= min_points]
        if len(components) <= 1:
            continue

        components.sort(key=len, reverse=True)
        new_ids_for_object = [object_id]

        for comp in components[1:]:
            point_indices = valid_idx[comp]
            new_ids[point_indices] = next_id
            new_ids_for_object.append(int(next_id))
            next_id += 1

        stats.append({
            "original_id": int(object_id),
            "new_ids": new_ids_for_object,
            "component_sizes": [int(len(comp)) for comp in components],
        })

    return new_ids, stats


def run_voting(orange_pts, K, R, t, masks, depths, min_votes=2, depth_abs_tol=0.03, depth_rel_tol=0.05):
    device = orange_pts.device
    N = orange_pts.shape[0]
    C, H, W = masks.shape

    votes = torch.zeros(N, dtype=torch.int32, device=device)

    BLOCK_SIZE = 512
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vote_kernel[grid](
        orange_pts, K, R, t, masks, depths, votes,
        N, C, H, W,
        DEPTH_ABS_TOL=depth_abs_tol,
        DEPTH_REL_TOL=depth_rel_tol,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return votes >= min_votes


def run_id_voting(orange_pts, object_ids, K, R, t, masks, depths, min_votes=2, depth_abs_tol=0.03, depth_rel_tol=0.05):
    device = orange_pts.device
    N = orange_pts.shape[0]
    C, H, W = masks.shape

    votes = torch.zeros(N, dtype=torch.int32, device=device)

    BLOCK_SIZE = 512
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vote_id_kernel[grid](
        orange_pts, object_ids, K, R, t, masks, depths, votes,
        N, C, H, W,
        DEPTH_ABS_TOL=depth_abs_tol,
        DEPTH_REL_TOL=depth_rel_tol,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return votes >= min_votes


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-view voting to filter orange pointcloud")
    parser.add_argument("--pts-dir", type=str, default="out/dust3r", help="Directory containing pts3d .npz files")
    parser.add_argument("--depth-dir", type=str, default="out/dust3r/depths", help="Directory containing DUSt3R depth .npz files")
    parser.add_argument("--mask-dir", type=str, default="out/masks", help="Directory containing SAM mask .npz files")
    parser.add_argument("--poses", type=str, default="out/dust3r/camera_poses.json")
    parser.add_argument("--out-dir", type=str, default="out/voted_pts")
    parser.add_argument("--min-votes", type=int, default=2, help="Minimum cameras that must agree")
    parser.add_argument("--depth-abs-tol", type=float, default=0.03, help="Absolute depth tolerance for visibility voting")
    parser.add_argument("--depth-rel-tol", type=float, default=0.05, help="Relative depth tolerance for visibility voting")
    parser.add_argument("--track-ids", action="store_true", help="Preserve instance IDs during voting. Use with out/tracked_masks.")
    parser.add_argument("--mask-key", type=str, default="instance_mask", help="Preferred mask key in .npz files")
    parser.add_argument("--tracks", type=str, default="out/tracks/tracks.json", help="Track JSON with global 3D centers")
    parser.add_argument("--no-compact-filter", action="store_true", help="Disable compact per-object 3D filtering")
    parser.add_argument("--max-object-radius", type=float, default=0.09, help="Max distance from object center kept after voting")
    parser.add_argument("--radius-quantile", type=float, default=0.82, help="Per-object distance quantile used for robust radius filtering")
    parser.add_argument("--min-object-points", type=int, default=80, help="Minimum points to keep an object after compact filtering")
    parser.add_argument("--no-split-disconnected", action="store_true", help="Disable splitting disconnected 3D components within one object ID")
    parser.add_argument("--split-voxel-size", type=float, default=0.012, help="Voxel size for disconnected 3D component splitting")
    parser.add_argument("--split-min-points", type=int, default=250, help="Minimum points for a disconnected component to become a new ID")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.poses) as f:
        poses = json.load(f)

    processed_hw_by_frame = {}
    for p in poses:
        stem = Path(p["filename"]).stem
        pts_path = os.path.join(args.pts_dir, f"{stem}_pts3d.npz")
        if os.path.exists(pts_path):
            processed_hw_by_frame[p["filename"]] = np.load(pts_path)["pts3d"].shape[:2]
        elif "processed_height" in p and "processed_width" in p:
            processed_hw_by_frame[p["filename"]] = (int(p["processed_height"]), int(p["processed_width"]))
        else:
            raise FileNotFoundError(
                f"Need {pts_path} to infer DUSt3R processed image size for old pose files"
            )

    H, W = processed_hw_by_frame[poses[0]["filename"]]
    print(f"[INFO] {len(poses)} cameras, DUSt3R image size {H}x{W}")

    K, R, t = build_camera_matrices(poses, H, W, processed_hw_by_frame, args.device)

    print("[INFO] Loading masks and depthmaps...")
    masks = torch.zeros(len(poses), H, W, dtype=torch.int32, device=args.device)
    depths = torch.zeros(len(poses), H, W, dtype=torch.float32, device=args.device)
    for i, p in enumerate(poses):
        mask_path = os.path.join(args.mask_dir, p["filename"].replace(".jpg", ".npz"))
        if not os.path.exists(mask_path):
            continue
        m = load_instance_mask(mask_path, args.mask_key)
        m = resize_mask_to_points(m, (H, W), p)
        masks[i] = torch.tensor(m.astype(np.int32), device=args.device)

        stem = Path(p["filename"]).stem
        depth_path = os.path.join(args.depth_dir, f"{stem}_depth.npz")
        if os.path.exists(depth_path):
            depth = np.load(depth_path)["depth"].astype(np.float32)
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            depths[i] = torch.tensor(depth, device=args.device)

    print("[INFO] Loading pts3d and pre-filtering to orange pixels...")
    all_pts = []
    all_frame_ids = []
    all_object_ids = []

    for i, p in enumerate(tqdm(poses, desc="Loading pts3d")):
        stem = Path(p["filename"]).stem
        pts_path = os.path.join(args.pts_dir, f"{stem}_pts3d.npz")
        mask_path = os.path.join(args.mask_dir, p["filename"].replace(".jpg", ".npz"))

        if not os.path.exists(pts_path) or not os.path.exists(mask_path):
            continue

        data = np.load(pts_path)
        pts3d = data["pts3d"]
        conf_mask = data["mask"]

        sam = load_instance_mask(mask_path, args.mask_key)
        pts_h, pts_w = conf_mask.shape
        sam = resize_mask_to_points(sam, (pts_h, pts_w), p)

        combined = (sam > 0) & conf_mask.astype(bool)
        orange_pts = pts3d[combined]
        object_ids = sam[combined].astype(np.int32)

        all_pts.append(torch.tensor(orange_pts, dtype=torch.float32))
        all_frame_ids.extend([i] * len(orange_pts))
        all_object_ids.append(torch.tensor(object_ids, dtype=torch.int32))

    if len(all_pts) == 0:
        raise RuntimeError("No candidate object points found. Check --mask-dir and --pts-dir.")

    all_pts = torch.cat(all_pts, dim=0).to(args.device)
    all_frame_ids = torch.tensor(all_frame_ids, dtype=torch.int32, device=args.device)
    all_object_ids = torch.cat(all_object_ids, dim=0).to(args.device)
    print(f"[INFO] Total orange candidate points: {len(all_pts):,}")

    print("[INFO] Running voting kernel...")
    if args.track_ids:
        keep = run_id_voting(
            all_pts, all_object_ids, K, R, t, masks, depths,
            min_votes=args.min_votes,
            depth_abs_tol=args.depth_abs_tol,
            depth_rel_tol=args.depth_rel_tol,
        )
    else:
        keep = run_voting(
            all_pts, K, R, t, masks, depths,
            min_votes=args.min_votes,
            depth_abs_tol=args.depth_abs_tol,
            depth_rel_tol=args.depth_rel_tol,
        )
    voted_pts = all_pts[keep].cpu().numpy()
    voted_object_ids = all_object_ids[keep].cpu().numpy()
    print(f"[INFO] Points surviving voting: {len(voted_pts):,} / {len(all_pts):,}")

    filter_stats = []
    if args.track_ids and not args.no_compact_filter:
        track_centers = load_track_centers(args.tracks)
        compact_keep, filter_stats = compact_filter_points(
            voted_pts,
            voted_object_ids,
            track_centers=track_centers,
            max_radius=args.max_object_radius,
            radius_quantile=args.radius_quantile,
            min_points=args.min_object_points,
        )
        print(f"[INFO] Points surviving compact filtering: {int(compact_keep.sum()):,} / {len(voted_pts):,}")
        voted_pts = voted_pts[compact_keep]
        voted_object_ids = voted_object_ids[compact_keep]

    split_stats = []
    if args.track_ids and not args.no_split_disconnected:
        voted_object_ids, split_stats = split_disconnected_object_ids(
            voted_pts,
            voted_object_ids,
            voxel_size=args.split_voxel_size,
            min_points=args.split_min_points,
        )
        if split_stats:
            print(f"[INFO] Split disconnected IDs: {split_stats}")

    if args.track_ids:
        np.savez_compressed(
            os.path.join(args.out_dir, "orange_pts3d.npz"),
            pts3d=voted_pts,
            object_ids=voted_object_ids.astype(np.int32),
            filter_stats=np.array(filter_stats, dtype=object),
            split_stats=np.array(split_stats, dtype=object),
        )
        colors = id_colors(voted_object_ids)
    else:
        np.savez_compressed(os.path.join(args.out_dir, "orange_pts3d.npz"), pts3d=voted_pts)
        colors = np.tile([255, 140, 0, 255], (len(voted_pts), 1)).astype(np.uint8)
    pc = trimesh.PointCloud(voted_pts, colors=colors)
    pc.export(os.path.join(args.out_dir, "orange_pts3d.glb"))
    print(f"[DONE] Saved GLB to {args.out_dir}/orange_pts3d.glb")
    print(f"[DONE] Saved to {args.out_dir}/orange_pts3d.npz")


if __name__ == "__main__":
    main()

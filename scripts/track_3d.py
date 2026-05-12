import argparse
import json
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign stable object IDs to SAM masks using DUSt3R 3D points."
    )
    parser.add_argument("--mask-dir", type=str, default="out/masks")
    parser.add_argument("--pts-dir", type=str, default="out/dust3r")
    parser.add_argument("--out-mask-dir", type=str, default="out/tracked_masks")
    parser.add_argument("--tracks-out", type=str, default="out/tracks/tracks.json")
    parser.add_argument("--use-local-key", type=str, default="local_instance_mask")
    parser.add_argument("--min-points", type=int, default=250)
    parser.add_argument(
        "--expected-objects",
        type=int,
        default=0,
        help="If set, cluster detections into exactly this many static 3D objects.",
    )
    parser.add_argument(
        "--cluster-dist",
        type=float,
        default=0.01,
        help="3D distance threshold for unsupervised object clustering.",
    )
    parser.add_argument(
        "--min-cluster-frames",
        type=int,
        default=2,
        help="Drop automatically found IDs seen in fewer than this many frames.",
    )
    parser.add_argument(
        "--min-cluster-detections",
        type=int,
        default=2,
        help="Drop automatically found IDs with fewer than this many mask detections.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mask_file(mask_path, preferred_key):
    data = np.load(mask_path, allow_pickle=True)

    for key in [preferred_key, "local_instance_mask", "instance_mask", "mask"]:
        if key in data:
            mask = np.squeeze(data[key]).astype(np.int32)
            break
    else:
        raise KeyError(f"{mask_path} has no usable mask key: {list(data.keys())}")

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask from {mask_path}, got {mask.shape}")

    objects = []
    if "objects" in data:
        for obj in data["objects"]:
            if isinstance(obj, dict):
                objects.append(obj)
            elif hasattr(obj, "item") and isinstance(obj.item(), dict):
                objects.append(obj.item())

    return mask, objects


def load_pts3d(pts_path):
    data = np.load(pts_path)

    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ["pts3d", "points", "pts", "points3d"]:
            if key in data:
                pts = data[key]
                break
        else:
            raise KeyError(f"{pts_path} has no pts3d key: {list(data.keys())}")
    else:
        pts = data

    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 points from {pts_path}, got {pts.shape}")

    return pts


def find_pts_file(pts_dir, frame_stem):
    candidates = [
        f"{frame_stem}.npz",
        f"{frame_stem}.npy",
        f"{frame_stem}_pts3d.npz",
        f"{frame_stem}_pts3d.npy",
        f"pts3d_{frame_stem}.npz",
        f"pts3d_{frame_stem}.npy",
    ]

    for name in candidates:
        path = os.path.join(pts_dir, name)
        if os.path.exists(path):
            return path

    return None


def resize_mask_to_pts(mask, pts3d):
    pts_h, pts_w = pts3d.shape[:2]
    if mask.shape == (pts_h, pts_w):
        return mask

    return cv2.resize(
        mask.astype(np.int32),
        (pts_w, pts_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)


def object_records_by_local_id(objects):
    return {
        int(obj["local_id"]): obj
        for obj in objects
        if "local_id" in obj
    }


def bbox_from_mask(mask, local_id):
    ys, xs = np.where(mask == local_id)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def robust_stats(pts):
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return None

    center = np.median(pts, axis=0)
    radius = float(np.median(np.linalg.norm(pts - center[None, :], axis=1)))
    return center.astype(np.float32), radius, int(len(pts))


def extract_detections(frame_name, frame_index, local_mask, objects, pts3d, min_points):
    mask_for_pts = resize_mask_to_pts(local_mask, pts3d)
    records = object_records_by_local_id(objects)
    detections = []

    for local_id in np.unique(mask_for_pts):
        local_id = int(local_id)
        if local_id == 0:
            continue

        stats = robust_stats(pts3d[mask_for_pts == local_id])
        if stats is None:
            continue

        center, radius, num_points = stats
        if num_points < min_points:
            continue

        bbox = bbox_from_mask(local_mask, local_id)
        if bbox is None:
            continue

        record = records.get(local_id, {})
        detections.append({
            "frame_name": frame_name,
            "frame_index": int(frame_index),
            "local_id": local_id,
            "centroid_3d": center.astype(float).tolist(),
            "radius_3d": float(radius),
            "num_points": int(num_points),
            "bbox": bbox,
            "score": float(record["score"]) if "score" in record else None,
            "area": int(record["area"]) if "area" in record else None,
            "sam_idx": int(record["sam_idx"]) if "sam_idx" in record else None,
        })

    return detections


def detection_points(detections):
    return np.asarray([det["centroid_3d"] for det in detections], dtype=np.float32)


def detection_weights(detections):
    weights = np.asarray(
        [max(1, int(det["num_points"])) for det in detections],
        dtype=np.float32,
    )
    return np.minimum(weights, np.percentile(weights, 85))

def unsupervised_clusters(
    detections,
    cluster_dist,
    min_cluster_frames,
    min_cluster_detections,
):
    """
    Infer object IDs from 3D detection centroids without knowing object count.

    High-confidence detections seed clusters first. Smaller/partial detections
    attach to the nearest existing 3D cluster only if they are close enough.
    Weak clusters that appear in too few frames are discarded as SAM noise.
    """
    clusters = []

    for det in sorted(detections, key=lambda d: -d["num_points"]):
        point = np.asarray(det["centroid_3d"], dtype=np.float32)
        best_idx = None
        best_dist = None

        for idx, cluster in enumerate(clusters):
            dist = float(np.linalg.norm(point - cluster["center"]))
            if dist <= cluster_dist and (best_dist is None or dist < best_dist):
                best_idx = idx
                best_dist = dist

        if best_idx is None:
            clusters.append({"center": point.copy(), "detections": [det]})
            continue

        clusters[best_idx]["detections"].append(det)
        members = clusters[best_idx]["detections"]
        centers = detection_points(members)
        weights = detection_weights(members)
        clusters[best_idx]["center"] = np.average(centers, axis=0, weights=weights)

    clusters = [
        cluster for cluster in clusters
        if len(cluster["detections"]) >= min_cluster_detections
        and len({det["frame_index"] for det in cluster["detections"]}) >= min_cluster_frames
    ]

    centers = np.asarray([cluster["center"] for cluster in clusters], dtype=np.float32)
    labels_by_id = {}
    for label, cluster in enumerate(clusters):
        for det in cluster["detections"]:
            labels_by_id[id(det)] = label

    kept_detections = [det for det in detections if id(det) in labels_by_id]
    labels = np.asarray([labels_by_id[id(det)] for det in kept_detections], dtype=np.int32)
    return kept_detections, centers, labels


def assign_global_ids(
    detections,
    expected_objects,
    cluster_dist,
    min_cluster_frames,
    min_cluster_detections,
):
    if len(detections) == 0:
        return [], []
    
    assigned_detections, centers, labels = unsupervised_clusters(
        detections,
        cluster_dist=cluster_dist,
        min_cluster_frames=min_cluster_frames,
        min_cluster_detections=min_cluster_detections,
    )

    if len(assigned_detections) == 0:
        return [], []

    order = sorted(
        range(len(centers)),
        key=lambda i: (float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])),
    )
    label_to_global = {old: new for new, old in enumerate(order, start=1)}
    cluster_members = {label_to_global[label]: [] for label in range(len(centers))}

    for det, label in zip(assigned_detections, labels):
        global_id = int(label_to_global[int(label)])
        det["global_id"] = global_id
        det["match_cost"] = float(np.linalg.norm(
            np.asarray(det["centroid_3d"], dtype=np.float32) - centers[int(label)]
        ))
        cluster_members[global_id].append(det)

    summaries = []
    for old_label in order:
        global_id = int(label_to_global[old_label])
        members = cluster_members[global_id]
        summaries.append({
            "global_id": global_id,
            "center_3d": centers[old_label].astype(float).tolist(),
            "num_detections": int(len(members)),
            "frames": sorted({det["frame_name"] for det in members}),
            "total_points": int(sum(int(det["num_points"]) for det in members)),
        })

    return assigned_detections, summaries


def make_global_mask(local_mask, detections):
    global_mask = np.zeros_like(local_mask, dtype=np.int32)

    for det in detections:
        global_mask[local_mask == int(det["local_id"])] = int(det["global_id"])

    return global_mask


def update_objects(objects, detections):
    local_to_global = {
        int(det["local_id"]): int(det["global_id"])
        for det in detections
    }
    updated = []
    seen = set()

    for obj in objects:
        obj = dict(obj)
        local_id = int(obj.get("local_id", -1))
        if local_id in local_to_global:
            obj["global_id"] = local_to_global[local_id]
        seen.add(local_id)
        updated.append(obj)

    for det in detections:
        local_id = int(det["local_id"])
        if local_id in seen:
            continue
        updated.append({
            "local_id": local_id,
            "global_id": int(det["global_id"]),
            "score": det.get("score"),
            "area": det.get("area"),
            "bbox": det.get("bbox"),
        })

    return updated


def main():
    args = parse_args()

    if not os.path.exists(args.mask_dir):
        raise FileNotFoundError(f"Missing mask directory: {args.mask_dir}")
    if not os.path.exists(args.pts_dir):
        raise FileNotFoundError(f"Missing pts3d directory: {args.pts_dir}")

    if os.path.exists(args.out_mask_dir):
        if args.overwrite:
            shutil.rmtree(args.out_mask_dir)
        else:
            raise FileExistsError(f"{args.out_mask_dir} exists. Use --overwrite.")

    os.makedirs(args.out_mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tracks_out), exist_ok=True)

    frame_records = []
    detections = []
    mask_files = sorted(f for f in os.listdir(args.mask_dir) if f.endswith(".npz"))

    for frame_index, mask_file in enumerate(tqdm(mask_files, desc="Extracting 3D detections", unit="frame")):
        frame_stem = os.path.splitext(mask_file)[0]
        pts_path = find_pts_file(args.pts_dir, frame_stem)
        if pts_path is None:
            print(f"[WARN] No pts3d file found for {frame_stem}, skipping")
            continue

        mask_path = os.path.join(args.mask_dir, mask_file)
        local_mask, objects = load_mask_file(mask_path, args.use_local_key)
        pts3d = load_pts3d(pts_path)
        frame_detections = extract_detections(
            frame_name=frame_stem,
            frame_index=frame_index,
            local_mask=local_mask,
            objects=objects,
            pts3d=pts3d,
            min_points=args.min_points,
        )

        detections.extend(frame_detections)
        frame_records.append({
            "frame_stem": frame_stem,
            "mask_file": mask_file,
            "pts_path": pts_path,
            "local_mask": local_mask,
            "objects": objects,
            "detections": frame_detections,
        })

    detections, cluster_summaries = assign_global_ids(
        detections,
        expected_objects=args.expected_objects,
        cluster_dist=args.cluster_dist,
        min_cluster_frames=args.min_cluster_frames,
        min_cluster_detections=args.min_cluster_detections,
    )

    detections_by_frame = {}
    for det in detections:
        detections_by_frame.setdefault(det["frame_name"], []).append(det)

    tracks = {
        "settings": {
            "mask_dir": args.mask_dir,
            "pts_dir": args.pts_dir,
            "out_mask_dir": args.out_mask_dir,
            "expected_objects": args.expected_objects,
            "cluster_dist": args.cluster_dist,
            "min_cluster_frames": args.min_cluster_frames,
            "min_cluster_detections": args.min_cluster_detections,
            "min_points": args.min_points,
        },
        "clusters": cluster_summaries,
        "frames": {},
    }

    for record in tqdm(frame_records, desc="Writing tracked masks", unit="frame"):
        frame_stem = record["frame_stem"]
        assigned = sorted(
            detections_by_frame.get(frame_stem, []),
            key=lambda det: int(det["local_id"]),
        )
        local_to_global = {
            int(det["local_id"]): int(det["global_id"])
            for det in assigned
        }

        np.savez_compressed(
            os.path.join(args.out_mask_dir, record["mask_file"]),
            instance_mask=make_global_mask(record["local_mask"], assigned),
            local_instance_mask=record["local_mask"].astype(np.int32),
            objects=np.array(update_objects(record["objects"], assigned), dtype=object),
            local_to_global=np.array(local_to_global, dtype=object),
            detections=np.array(assigned, dtype=object),
        )

        tracks["frames"][frame_stem] = {
            "mask_file": record["mask_file"],
            "pts_file": os.path.relpath(record["pts_path"]),
            "local_to_global": local_to_global,
            "objects": assigned,
        }

    with open(args.tracks_out, "w") as f:
        json.dump(tracks, f, indent=2)

    print(f"[DONE] Saved tracked masks to {args.out_mask_dir}")
    print(f"[DONE] Saved tracks to {args.tracks_out}")
    print(f"[INFO] Found {len(cluster_summaries)} global object IDs")


if __name__ == "__main__":
    main()

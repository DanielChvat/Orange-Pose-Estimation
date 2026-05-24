import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import importlib.metadata
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

SAM3_BPE_PATH = Path(__file__).resolve().parents[1] / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config
from pipeline_progress import progress_bar, progress_iter


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from video and run SAM3 instance-style segmentation"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--prompt", action="append", default=None, help="May be given multiple times. A single value may still embed ';' or '|' separators.")
    parser.add_argument("--fps", type=float, default=None, help="Target video FPS sampled for SAM masks")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--no-extract", action="store_true", help="Skip frame extraction and only run SAM on existing frames")
    parser.add_argument("--full-image-pass", type=str_to_bool, default=None, help="Run whole-frame SAM before tiled recall")
    parser.add_argument("--detail-pass", type=str_to_bool, default=None, help="Run padded tiled SAM recall after the whole-frame pass")
    parser.add_argument("--tile-context-margin", type=int, default=None, help="-1 uses an automatic padded tile context margin")
    cli = parser.parse_args()
    overrides = {
        "video_path": cli.video,
        "text_prompt": cli.prompt,
        "fps": cli.fps,
        "no_extract": cli.no_extract,
        "full_image_pass": cli.full_image_pass,
        "detail_pass": cli.detail_pass,
        "tile_context_margin": cli.tile_context_margin,
    }
    if cli.out_root:
        overrides["frame_dir"] = os.path.join(cli.out_root, "frames")
        overrides["mask_dir"] = os.path.join(cli.out_root, "masks")
    args = namespace_from_config(cli.config, "masks", overrides)
    args.config = cli.config
    args.out_root = cli.out_root
    return args


def expand_prompts(primary_prompt, extra_prompts):
    raw_prompts = []
    if isinstance(primary_prompt, (list, tuple)):
        raw_prompts.extend(str(item) for item in primary_prompt if item is not None)
    elif primary_prompt:
        raw_prompts.append(str(primary_prompt))
    if isinstance(extra_prompts, str):
        raw_prompts.append(extra_prompts)
    else:
        raw_prompts.extend(str(prompt) for prompt in (extra_prompts or []))

    prompts = []
    seen = set()
    for raw_prompt in raw_prompts:
        parts = re.split(r"\s*[;|]\s*", raw_prompt)
        for part in parts:
            prompt = part.strip()
            if not prompt:
                continue
            key = prompt.casefold()
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prompt)
    return prompts


def version_tuple(value):
    parts = []
    for piece in re.split(r"[^0-9]+", str(value)):
        if piece:
            parts.append(int(piece))
    return tuple(parts or [0])


def extract_frames(video_path, frame_dir, fps):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        raise ValueError(f"Could not read FPS from video: {video_path}")

    frame_interval = max(1, int(round(video_fps / fps)))
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)

    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    os.makedirs(frame_dir, exist_ok=True)

    frame_idx = 0
    saved_idx = 0
    manifest = []

    with progress_bar(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if rotation in rotations:
                frame = cv2.rotate(frame, rotations[rotation])

            if frame_idx % frame_interval == 0:
                out_path = os.path.join(frame_dir, f"frame_{saved_idx:04d}.jpg")
                cv2.imwrite(out_path, frame)
                manifest.append({
                    "file_name": os.path.basename(out_path),
                    "saved_idx": int(saved_idx),
                    "source_frame_idx": int(frame_idx),
                    "time_sec": float(frame_idx / video_fps),
                })
                saved_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    manifest_path = Path(frame_dir) / "frames_manifest.json"
    manifest_path.write_text(json.dumps({
        "video_path": str(video_path),
        "video_fps": float(video_fps),
        "sample_fps": float(fps),
        "frame_interval": int(frame_interval),
        "total_video_frames": int(total_frames),
        "frames": manifest,
    }, indent=2))
    print(f"[INFO] Extracted {saved_idx} frames")


def component_records(binary_mask, sam_idx, score, min_area, offset=(0, 0)):
    binary_mask = np.squeeze(binary_mask).astype(np.uint8)

    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {binary_mask.shape}")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    components = []
    offset_x, offset_y = offset

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        component = labels == label_id
        x = int(stats[label_id, cv2.CC_STAT_LEFT]) + offset_x
        y = int(stats[label_id, cv2.CC_STAT_TOP]) + offset_y
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]

        components.append({
            "mask": component,
            "sam_idx": int(sam_idx),
            "score": float(score),
            "area": area,
            "bbox": [x, y, x + w - 1, y + h - 1],
            "centroid": [float(cx + offset_x), float(cy + offset_y)],
        })

    return components


def candidate_source_rank(candidate):
    source = str(candidate.get("source", "")).casefold()
    return {
        "global": 4,
        "full": 3,
        "context_refine": 3,
        "box_refine": 3,
        "tile_cluster": 2,
        "tile_fragment": 1,
        "tile": 1,
    }.get(source, 0)


def crop_windows(width, height, tile_size, overlap):
    if tile_size <= 0 or tile_size >= max(width, height):
        return [(0, 0, width, height)]

    step = max(1, int(round(tile_size * (1.0 - overlap))))

    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))
    if xs[-1] != width - tile_size:
        xs.append(max(0, width - tile_size))
    if ys[-1] != height - tile_size:
        ys.append(max(0, height - tile_size))

    return [
        (x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in ys
        for x in xs
    ]


def tile_core_bounds(window, image_size, tile_size, overlap, margin):
    x0, y0, x1, y1 = window
    width, height = image_size
    if tile_size <= 0 or tile_size >= max(width, height):
        return 0, 0, width, height

    if margin < 0:
        margin = int(round(tile_size * overlap * 0.5))

    cx0 = x0 + margin if x0 > 0 else 0
    cy0 = y0 + margin if y0 > 0 else 0
    cx1 = x1 - margin if x1 < width else width
    cy1 = y1 - margin if y1 < height else height

    if cx0 >= cx1 or cy0 >= cy1:
        return x0, y0, x1, y1
    return cx0, cy0, cx1, cy1


def component_is_in_tile_core(component, core_bounds, min_core_area_ratio):
    cx, cy = component["centroid"]
    x0, y0, x1, y1 = core_bounds
    if x0 <= cx < x1 and y0 <= cy < y1:
        return True

    if min_core_area_ratio <= 0:
        return False

    mask = component["mask"]
    area = int(component["area"])
    if area <= 0:
        return False

    core_area = int(mask[y0:y1, x0:x1].sum())
    return core_area / area >= min_core_area_ratio


def expand_window(window, image_size, margin):
    x0, y0, x1, y1 = window
    width, height = image_size
    margin = max(0, int(round(margin)))
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def touches_internal_crop_edge(component, crop_window, image_size, tolerance=2):
    x0, y0, x1, y1 = crop_window
    width, height = image_size
    bx0, by0, bx1, by1 = component["bbox"]
    tolerance = max(0, int(tolerance))
    touches_left = x0 > 0 and bx0 <= x0 + tolerance
    touches_top = y0 > 0 and by0 <= y0 + tolerance
    touches_right = x1 < width and bx1 >= x1 - 1 - tolerance
    touches_bottom = y1 < height and by1 >= y1 - 1 - tolerance
    return bool(touches_left or touches_top or touches_right or touches_bottom)


def bbox_touches_image_edge(bbox, image_size, tolerance=2):
    width, height = image_size
    x0, y0, x1, y1 = [int(v) for v in bbox]
    tolerance = max(0, int(tolerance))
    return bool(
        x0 <= tolerance
        or y0 <= tolerance
        or x1 >= width - 1 - tolerance
        or y1 >= height - 1 - tolerance
    )


def mask_iou(mask_a, mask_b):
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / max(1, union)


def bbox_intersection(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x0 > x1 or y0 > y1:
        return None
    return x0, y0, x1, y1


def candidate_overlap(candidate_a, candidate_b):
    intersection = bbox_intersection(candidate_a["bbox"], candidate_b["bbox"])
    if intersection is None:
        return 0.0, 0.0

    x0, y0, x1, y1 = intersection
    mask_a = candidate_a["mask"][y0:y1 + 1, x0:x1 + 1]
    mask_b = candidate_b["mask"][y0:y1 + 1, x0:x1 + 1]
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0, 0.0

    area_a = int(candidate_a["area"])
    area_b = int(candidate_b["area"])
    union = area_a + area_b - inter
    return inter / max(1, union), inter / max(1, min(area_a, area_b))


def bbox_area(bbox):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if x1 < x0 or y1 < y0:
        return 0
    return int((x1 - x0 + 1) * (y1 - y0 + 1))


def bbox_contains(outer, inner, image_shape, margin_ratio=0.0):
    height, width = image_shape
    ox0, oy0, ox1, oy1 = [float(v) for v in outer]
    ix0, iy0, ix1, iy1 = [float(v) for v in inner]
    ow = max(1.0, ox1 - ox0 + 1.0)
    oh = max(1.0, oy1 - oy0 + 1.0)
    margin = float(margin_ratio) * max(ow, oh)
    ox0 = max(0.0, ox0 - margin)
    oy0 = max(0.0, oy0 - margin)
    ox1 = min(float(width - 1), ox1 + margin)
    oy1 = min(float(height - 1), oy1 + margin)
    return bool(ox0 <= ix0 and oy0 <= iy0 and ox1 >= ix1 and oy1 >= iy1)


def bbox_overlap_over_smaller(bbox_a, bbox_b):
    intersection = bbox_intersection(bbox_a, bbox_b)
    if intersection is None:
        return 0.0
    inter_area = bbox_area(intersection)
    return inter_area / max(1, min(bbox_area(bbox_a), bbox_area(bbox_b)))


def normalize_source_set(sources):
    if sources is None:
        return set()
    if isinstance(sources, str):
        sources = re.split(r"\s*[,|;]\s*", sources)
    return {str(source).strip().casefold() for source in sources if str(source).strip()}


def filter_duplicate_artifact_candidates(
    candidates,
    image_shape,
    enabled=True,
    contained_area_ratio=0.02,
    edge_area_ratio=0.08,
    bbox_margin_ratio=0.04,
    min_reference_area=20000,
    edge_sources=None,
):
    if not enabled or len(candidates) <= 1:
        return candidates, []

    edge_sources = normalize_source_set(edge_sources)
    prompt_groups = {}
    for idx, candidate in enumerate(candidates):
        prompt_groups.setdefault(str(candidate.get("prompt", "")), []).append((idx, candidate))

    drop_indices = set()
    drop_reasons = []
    for prompt, group in prompt_groups.items():
        if not prompt or len(group) <= 1:
            continue

        by_area = sorted(
            group,
            key=lambda item: int(item[1].get("area", 0)),
            reverse=True,
        )
        for idx, candidate in group:
            area = int(candidate.get("area", 0))
            if area <= 0:
                continue

            source = str(candidate.get("source", "")).casefold()
            for ref_idx, reference in by_area:
                if ref_idx == idx:
                    continue
                ref_area = int(reference.get("area", 0))
                if ref_area <= area or ref_area < int(min_reference_area):
                    continue

                area_ratio = area / max(1, ref_area)
                nested = (
                    area_ratio <= float(contained_area_ratio)
                    and (
                        bbox_contains(
                            reference["bbox"],
                            candidate["bbox"],
                            image_shape=image_shape,
                            margin_ratio=bbox_margin_ratio,
                        )
                        or bbox_overlap_over_smaller(candidate["bbox"], reference["bbox"]) >= 0.92
                    )
                )
                edge_artifact = (
                    area_ratio <= float(edge_area_ratio)
                    and (not edge_sources or source in edge_sources)
                    and bbox_touches_image_edge(
                        candidate["bbox"],
                        image_size=(image_shape[1], image_shape[0]),
                        tolerance=3,
                    )
                )

                if nested or edge_artifact:
                    drop_indices.add(idx)
                    drop_reasons.append({
                        "prompt": prompt,
                        "source": candidate.get("source"),
                        "score": float(candidate.get("score", 0.0)),
                        "area": area,
                        "reference_area": ref_area,
                        "reason": "nested" if nested else "edge",
                    })
                    break

    if not drop_indices:
        return candidates, []

    return [
        candidate for idx, candidate in enumerate(candidates)
        if idx not in drop_indices
    ], drop_reasons


def candidate_seam_merge(candidate_a, candidate_b, max_gap=18, min_axis_overlap=0.35):
    if str(candidate_a.get("prompt", "")) != str(candidate_b.get("prompt", "")):
        return False

    tile_a = candidate_a.get("tile")
    tile_b = candidate_b.get("tile")
    if tile_a is not None and tile_b is not None and tile_a == tile_b:
        return False

    ax0, ay0, ax1, ay1 = candidate_a["bbox"]
    bx0, by0, bx1, by1 = candidate_b["bbox"]
    aw = max(1, ax1 - ax0 + 1)
    ah = max(1, ay1 - ay0 + 1)
    bw = max(1, bx1 - bx0 + 1)
    bh = max(1, by1 - by0 + 1)

    x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    y_overlap = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    x_gap = max(0, max(ax0, bx0) - min(ax1, bx1) - 1)
    y_gap = max(0, max(ay0, by0) - min(ay1, by1) - 1)

    horizontal_join = x_gap <= max_gap and y_overlap / max(1, min(ah, bh)) >= min_axis_overlap
    vertical_join = y_gap <= max_gap and x_overlap / max(1, min(aw, bw)) >= min_axis_overlap
    if not (horizontal_join or vertical_join):
        return False

    area_a = int(candidate_a["area"])
    area_b = int(candidate_b["area"])
    if min(area_a, area_b) / max(1, max(area_a, area_b)) < 0.08:
        return False

    return True


def candidate_bbox_cluster_merge(
    candidate_a,
    candidate_b,
    max_gap_ratio=0.8,
    min_axis_overlap=0.12,
):
    if str(candidate_a.get("prompt", "")) != str(candidate_b.get("prompt", "")):
        return False

    ax0, ay0, ax1, ay1 = candidate_a["bbox"]
    bx0, by0, bx1, by1 = candidate_b["bbox"]
    aw = max(1, ax1 - ax0 + 1)
    ah = max(1, ay1 - ay0 + 1)
    bw = max(1, bx1 - bx0 + 1)
    bh = max(1, by1 - by0 + 1)

    x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    y_overlap = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    x_gap = max(0, max(ax0, bx0) - min(ax1, bx1) - 1)
    y_gap = max(0, max(ay0, by0) - min(ay1, by1) - 1)
    max_x_gap = max(16.0, float(max_gap_ratio) * min(aw, bw))
    max_y_gap = max(16.0, float(max_gap_ratio) * min(ah, bh))

    horizontal_join = (
        x_gap <= max_x_gap
        and y_overlap / max(1, min(ah, bh)) >= float(min_axis_overlap)
    )
    vertical_join = (
        y_gap <= max_y_gap
        and x_overlap / max(1, min(aw, bw)) >= float(min_axis_overlap)
    )
    if horizontal_join or vertical_join:
        return True

    acx = 0.5 * (ax0 + ax1)
    acy = 0.5 * (ay0 + ay1)
    bcx = 0.5 * (bx0 + bx1)
    bcy = 0.5 * (by0 + by1)
    dist = float(np.hypot(acx - bcx, acy - bcy))
    scale = max(1.0, 0.5 * (np.hypot(aw, ah) + np.hypot(bw, bh)))
    return dist <= max(1.15, float(max_gap_ratio) + 0.45) * scale


def cluster_fragment_proposals(
    proposals,
    min_fragments,
    max_gap_ratio,
    min_axis_overlap,
):
    if len(proposals) < max(2, int(min_fragments)):
        return []

    uf = UnionFind(len(proposals))
    for i, candidate_a in enumerate(proposals):
        for j in range(i + 1, len(proposals)):
            candidate_b = proposals[j]
            if candidate_bbox_cluster_merge(
                candidate_a,
                candidate_b,
                max_gap_ratio=max_gap_ratio,
                min_axis_overlap=min_axis_overlap,
            ):
                uf.union(i, j)

    groups = {}
    for idx in range(len(proposals)):
        groups.setdefault(uf.find(idx), []).append(idx)

    clustered = []
    for group_indices in groups.values():
        if len(group_indices) < int(min_fragments):
            continue

        group = [proposals[idx] for idx in group_indices]
        best = max(group, key=lambda c: (float(c.get("score", 0.0)), int(c.get("area", 0))))
        mask = np.zeros_like(best["mask"], dtype=bool)
        for proposal in group:
            mask |= proposal["mask"].astype(bool)

        record = {k: v for k, v in best.items() if k != "mask"}
        record["mask"] = mask
        record["source"] = "tile_cluster"
        record["area"] = int(mask.sum())
        record["bbox"] = merged_bbox(mask)
        record["centroid"] = merged_centroid(mask)
        record["cluster_count"] = int(len(group))
        record["cluster_bboxes"] = [proposal["bbox"] for proposal in group]
        record["cluster_scores"] = [float(proposal.get("score", 0.0)) for proposal in group]
        record["truncated_by_crop"] = True
        clustered.append(record)

    return sorted(
        clustered,
        key=lambda c: (int(c.get("cluster_count", 0)), int(c.get("area", 0)), float(c.get("score", 0.0))),
        reverse=True,
    )


class UnionFind:
    def __init__(self, n_items):
        self.parent = list(range(n_items))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True


def merged_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, -1, -1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def merged_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def expanded_bbox_xyxy(bbox, image_size, padding_ratio):
    width, height = image_size
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    pad = float(padding_ratio)
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    x0 = max(0.0, min(float(width - 1), x0))
    x1 = max(0.0, min(float(width - 1), x1))
    y0 = max(0.0, min(float(height - 1), y0))
    y1 = max(0.0, min(float(height - 1), y1))
    return x0, y0, x1, y1


def normalized_cxcywh_from_xyxy(box, image_size):
    width, height = image_size
    x0, y0, x1, y1 = box
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    cx = x0 + 0.5 * bw
    cy = y0 + 0.5 * bh
    return [
        float(cx / max(width, 1)),
        float(cy / max(height, 1)),
        float(bw / max(width, 1)),
        float(bh / max(height, 1)),
    ]


def clone_image_state(state):
    return {
        "original_height": state["original_height"],
        "original_width": state["original_width"],
        "backbone_out": dict(state["backbone_out"]),
    }


def refine_tile_fragments_with_boxes(
    image,
    proposals,
    threshold,
    mask_threshold,
    min_area,
    processor,
    device,
    max_proposals,
    min_proposal_area,
    padding_ratio,
    min_overlap,
):
    if not proposals:
        return []

    width, height = image.size
    proposals = [
        proposal for proposal in proposals
        if int(proposal.get("area", 0)) >= int(min_proposal_area)
    ]
    if not proposals:
        return []

    proposals = sorted(
        proposals,
        key=lambda c: (str(c.get("prompt", "")), -int(c.get("area", 0)), -float(c.get("score", 0.0))),
    )[:max(1, int(max_proposals))]

    use_cuda = device == "cuda"
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
        base_state = processor.set_image(image)

    refined = []
    for proposal in proposals:
        if device == "cuda":
            torch.cuda.empty_cache()
        prompt = str(proposal.get("prompt", "")).strip()
        if not prompt:
            continue
        box_xyxy = expanded_bbox_xyxy(
            proposal["bbox"],
            (width, height),
            padding_ratio,
        )
        box = normalized_cxcywh_from_xyxy(box_xyxy, (width, height))
        state = clone_image_state(base_state)
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
                prompt_outputs_for_state(
                    processor=processor,
                    state=state,
                    prompt=prompt,
                    device=device,
                )
                output = processor.add_geometric_prompt(box=box, label=True, state=state)
        except Exception as exc:
            print(f"[WARN] Box-refine SAM prompt failed for {prompt!r}: {exc}", flush=True)
            continue

        masks = output["masks"]
        scores = output["scores"]
        if len(masks) == 0:
            continue
        masks_np = masks.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy().reshape(-1)
        proposal_mask = proposal["mask"].astype(bool)
        for idx in np.where(scores_np > threshold)[0]:
            mask = np.squeeze(masks_np[idx])
            if mask.ndim != 2:
                continue
            mask = mask > mask_threshold
            overlap = int(np.logical_and(mask, proposal_mask).sum()) / max(1, int(proposal_mask.sum()))
            if overlap < float(min_overlap):
                continue
            for component in component_records(
                mask,
                sam_idx=idx,
                score=scores_np[idx],
                min_area=max(int(min_area), int(min_proposal_area)),
            ):
                component["prompt"] = prompt
                component["prompt_idx"] = int(proposal.get("prompt_idx", -1))
                component["source"] = "box_refine"
                component["crop"] = [0, 0, int(width), int(height)]
                component["tile"] = proposal.get("tile")
                component["tile_core"] = proposal.get("tile_core")
                component["tile_core_kept"] = True
                component["truncated_by_crop"] = False
                component["proposal_bbox"] = proposal.get("bbox")
                component["proposal_score"] = float(proposal.get("score", 0.0))
                component["proposal_source"] = proposal.get("source")
                refined.append(component)

    return refined


def refine_tile_fragments_with_context(
    image,
    proposals,
    mask_threshold,
    min_area,
    processor,
    device,
    max_proposals,
    max_side,
    padding_ratio,
    min_score,
    min_overlap,
    max_area_ratio,
    reject_image_edge,
    edge_tolerance,
    cluster_min_score,
):
    if not proposals:
        return []

    width, height = image.size
    context_processor = Sam3Processor(processor.model, confidence_threshold=0.0, device=device)
    proposals = sorted(
        proposals,
        key=lambda c: (str(c.get("prompt", "")), -int(c.get("area", 0)), -float(c.get("score", 0.0))),
    )[:max(1, int(max_proposals))]

    refined = []
    for proposal in proposals:
        prompt = str(proposal.get("prompt", "")).strip()
        if not prompt:
            continue

        crop_xyxy = expanded_bbox_xyxy(proposal["bbox"], (width, height), padding_ratio)
        x0, y0, x1, y1 = [int(round(v)) for v in crop_xyxy]
        x1 = min(width, max(x0 + 1, x1 + 1))
        y1 = min(height, max(y0 + 1, y1 + 1))
        crop_image = image.crop((x0, y0, x1, y1))
        crop_w, crop_h = crop_image.size
        scale = min(float(max_side) / float(max(crop_w, crop_h)), 1.0) if max_side > 0 else 1.0
        small_w = max(16, int(round(crop_w * scale)))
        small_h = max(16, int(round(crop_h * scale)))
        if scale < 1.0:
            crop_image = crop_image.resize((small_w, small_h), Image.Resampling.BILINEAR)

        proposal_min_score = float(cluster_min_score) if str(proposal.get("source", "")) == "tile_cluster" else float(min_score)
        _, crop_candidates = sam_components_for_crop(
            image=crop_image,
            crop_window=(0, 0, small_w, small_h),
            keep_bounds=(0, 0, small_w, small_h),
            source="context_refine_small",
            prompts=[prompt],
            threshold=proposal_min_score,
            mask_threshold=mask_threshold,
            min_area=max(1, int(round(min_area * scale * scale))),
            tile_min_core_area_ratio=0.0,
            keep_bounds_filter=False,
            reject_truncated_components=False,
            edge_tolerance=0,
            collect_raw_candidates=False,
            processor=context_processor,
            device=device,
        )

        proposal_mask = proposal["mask"].astype(bool)
        proposal_area = max(1, int(proposal_mask.sum()))
        proposal_refined = []
        for candidate in crop_candidates:
            crop_mask = cv2.resize(
                candidate["mask"].astype(np.uint8),
                (crop_w, crop_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y0:y1, x0:x1] = crop_mask
            area = int(full_mask.sum())
            if area < min_area:
                continue
            if area / float(proposal_area) > float(max_area_ratio):
                continue
            overlap = int(np.logical_and(full_mask, proposal_mask).sum()) / float(proposal_area)
            if overlap < float(min_overlap):
                continue
            bbox = merged_bbox(full_mask)
            reject_edge = bool(reject_image_edge) and str(proposal.get("source", "")) != "tile_cluster"
            if reject_edge and bbox_touches_image_edge(
                bbox,
                image_size=(width, height),
                tolerance=edge_tolerance,
            ):
                continue

            record = {k: v for k, v in candidate.items() if k != "mask"}
            record["mask"] = full_mask
            record["prompt"] = prompt
            record["prompt_idx"] = int(proposal.get("prompt_idx", -1))
            record["source"] = "context_refine"
            sam_score = float(candidate.get("score", 0.0))
            support_score = float(proposal.get("score", sam_score))
            record["score"] = max(sam_score, support_score) if str(proposal.get("source", "")) == "tile_cluster" else sam_score
            record["sam_score"] = sam_score
            record["support_score"] = support_score
            record["area"] = area
            record["bbox"] = bbox
            record["centroid"] = merged_centroid(full_mask)
            record["crop"] = [int(x0), int(y0), int(x1), int(y1)]
            record["context_scale"] = float(scale)
            record["tile"] = proposal.get("tile")
            record["tile_core"] = proposal.get("tile_core")
            record["tile_core_kept"] = True
            record["truncated_by_crop"] = False
            record["proposal_bbox"] = proposal.get("bbox")
            record["proposal_score"] = float(proposal.get("score", 0.0))
            record["proposal_source"] = proposal.get("source")
            record["proposal_overlap"] = float(overlap)
            proposal_refined.append(record)

        if proposal_refined:
            refined.append(max(
                proposal_refined,
                key=lambda record: (
                    float(record.get("score", 0.0)),
                    -float(record.get("area", 0)),
                ),
            ))

        if device == "cuda":
            torch.cuda.empty_cache()

    return refined


def merge_candidate_instances(candidates, merge_iou, merge_overlap):
    if len(candidates) <= 1:
        return candidates

    uf = UnionFind(len(candidates))
    for i, candidate_a in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            candidate_b = candidates[j]
            if str(candidate_a.get("prompt", "")) != str(candidate_b.get("prompt", "")):
                continue

            iou, overlap = candidate_overlap(candidate_a, candidate_b)
            source_pair = {str(candidate_a.get("source", "")), str(candidate_b.get("source", ""))}
            has_context_candidate = bool(source_pair & {"full", "global"})
            # Stricter containment requirement for full/global passes. The whole-
            # image SAM passes can produce blob masks that cover many adjacent
            # objects in dense scenes (e.g. tight clusters of fruit). At the
            # previous 0.08 threshold an 8% containment was enough to absorb
            # every tile-level individual into the blob -- collapsing N real
            # objects into one mask. We require near-full containment (70%)
            # before allowing context-pass absorption.
            context_overlap = has_context_candidate and overlap >= 0.70
            if iou >= merge_iou or overlap >= merge_overlap or context_overlap or candidate_seam_merge(candidate_a, candidate_b):
                uf.union(i, j)

    groups = {}
    for idx in range(len(candidates)):
        groups.setdefault(uf.find(idx), []).append(idx)

    merged = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            merged.append(candidates[group_indices[0]])
            continue

        group = [candidates[idx] for idx in group_indices]
        best = max(group, key=lambda c: (candidate_source_rank(c), float(c["score"]), int(c["area"])))
        mask = np.zeros_like(best["mask"], dtype=bool)
        for candidate in group:
            mask |= candidate["mask"].astype(bool)

        record = {k: v for k, v in best.items() if k != "mask"}
        record["mask"] = mask
        record["area"] = int(mask.sum())
        record["bbox"] = merged_bbox(mask)
        record["centroid"] = merged_centroid(mask)
        record["source_count"] = int(len(group))
        record["source_bboxes"] = [candidate["bbox"] for candidate in group]
        record["source_tiles"] = [candidate.get("tile") for candidate in group if "tile" in candidate]
        record["source_prompts"] = sorted({str(candidate.get("prompt", "")) for candidate in group})
        merged.append(record)

    return sorted(merged, key=lambda c: (candidate_source_rank(c), float(c["score"]), int(c["area"])), reverse=True)


def add_candidate_instances(candidates, image_shape, nms_iou, min_new_area_ratio):
    instance_mask = np.zeros(image_shape, dtype=np.int32)
    object_records = []
    accepted = []
    next_local_id = 1

    candidates = sorted(candidates, key=lambda c: (candidate_source_rank(c), float(c["score"]), int(c["area"])), reverse=True)

    for candidate in candidates:
        mask = candidate["mask"].astype(bool)
        area = int(mask.sum())
        if area == 0:
            continue

        if any(mask_iou(mask, prev) > nms_iou for prev in accepted):
            continue

        unclaimed = mask & (instance_mask == 0)
        if int(unclaimed.sum()) / area < min_new_area_ratio:
            continue

        instance_mask[unclaimed] = next_local_id
        record = {k: v for k, v in candidate.items() if k != "mask"}
        record["local_id"] = int(next_local_id)
        record["area"] = int(unclaimed.sum())
        object_records.append(record)
        accepted.append(unclaimed)
        next_local_id += 1

    return instance_mask, object_records


def paint_candidate_instances(candidates, image_shape):
    instance_mask = np.zeros(image_shape, dtype=np.int32)
    object_records = []
    next_local_id = 1

    candidates = sorted(candidates, key=lambda c: (candidate_source_rank(c), float(c["score"]), int(c["area"])), reverse=True)

    for candidate in candidates:
        mask = candidate["mask"].astype(bool)
        unclaimed = mask & (instance_mask == 0)
        area = int(unclaimed.sum())
        if area == 0:
            continue

        instance_mask[unclaimed] = next_local_id
        record = {k: v for k, v in candidate.items() if k != "mask"}
        record["local_id"] = int(next_local_id)
        record["area"] = area
        object_records.append(record)
        next_local_id += 1

    return instance_mask, object_records


def prompt_outputs_for_state(processor, state, prompt, device):
    use_cuda = device == "cuda"
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
        return processor.set_text_prompt(state=state, prompt=prompt)


def sam_components_for_crop(
    image,
    crop_window,
    keep_bounds,
    source,
    prompts,
    threshold,
    mask_threshold,
    min_area,
    tile_min_core_area_ratio,
    keep_bounds_filter,
    reject_truncated_components,
    edge_tolerance,
    collect_raw_candidates,
    processor,
    device,
):
    width, height = image.size
    x0, y0, x1, y1 = crop_window
    crop = image.crop((x0, y0, x1, y1))
    use_cuda = device == "cuda"

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
        state = processor.set_image(crop)

    raw_candidates = [] if collect_raw_candidates else None
    candidates = []

    for prompt_idx, prompt in enumerate(prompts):
        output = prompt_outputs_for_state(
            processor=processor,
            state=state,
            prompt=prompt,
            device=device,
        )

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            continue

        masks_np = masks.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy().reshape(-1)

        for idx in np.where(scores_np > threshold)[0]:
            crop_mask = np.squeeze(masks_np[idx])
            if crop_mask.ndim != 2:
                continue

            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y0:y1, x0:x1] = crop_mask > mask_threshold

            components = component_records(
                full_mask,
                sam_idx=idx,
                score=scores_np[idx],
                min_area=min_area,
            )
            for component in components:
                component["prompt"] = prompt
                component["prompt_idx"] = int(prompt_idx)
                component["source"] = source
                component["crop"] = [int(x0), int(y0), int(x1), int(y1)]
                component["tile"] = [int(v) for v in keep_bounds] if source == "tile" else None
                component["tile_core"] = [int(v) for v in keep_bounds]
                component["truncated_by_crop"] = bool(
                    source == "tile"
                    and touches_internal_crop_edge(
                        component,
                        crop_window=(x0, y0, x1, y1),
                        image_size=(width, height),
                        tolerance=edge_tolerance,
                    )
                )
                component["tile_core_kept"] = bool(
                    component_is_in_tile_core(
                        component,
                        keep_bounds,
                        tile_min_core_area_ratio,
                    )
                )
                if collect_raw_candidates:
                    raw_candidates.append(component)
                if keep_bounds_filter and not component["tile_core_kept"]:
                    continue
                if reject_truncated_components and component["truncated_by_crop"]:
                    continue
                candidates.append(component)

    return raw_candidates, candidates


def sam_components_for_image(
    image,
    prompts,
    threshold,
    mask_threshold,
    min_area,
    tile_size,
    tile_overlap,
    tile_core_margin,
    tile_min_core_area_ratio,
    tile_core_filter,
    full_image_pass,
    detail_pass,
    tile_context_margin,
    global_resize_pass,
    global_resize_max_side,
    reject_truncated_tile_components,
    tile_edge_tolerance,
    context_refine_truncated_tiles,
    context_refine_max_proposals,
    context_refine_max_side,
    context_refine_padding,
    context_refine_min_score,
    context_refine_min_overlap,
    context_refine_max_area_ratio,
    context_refine_reject_image_edge,
    context_refine_edge_tolerance,
    context_refine_cluster_fragments,
    context_refine_cluster_min_fragments,
    context_refine_cluster_max_gap_ratio,
    context_refine_cluster_min_axis_overlap,
    context_refine_cluster_min_score,
    box_refine_truncated_tiles,
    box_refine_max_proposals,
    box_refine_min_area,
    box_refine_padding,
    box_refine_min_overlap,
    collect_raw_candidates,
    processor,
    device,
):
    width, height = image.size
    raw_candidates = [] if collect_raw_candidates else None
    candidates = []

    if global_resize_pass and global_resize_max_side > 0 and max(width, height) > global_resize_max_side:
        scale = float(global_resize_max_side) / float(max(width, height))
        small_w = max(16, int(round(width * scale)))
        small_h = max(16, int(round(height * scale)))
        small_image = image.resize((small_w, small_h), Image.Resampling.BILINEAR)
        crop_raw, crop_candidates = sam_components_for_crop(
            image=small_image,
            crop_window=(0, 0, small_w, small_h),
            keep_bounds=(0, 0, small_w, small_h),
            source="global",
            prompts=prompts,
            threshold=threshold,
            mask_threshold=mask_threshold,
            min_area=max(1, int(round(min_area * scale * scale))),
            tile_min_core_area_ratio=0.0,
            keep_bounds_filter=False,
            reject_truncated_components=False,
            edge_tolerance=0,
            collect_raw_candidates=collect_raw_candidates,
            processor=processor,
            device=device,
        )

        def scale_candidate(candidate):
            full_mask = cv2.resize(
                candidate["mask"].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            record = {k: v for k, v in candidate.items() if k != "mask"}
            record["mask"] = full_mask
            record["area"] = int(full_mask.sum())
            record["bbox"] = merged_bbox(full_mask)
            record["centroid"] = merged_centroid(full_mask)
            record["global_scale"] = float(scale)
            return record

        if collect_raw_candidates:
            raw_candidates.extend(scale_candidate(candidate) for candidate in (crop_raw or []))
        candidates.extend(scale_candidate(candidate) for candidate in crop_candidates)

    if full_image_pass:
        crop_raw, crop_candidates = sam_components_for_crop(
            image=image,
            crop_window=(0, 0, width, height),
            keep_bounds=(0, 0, width, height),
            source="full",
            prompts=prompts,
            threshold=threshold,
            mask_threshold=mask_threshold,
            min_area=min_area,
            tile_min_core_area_ratio=0.0,
            keep_bounds_filter=False,
            reject_truncated_components=False,
            edge_tolerance=0,
            collect_raw_candidates=collect_raw_candidates,
            processor=processor,
            device=device,
        )
        if collect_raw_candidates:
            raw_candidates.extend(crop_raw or [])
        candidates.extend(crop_candidates)

    if detail_pass and tile_size > 0 and tile_size < max(width, height):
        if tile_context_margin < 0:
            tile_context_margin = int(round(tile_size * tile_overlap * 0.5))

        truncated_tile_proposals = []
        repair_tile_proposals = []
        for tile_window in crop_windows(width, height, tile_size, tile_overlap):
            core_bounds = tile_core_bounds(
                tile_window,
                (width, height),
                tile_size,
                tile_overlap,
                tile_core_margin,
            )
            crop_window = expand_window(
                tile_window,
                (width, height),
                tile_context_margin,
            )
            tile_collect_raw = bool(collect_raw_candidates or context_refine_cluster_fragments)
            crop_raw, crop_candidates = sam_components_for_crop(
                image=image,
                crop_window=crop_window,
                keep_bounds=core_bounds,
                source="tile",
                prompts=prompts,
                    threshold=threshold,
                mask_threshold=mask_threshold,
                min_area=min_area,
                tile_min_core_area_ratio=tile_min_core_area_ratio,
                keep_bounds_filter=True,
                reject_truncated_components=False,
                edge_tolerance=tile_edge_tolerance,
                collect_raw_candidates=tile_collect_raw,
                processor=processor,
                device=device,
            )
            if collect_raw_candidates:
                raw_candidates.extend(crop_raw or [])
            if crop_raw:
                repair_tile_proposals.extend(
                    candidate for candidate in crop_raw
                    if bool(candidate.get("truncated_by_crop", False))
                )
            for candidate in crop_candidates:
                if bool(candidate.get("truncated_by_crop", False)):
                    truncated_tile_proposals.append(candidate)
                    repair_tile_proposals.append(candidate)
                    if reject_truncated_tile_components:
                        continue
                candidates.append(candidate)

        cluster_proposals = []
        if context_refine_cluster_fragments and repair_tile_proposals:
            cluster_proposals = cluster_fragment_proposals(
                repair_tile_proposals,
                min_fragments=context_refine_cluster_min_fragments,
                max_gap_ratio=context_refine_cluster_max_gap_ratio,
                min_axis_overlap=context_refine_cluster_min_axis_overlap,
            )

        if context_refine_truncated_tiles and (cluster_proposals or truncated_tile_proposals):
            refined_candidates = refine_tile_fragments_with_context(
                image=image,
                proposals=cluster_proposals + truncated_tile_proposals,
                    mask_threshold=mask_threshold,
                min_area=min_area,
                processor=processor,
                device=device,
                max_proposals=context_refine_max_proposals,
                max_side=context_refine_max_side,
                padding_ratio=context_refine_padding,
                min_score=context_refine_min_score,
                min_overlap=context_refine_min_overlap,
                max_area_ratio=context_refine_max_area_ratio,
                reject_image_edge=context_refine_reject_image_edge,
                edge_tolerance=context_refine_edge_tolerance,
                cluster_min_score=context_refine_cluster_min_score,
            )
            if collect_raw_candidates:
                raw_candidates.extend(refined_candidates)
            candidates.extend(refined_candidates)

        if box_refine_truncated_tiles and truncated_tile_proposals:
            refined_candidates = refine_tile_fragments_with_boxes(
                image=image,
                proposals=truncated_tile_proposals,
                    threshold=threshold,
                mask_threshold=mask_threshold,
                min_area=min_area,
                processor=processor,
                device=device,
                max_proposals=box_refine_max_proposals,
                min_proposal_area=box_refine_min_area,
                padding_ratio=box_refine_padding,
                min_overlap=box_refine_min_overlap,
            )
            if collect_raw_candidates:
                raw_candidates.extend(refined_candidates)
            candidates.extend(refined_candidates)

        if reject_truncated_tile_components and truncated_tile_proposals:
            coverage_candidates = [
                candidate for candidate in candidates
                if candidate_source_rank(candidate) >= candidate_source_rank({"source": "box_refine"})
            ]
            for proposal in truncated_tile_proposals:
                covered = False
                for candidate in coverage_candidates:
                    if str(candidate.get("prompt", "")) != str(proposal.get("prompt", "")):
                        continue
                    _, overlap = candidate_overlap(candidate, proposal)
                    if overlap >= 0.35:
                        covered = True
                        break
                if not covered:
                    if context_refine_reject_image_edge and bbox_touches_image_edge(
                        proposal["bbox"],
                        image_size=(width, height),
                        tolerance=context_refine_edge_tolerance,
                    ):
                        continue
                    proposal["source"] = "tile_fragment"
                    candidates.append(proposal)

    return raw_candidates, candidates


def run_sam(
    frame_dir,
    mask_dir,
    threshold,
    mask_threshold,
    min_area,
    tile_size,
    tile_overlap,
    tile_core_margin,
    tile_min_core_area_ratio,
    tile_core_filter,
    full_image_pass,
    detail_pass,
    tile_context_margin,
    global_resize_pass,
    global_resize_max_side,
    reject_truncated_tile_components,
    tile_edge_tolerance,
    context_refine_truncated_tiles,
    context_refine_max_proposals,
    context_refine_max_side,
    context_refine_padding,
    context_refine_min_score,
    context_refine_min_overlap,
    context_refine_max_area_ratio,
    context_refine_reject_image_edge,
    context_refine_edge_tolerance,
    context_refine_cluster_fragments,
    context_refine_cluster_min_fragments,
    context_refine_cluster_max_gap_ratio,
    context_refine_cluster_min_axis_overlap,
    context_refine_cluster_min_score,
    box_refine_truncated_tiles,
    box_refine_max_proposals,
    box_refine_min_area,
    box_refine_padding,
    box_refine_min_overlap,
    merge_candidates,
    merge_iou,
    merge_overlap,
    nms_iou,
    min_new_area_ratio,
    artifact_fragment_filter,
    artifact_fragment_contained_area_ratio,
    artifact_fragment_edge_area_ratio,
    artifact_fragment_bbox_margin_ratio,
    artifact_fragment_min_reference_area,
    artifact_fragment_edge_sources,
    save_premerge_mask,
    prompts,
    processor,
    device,
):
    fnames = sorted(
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if len(fnames) == 0:
        raise RuntimeError(f"No frames found in {frame_dir}")

    os.makedirs(mask_dir, exist_ok=True)

    for fname in progress_iter(fnames, desc="Running SAM", unit="frame"):
        frame_path = os.path.join(frame_dir, fname)
        image = Image.open(frame_path).convert("RGB")
        raw_candidates, candidates = sam_components_for_image(
            image=image,
            prompts=prompts,
            threshold=threshold,
            mask_threshold=mask_threshold,
            min_area=min_area,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_core_margin=tile_core_margin,
            tile_min_core_area_ratio=tile_min_core_area_ratio,
            tile_core_filter=tile_core_filter,
            full_image_pass=full_image_pass,
            detail_pass=detail_pass,
            tile_context_margin=tile_context_margin,
            global_resize_pass=global_resize_pass,
            global_resize_max_side=global_resize_max_side,
            reject_truncated_tile_components=reject_truncated_tile_components,
            tile_edge_tolerance=tile_edge_tolerance,
            context_refine_truncated_tiles=context_refine_truncated_tiles,
            context_refine_max_proposals=context_refine_max_proposals,
            context_refine_max_side=context_refine_max_side,
            context_refine_padding=context_refine_padding,
            context_refine_min_score=context_refine_min_score,
            context_refine_min_overlap=context_refine_min_overlap,
            context_refine_max_area_ratio=context_refine_max_area_ratio,
            context_refine_reject_image_edge=context_refine_reject_image_edge,
            context_refine_edge_tolerance=context_refine_edge_tolerance,
            context_refine_cluster_fragments=context_refine_cluster_fragments,
            context_refine_cluster_min_fragments=context_refine_cluster_min_fragments,
            context_refine_cluster_max_gap_ratio=context_refine_cluster_max_gap_ratio,
            context_refine_cluster_min_axis_overlap=context_refine_cluster_min_axis_overlap,
            context_refine_cluster_min_score=context_refine_cluster_min_score,
            box_refine_truncated_tiles=box_refine_truncated_tiles,
            box_refine_max_proposals=box_refine_max_proposals,
            box_refine_min_area=box_refine_min_area,
            box_refine_padding=box_refine_padding,
            box_refine_min_overlap=box_refine_min_overlap,
            collect_raw_candidates=save_premerge_mask,
            processor=processor,
            device=device,
        )

        premerge_mask = None
        premerge_records = []
        raw_tile_mask = None
        raw_tile_records = []
        if save_premerge_mask:
            raw_tile_mask, raw_tile_records = paint_candidate_instances(
                raw_candidates or [],
                image_shape=(image.height, image.width),
            )
            premerge_mask, premerge_records = add_candidate_instances(
                candidates,
                image_shape=(image.height, image.width),
                nms_iou=nms_iou,
                min_new_area_ratio=min_new_area_ratio,
            )

        if merge_candidates:
            candidates = merge_candidate_instances(
                candidates,
                merge_iou=merge_iou,
                merge_overlap=merge_overlap,
            )
            merged_fragments = [
                candidate for candidate in candidates
                if str(candidate.get("source", "")) == "tile_fragment"
                or bool(candidate.get("truncated_by_crop", False))
            ]
            if context_refine_truncated_tiles and merged_fragments:
                repaired_candidates = refine_tile_fragments_with_context(
                    image=image,
                    proposals=merged_fragments,
                            mask_threshold=mask_threshold,
                    min_area=min_area,
                    processor=processor,
                    device=device,
                    max_proposals=context_refine_max_proposals,
                    max_side=context_refine_max_side,
                    padding_ratio=context_refine_padding,
                    min_score=context_refine_min_score,
                    min_overlap=context_refine_min_overlap,
                    max_area_ratio=context_refine_max_area_ratio,
                    reject_image_edge=context_refine_reject_image_edge,
                    edge_tolerance=context_refine_edge_tolerance,
                    cluster_min_score=context_refine_cluster_min_score,
                )
                if repaired_candidates:
                    print(
                        f"[INFO] Context-refined {len(repaired_candidates)} merged tile fragment(s) in {fname}",
                        flush=True,
                    )
                    candidates.extend(repaired_candidates)
                    candidates = merge_candidate_instances(
                        candidates,
                        merge_iou=merge_iou,
                        merge_overlap=merge_overlap,
                    )

        candidates, removed_artifacts = filter_duplicate_artifact_candidates(
            candidates,
            image_shape=(image.height, image.width),
            enabled=artifact_fragment_filter,
            contained_area_ratio=artifact_fragment_contained_area_ratio,
            edge_area_ratio=artifact_fragment_edge_area_ratio,
            bbox_margin_ratio=artifact_fragment_bbox_margin_ratio,
            min_reference_area=artifact_fragment_min_reference_area,
            edge_sources=artifact_fragment_edge_sources,
        )
        if removed_artifacts:
            summary = {}
            for item in removed_artifacts:
                summary[item["reason"]] = summary.get(item["reason"], 0) + 1
            print(
                f"[INFO] Removed {len(removed_artifacts)} duplicate SAM fragment(s) in {fname}: {summary}",
                flush=True,
            )

        local_instance_mask, object_records = add_candidate_instances(
            candidates,
            image_shape=(image.height, image.width),
            nms_iou=nms_iou,
            min_new_area_ratio=min_new_area_ratio,
        )

        final_fragments = []
        if context_refine_truncated_tiles:
            for record in object_records:
                if (
                    str(record.get("source", "")) != "tile_fragment"
                    and not bool(record.get("truncated_by_crop", False))
                ):
                    continue
                local_id = int(record.get("local_id", -1))
                if local_id <= 0:
                    continue
                proposal = {k: v for k, v in record.items() if k != "mask"}
                proposal["mask"] = local_instance_mask == local_id
                proposal["area"] = int(proposal["mask"].sum())
                proposal["bbox"] = merged_bbox(proposal["mask"])
                proposal["centroid"] = merged_centroid(proposal["mask"])
                final_fragments.append(proposal)
        if final_fragments:
            print(
                f"[INFO] Retrying {len(final_fragments)} final tile fragment(s) with expanded context in {fname}",
                flush=True,
            )
            repaired_candidates = refine_tile_fragments_with_context(
                image=image,
                proposals=final_fragments,
                    mask_threshold=mask_threshold,
                min_area=min_area,
                processor=processor,
                device=device,
                max_proposals=context_refine_max_proposals,
                max_side=context_refine_max_side,
                padding_ratio=context_refine_padding,
                min_score=context_refine_min_score,
                min_overlap=context_refine_min_overlap,
                max_area_ratio=context_refine_max_area_ratio,
                reject_image_edge=context_refine_reject_image_edge,
                edge_tolerance=context_refine_edge_tolerance,
                cluster_min_score=context_refine_cluster_min_score,
            )
            if repaired_candidates:
                print(
                    f"[INFO] Context-refined {len(repaired_candidates)} final tile fragment(s) in {fname}",
                    flush=True,
                )
                candidates.extend(repaired_candidates)
                if merge_candidates:
                    candidates = merge_candidate_instances(
                        candidates,
                        merge_iou=merge_iou,
                        merge_overlap=merge_overlap,
                    )
                candidates, removed_artifacts = filter_duplicate_artifact_candidates(
                    candidates,
                    image_shape=(image.height, image.width),
                    enabled=artifact_fragment_filter,
                    contained_area_ratio=artifact_fragment_contained_area_ratio,
                    edge_area_ratio=artifact_fragment_edge_area_ratio,
                    bbox_margin_ratio=artifact_fragment_bbox_margin_ratio,
                    min_reference_area=artifact_fragment_min_reference_area,
                    edge_sources=artifact_fragment_edge_sources,
                )
                if removed_artifacts:
                    summary = {}
                    for item in removed_artifacts:
                        summary[item["reason"]] = summary.get(item["reason"], 0) + 1
                    print(
                        f"[INFO] Removed {len(removed_artifacts)} duplicate SAM fragment(s) after repair in {fname}: {summary}",
                        flush=True,
                    )
                local_instance_mask, object_records = add_candidate_instances(
                    candidates,
                    image_shape=(image.height, image.width),
                    nms_iou=nms_iou,
                    min_new_area_ratio=min_new_area_ratio,
                )

        if len(object_records) == 0:
            print(f"[WARN] No valid components saved for {fname}", flush=True)
            continue

        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(mask_dir, f"{stem}.npz")

        payload = {
            "instance_mask": local_instance_mask.astype(np.int32),
            "local_instance_mask": local_instance_mask.astype(np.int32),
            "objects": np.array(object_records, dtype=object),
        }
        if save_premerge_mask and premerge_mask is not None:
            payload["raw_tile_instance_mask"] = raw_tile_mask.astype(np.int32)
            payload["raw_tile_objects"] = np.array(raw_tile_records, dtype=object)
            payload["premerge_instance_mask"] = premerge_mask.astype(np.int32)
            payload["premerge_objects"] = np.array(premerge_records, dtype=object)

        np.savez_compressed(out_path, **payload)


def main():
    args = parse_args()

    video_path = os.path.expanduser(args.video_path)

    if not args.no_extract:
        if os.path.exists(args.frame_dir):
            shutil.rmtree(args.frame_dir)
        os.makedirs(args.frame_dir, exist_ok=True)

    if os.path.exists(args.mask_dir):
        shutil.rmtree(args.mask_dir)
    os.makedirs(args.mask_dir, exist_ok=True)

    if not args.no_extract:
        extract_frames(video_path, args.frame_dir, args.fps)

    try:
        sam3_version = importlib.metadata.version("sam3")
        print(f"[INFO] sam3 package version: {sam3_version}")
        if version_tuple(sam3_version) < (0, 1, 3):
            print(
                "[WARN] sam3 is older than the latest 0.1.3 package; upgrading may unlock SAM 3.1 speedups.",
                flush=True,
            )
    except importlib.metadata.PackageNotFoundError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = build_sam3_image_model(bpe_path=str(SAM3_BPE_PATH)).to(device).eval()
    processor = Sam3Processor(model)
    prompts = expand_prompts(args.text_prompt, args.extra_prompts)
    if not prompts:
        raise ValueError("At least one SAM prompt is required")
    print(f"[INFO] SAM prompts: {', '.join(prompts)}")

    full_image_pass = bool(getattr(args, "full_image_pass", True))
    detail_pass = bool(getattr(args, "detail_pass", True))
    tile_context_margin = getattr(args, "tile_context_margin", -1)
    pass_names = []
    if full_image_pass:
        pass_names.append("whole-frame")
    if detail_pass:
        pass_names.append("padded-tile recall")
    print(f"[INFO] SAM pass mode: {' + '.join(pass_names) if pass_names else 'disabled'}")

    run_sam(
        frame_dir=args.frame_dir,
        mask_dir=args.mask_dir,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        min_area=args.min_area,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_core_margin=args.tile_core_margin,
        tile_min_core_area_ratio=args.tile_min_core_area_ratio,
        tile_core_filter=args.tile_core_filter,
        full_image_pass=full_image_pass,
        detail_pass=detail_pass,
        tile_context_margin=tile_context_margin,
        global_resize_pass=bool(getattr(args, "global_resize_pass", True)),
        global_resize_max_side=int(getattr(args, "global_resize_max_side", 1536)),
        reject_truncated_tile_components=bool(getattr(args, "reject_truncated_tile_components", True)),
        tile_edge_tolerance=int(getattr(args, "tile_edge_tolerance", 2)),
        context_refine_truncated_tiles=bool(getattr(args, "context_refine_truncated_tiles", True)),
        context_refine_max_proposals=int(getattr(args, "context_refine_max_proposals", 24)),
        context_refine_max_side=int(getattr(args, "context_refine_max_side", 1536)),
        context_refine_padding=float(getattr(args, "context_refine_padding", 1.0)),
        context_refine_min_score=float(getattr(args, "context_refine_min_score", 0.05)),
        context_refine_min_overlap=float(getattr(args, "context_refine_min_overlap", 0.45)),
        context_refine_max_area_ratio=float(getattr(args, "context_refine_max_area_ratio", 8.0)),
        context_refine_reject_image_edge=bool(getattr(args, "context_refine_reject_image_edge", True)),
        context_refine_edge_tolerance=int(getattr(args, "context_refine_edge_tolerance", 2)),
        context_refine_cluster_fragments=bool(getattr(args, "context_refine_cluster_fragments", True)),
        context_refine_cluster_min_fragments=int(getattr(args, "context_refine_cluster_min_fragments", 2)),
        context_refine_cluster_max_gap_ratio=float(getattr(args, "context_refine_cluster_max_gap_ratio", 0.8)),
        context_refine_cluster_min_axis_overlap=float(getattr(args, "context_refine_cluster_min_axis_overlap", 0.12)),
        context_refine_cluster_min_score=float(getattr(args, "context_refine_cluster_min_score", 0.02)),
        box_refine_truncated_tiles=bool(getattr(args, "box_refine_truncated_tiles", False)),
        box_refine_max_proposals=int(getattr(args, "box_refine_max_proposals", 24)),
        box_refine_min_area=int(getattr(args, "box_refine_min_area", 5000)),
        box_refine_padding=float(getattr(args, "box_refine_padding", 0.45)),
        box_refine_min_overlap=float(getattr(args, "box_refine_min_overlap", 0.02)),
        merge_candidates=args.merge_candidates,
        merge_iou=args.merge_iou,
        merge_overlap=args.merge_overlap,
        nms_iou=args.nms_iou,
        min_new_area_ratio=args.min_new_area_ratio,
        artifact_fragment_filter=bool(getattr(args, "artifact_fragment_filter", True)),
        artifact_fragment_contained_area_ratio=float(getattr(args, "artifact_fragment_contained_area_ratio", 0.02)),
        artifact_fragment_edge_area_ratio=float(getattr(args, "artifact_fragment_edge_area_ratio", 0.08)),
        artifact_fragment_bbox_margin_ratio=float(getattr(args, "artifact_fragment_bbox_margin_ratio", 0.04)),
        artifact_fragment_min_reference_area=int(getattr(args, "artifact_fragment_min_reference_area", 20000)),
        artifact_fragment_edge_sources=getattr(args, "artifact_fragment_edge_sources", ["tile", "tile_fragment"]),
        save_premerge_mask=args.save_premerge_mask,
        prompts=prompts,
        processor=processor,
        device=device,
    )

    print(f"[DONE] Saved masks to: {args.mask_dir}")


if __name__ == "__main__":
    main()

import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from video and run SAM3 instance-style segmentation"
    )

    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt, e.g. 'orange fruit'")
    parser.add_argument("--frame-dir", type=str, default="out/frames", help="Output directory for extracted frames")
    parser.add_argument("--mask-dir", type=str, default="out/masks", help="Output directory for masks")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    parser.add_argument("--threshold", type=float, default=0.3, help="SAM score threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Pixel threshold for mask logits/probs")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum connected-component area")
    parser.add_argument("--no-extract", action="store_true", help="Skip frame extraction and only run SAM on existing frames")

    return parser.parse_args()


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

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if rotation in rotations:
                frame = cv2.rotate(frame, rotations[rotation])

            if frame_idx % frame_interval == 0:
                out_path = os.path.join(frame_dir, f"frame_{saved_idx:04d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    tqdm.write(f"[INFO] Extracted {saved_idx} frames")


def add_connected_components_to_instance_mask(
    instance_mask,
    binary_mask,
    next_local_id,
    sam_idx,
    score,
    min_area,
):
    """
    Splits one binary SAM mask into separated connected components.

    Connected components turn that into:
        id 1 = orange A
        id 2 = orange B
        id 3 = orange C
    when the blobs are disconnected.
    """
    binary_mask = np.squeeze(binary_mask).astype(np.uint8)

    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {binary_mask.shape}")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    records = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        component = labels == label_id

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        instance_mask[component] = next_local_id

        records.append({
            "local_id": int(next_local_id),
            "sam_idx": int(sam_idx),
            "score": float(score),
            "area": area,
            "bbox": [x, y, x + w - 1, y + h - 1],
        })

        next_local_id += 1

    return instance_mask, records, next_local_id


def run_sam(
    frame_dir,
    mask_dir,
    text_prompt,
    threshold,
    mask_threshold,
    min_area,
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

    use_cuda = device == "cuda"

    for fname in tqdm(fnames, desc="Running SAM", unit="frame"):
        frame_path = os.path.join(frame_dir, fname)
        image = Image.open(frame_path).convert("RGB")

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=text_prompt)

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            tqdm.write(f"[WARN] No masks for {fname}")
            continue

        masks_np = masks.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy().reshape(-1)

        valid = np.where(scores_np > threshold)[0]

        if len(valid) == 0:
            tqdm.write(f"[WARN] No masks above threshold for {fname}")
            continue

        h, w = masks_np.shape[-2:]
        local_instance_mask = np.zeros((h, w), dtype=np.int32)

        object_records = []
        next_local_id = 1

        for idx in valid:
            m = np.squeeze(masks_np[idx])

            if m.ndim != 2:
                tqdm.write(f"[WARN] Unexpected mask shape for {fname}, idx={idx}: {m.shape}")
                continue

            binary = m > mask_threshold

            if int(binary.sum()) < min_area:
                continue

            local_instance_mask, records, next_local_id = add_connected_components_to_instance_mask(
                instance_mask=local_instance_mask,
                binary_mask=binary,
                next_local_id=next_local_id,
                sam_idx=idx,
                score=scores_np[idx],
                min_area=min_area,
            )

            object_records.extend(records)

        if len(object_records) == 0:
            tqdm.write(f"[WARN] No valid components saved for {fname}")
            continue

        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(mask_dir, f"{stem}.npz")

        np.savez_compressed(
            out_path,
            instance_mask=local_instance_mask.astype(np.int32),
            local_instance_mask=local_instance_mask.astype(np.int32),
            objects=np.array(object_records, dtype=object),
        )


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = build_sam3_image_model().to(device).eval()
    processor = Sam3Processor(model)

    if not args.no_extract:
        extract_frames(video_path, args.frame_dir, args.fps)

    run_sam(
        frame_dir=args.frame_dir,
        mask_dir=args.mask_dir,
        text_prompt=args.text_prompt,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        min_area=args.min_area,
        processor=processor,
        device=device,
    )

    print(f"[DONE] Saved masks to: {args.mask_dir}")


if __name__ == "__main__":
    main()
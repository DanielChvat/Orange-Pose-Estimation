import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from video and run SAM3 segmentation")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--text_prompt", type=str, help="Text prompt for segmentation (e.g. 'orange fruit')")
    parser.add_argument("--frame-dir", type=str, default="out/frames", help="Output directory for extracted frames")
    parser.add_argument("--mask-dir", type=str, default="out/masks", help="Output directory for masks")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample (default: 2)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for valid masks (default: 0.3)")
    return parser.parse_args()


def extract_frames(video_path, frame_dir, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)

    ROTATIONS = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    frame_idx = 0
    saved_idx = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rotation in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[rotation])

            if frame_idx % frame_interval == 0:
                cv2.imwrite(os.path.join(frame_dir, f"frame_{saved_idx:04d}.jpg"), frame)
                saved_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    tqdm.write(f"[INFO] Extracted {saved_idx} frames")


def run_sam(frame_dir, mask_dir, text_prompt, threshold, processor):
    fnames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))

    for fname in tqdm(fnames, desc="Running SAM", unit="frame"):
        image = Image.open(os.path.join(frame_dir, fname)).convert("RGB")

        with torch.inference_mode(), torch.cuda.amp.autocast():
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=text_prompt)

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            tqdm.write(f"[WARN] No masks for {fname}")
            continue

        scores_np = scores.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        valid = np.where(scores_np > threshold)[0]

        if len(valid) == 0:
            tqdm.write(f"[WARN] No masks above threshold for {fname}")
            continue

        combined = np.zeros_like(masks_np[0])
        for idx in valid:
            m = masks_np[idx]
            if m.ndim == 3:
                m = m.squeeze(0)
            combined = np.logical_or(combined, m)

        combined = combined.astype(np.uint8)
        np.savez_compressed(os.path.join(mask_dir, fname.replace(".jpg", ".npz")), mask=combined)


def main():
    args = parse_args()

    video_path = os.path.expanduser(args.video_path)
    os.makedirs(args.frame_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model().to(device).eval()
    processor = Sam3Processor(model)

    extract_frames(video_path, args.frame_dir, args.fps)
    run_sam(args.frame_dir, args.mask_dir, args.text_prompt, args.threshold, processor)


if __name__ == "__main__":
    main()
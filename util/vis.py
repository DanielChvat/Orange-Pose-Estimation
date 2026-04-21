import argparse
import os
import subprocess
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate side-by-side comparisons of frames, masks and depthmaps")
    parser.add_argument("--frame-dir", type=str, default="out/frames")
    parser.add_argument("--mask-dir", type=str, default="out/masks")
    parser.add_argument("--depth-dir", type=str, default="out/dust3r/depths")
    parser.add_argument("--out-dir", type=str, default="vis")
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def colorize_depth(depth):
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = 255 - (depth_norm * 255).astype(np.uint8)  # flip
    return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)


def frames_to_video(frame_dir, video_path, fps):
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(frame_dir, "*.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path,
    ], check=True)


def main():
    args = parse_args()

    masks_img_dir = os.path.join(args.out_dir, "masks")
    depth_img_dir = os.path.join(args.out_dir, "depths")
    video_dir = os.path.join(args.out_dir, "videos")

    for d in [masks_img_dir, depth_img_dir, video_dir]:
        os.makedirs(d, exist_ok=True)

    fnames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".jpg"))
    has_depth = os.path.isdir(args.depth_dir)

    for fname in tqdm(fnames, desc="Processing", unit="frame"):
        mask_path = os.path.join(args.mask_dir, fname.replace(".jpg", ".npz"))
        if not os.path.exists(mask_path):
            tqdm.write(f"[WARN] No mask for {fname}")
            continue

        image = cv2.cvtColor(np.array(Image.open(os.path.join(args.frame_dir, fname)).convert("RGB")), cv2.COLOR_RGB2BGR)

        mask = np.load(mask_path)["mask"]
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = (mask > 0.5).astype(np.uint8)

        masked = image.copy()
        masked[mask == 0] = 0

        cv2.imwrite(os.path.join(masks_img_dir, fname), np.hstack((image, masked)))

        if has_depth:
            stem = os.path.splitext(fname)[0]
            depth_path = os.path.join(args.depth_dir, f"{stem}_depth.npz")
            if not os.path.exists(depth_path):
                tqdm.write(f"[WARN] No depth for {fname}")
                continue

            depth = np.load(depth_path)["depth"]
            depth_vis = colorize_depth(depth)
            depth_vis = cv2.resize(depth_vis, (image.shape[1], image.shape[0]))

            cv2.imwrite(os.path.join(depth_img_dir, fname), np.hstack((image, masked, depth_vis)))

    print("[INFO] Rendering video...")
    frames_to_video(depth_img_dir, os.path.join(video_dir, "vis.mp4"), args.fps)
    print(f"[DONE] Videos saved to {video_dir}")


if __name__ == "__main__":
    main()
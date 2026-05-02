import argparse
import os
import copy
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import trimesh
import sys
from pathlib import Path
torch.serialization.add_safe_globals([argparse.Namespace])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dust3r"))

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model import AsymmetricCroCo3DStereo

import json


def parse_args():
    parser = argparse.ArgumentParser(description="Run DUSt3R inference on extracted frames")
    parser.add_argument("--frame-dir", type=str, default="out/frames", help="Input frames directory")
    parser.add_argument("--out-dir", type=str, default="out/dust3r", help="Output directory")
    parser.add_argument("--weights", type=str, required=True, help="Path to DUSt3R model weights")
    parser.add_argument("--image-size", type=int, default=512, choices=[512, 224])
    parser.add_argument("--niter", type=int, default=300, help="Global alignment iterations")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--min-conf-thr", type=float, default=3.0)
    parser.add_argument("--scenegraph", type=str, default="complete", choices=["complete", "swin", "oneref"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--as-pointcloud", action="store_true")
    return parser.parse_args()


def load_model(weights, device):
    model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(device).eval()
    return model


def _image_size(path):
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def _dust3r_resize_crop(original_w, original_h, size, square_ok=False, patch_size=16):
    long_edge_size = round(size * max(original_w / original_h, original_h / original_w)) if size == 224 else size
    scale = long_edge_size / max(original_w, original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    cx, cy = resized_w // 2, resized_h // 2

    if size == 224:
        half = min(cx, cy)
        crop_left, crop_top = cx - half, cy - half
        crop_right, crop_bottom = cx + half, cy + half
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not square_ok and resized_w == resized_h:
            halfh = 3 * halfw / 4
        crop_left, crop_top = cx - halfw, cy - halfh
        crop_right, crop_bottom = cx + halfw, cy + halfh

    return {
        "resized_width": int(resized_w),
        "resized_height": int(resized_h),
        "crop_left": float(crop_left),
        "crop_top": float(crop_top),
        "crop_right": float(crop_right),
        "crop_bottom": float(crop_bottom),
    }


def save_camera_poses(scene, fnames, filepaths, out_dir, image_size):
    cams2world = to_numpy(scene.get_im_poses().cpu())
    focals = to_numpy(scene.get_focals().cpu())
    imgs = to_numpy(scene.imgs)

    records = []
    for i, (fname, path, c2w, focal, img) in enumerate(zip(fnames, filepaths, cams2world, focals, imgs)):
        r = Rotation.from_matrix(c2w[:3, :3])
        quat = r.as_quat()
        t = c2w[:3, 3]
        original_w, original_h = _image_size(path)
        processed_h, processed_w = img.shape[:2]
        image_meta = _dust3r_resize_crop(original_w, original_h, image_size)
        records.append({
            "frame_idx": i,
            "filename": fname,
            "tx": float(t[0]), "ty": float(t[1]), "tz": float(t[2]),
            "qx": float(quat[0]), "qy": float(quat[1]), "qz": float(quat[2]), "qw": float(quat[3]),
            "focal": float(focal.item()),
            "processed_width": int(processed_w),
            "processed_height": int(processed_h),
            "original_width": int(original_w),
            "original_height": int(original_h),
            **image_meta,
        })

    with open(os.path.join(out_dir, "camera_poses.json"), "w") as f:
        json.dump(records, f, indent=2)


def save_depthmaps(scene, fnames, out_dir):
    depths_dir = os.path.join(out_dir, "depths")
    os.makedirs(depths_dir, exist_ok=True)
    depths = to_numpy(scene.get_depthmaps())
    for fname, depth in tqdm(zip(fnames, depths), desc="Saving depthmaps", total=len(fnames)):
        stem = Path(fname).stem
        np.savez_compressed(os.path.join(depths_dir, f"{stem}_depth.npz"), depth=depth)


def save_pointcloud(scene, out_dir, as_pointcloud, min_conf_thr):
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = to_numpy(scene.get_masks())
    imgs = to_numpy(scene.imgs)
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())

    trimesh_scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, masks)])
        col = np.concatenate([p[m] for p, m in zip(imgs, masks)])
        trimesh_scene.add_geometry(trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3)))
    else:
        meshes = [pts3d_to_trimesh(imgs[i], pts3d[i], masks[i]) for i in range(len(imgs))]
        trimesh_scene.add_geometry(trimesh.Trimesh(**cat_meshes(meshes)))

    for i, (pose_c2w, focal, img) in enumerate(zip(cams2world, focals, imgs)):
        add_scene_cam(trimesh_scene, pose_c2w, CAM_COLORS[i % len(CAM_COLORS)],
                      img, focal, imsize=img.shape[1::-1], screen_width=0.05)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    trimesh_scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

    outfile = os.path.join(out_dir, "scene.glb")
    trimesh_scene.export(file_obj=outfile)
    print(f"[INFO] Saved pointcloud to {outfile}")


def save_pts3d(scene, fnames, out_dir):
    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())
    for fname, pts, mask in tqdm(zip(fnames, pts3d, masks), desc="Saving pts3d", total=len(fnames)):
        stem = Path(fname).stem
        np.savez_compressed(os.path.join(out_dir, f"{stem}_pts3d.npz"), pts3d=pts, mask=mask)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fnames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".jpg"))
    filepaths = [os.path.join(args.frame_dir, f) for f in fnames]
    print(f"[INFO] Found {len(fnames)} frames")

    print("[INFO] Loading model...")
    model = load_model(args.weights, args.device)

    print("[INFO] Loading images...")
    imgs = load_images(filepaths, size=args.image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    print("[INFO] Running inference...")
    pairs = make_pairs(imgs, scene_graph=args.scenegraph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=1)

    print("[INFO] Running global alignment...")
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=args.device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init='mst', niter=args.niter, schedule=args.schedule, lr=0.01)

    print("[INFO] Saving outputs...")
    save_camera_poses(scene, fnames, filepaths, args.out_dir, args.image_size)
    save_depthmaps(scene, fnames, args.out_dir)
    save_pointcloud(scene, args.out_dir, args.as_pointcloud, args.min_conf_thr)
    save_pts3d(scene, fnames, args.out_dir)

    print(f"[DONE] All outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()

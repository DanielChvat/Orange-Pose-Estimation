# Pipeline intermediate and final outputs.

- `frames/` — extracted frames from the input video (output of `get_masks.py`)
- `masks/` — SAM3 segmentation masks per frame as `.npz` files (output of `get_masks.py`)
- `dust3r/` — DUSt3R reconstruction outputs (output of `generate_pointcloud.py`)
  - `camera_poses.json` — per-frame camera pose (translation, quaternion, focal length)
  - `depths/` — per-frame depthmaps as `.npz` files
  - `scene.glb` — 3D pointcloud/mesh with cameras
  - `frame_XXXX_pts3d.npz` — raw 3D points and confidence mask per frame
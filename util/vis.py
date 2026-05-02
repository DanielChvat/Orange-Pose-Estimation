import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Render full-vs-segmented GLBs from Open3D keyframes")
    parser.add_argument("--glb", type=str, default="out/dust3r/scene.glb")
    parser.add_argument("--segmented-glb", type=str, default="out/voted_pts/orange_pts3d.glb")
    parser.add_argument("--keyframes", type=str, default="vis/open3d_keyframes.json")
    parser.add_argument("--projected-video", action="store_true")
    parser.add_argument("--frame-dir", type=str, default="out/frames")
    parser.add_argument("--orange-pts", type=str, default="out/voted_pts/orange_pts3d.npz")
    parser.add_argument("--poses", type=str, default="out/dust3r/camera_poses.json")
    parser.add_argument("--pts-dir", type=str, default="out/dust3r")
    parser.add_argument("--depth-dir", type=str, default="out/dust3r/depths")
    parser.add_argument("--out-dir", type=str, default="vis")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=120)
    return parser.parse_args()


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


def processed_hw_from_pose(pose):
    if "processed_height" in pose and "processed_width" in pose:
        return int(pose["processed_height"]), int(pose["processed_width"])
    return None


def project_pts_to_frame(pts3d, pose, image_hw, processed_hw=None, depth=None, depth_abs_tol=0.03, depth_rel_tol=0.05):
    image_h, image_w = image_hw
    if processed_hw is None:
        processed_hw = processed_hw_from_pose(pose)
    proc_h, proc_w = processed_hw if processed_hw is not None else (image_h, image_w)

    rotation = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]]).as_matrix()
    translation = np.array([pose["tx"], pose["ty"], pose["tz"]], dtype=np.float64)

    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    w2c = np.linalg.inv(c2w)

    p_cam = w2c[:3, :3] @ pts3d.T + w2c[:3, 3:4]
    valid = p_cam[2] > 0

    u_proc = pose["focal"] * p_cam[0] / p_cam[2] + proc_w / 2.0
    v_proc = pose["focal"] * p_cam[1] / p_cam[2] + proc_h / 2.0
    u_int = u_proc.astype(int)
    v_int = v_proc.astype(int)

    in_bounds = valid & (u_int >= 0) & (u_int < proc_w) & (v_int >= 0) & (v_int < proc_h)

    if depth is not None:
        if depth.shape != (proc_h, proc_w):
            depth = cv2.resize(depth.astype(np.float32), (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        depth_vals = np.zeros_like(p_cam[2])
        depth_vals[in_bounds] = depth[v_int[in_bounds], u_int[in_bounds]]
        depth_tol = np.maximum(depth_abs_tol, depth_rel_tol * depth_vals)
        in_bounds &= np.abs(p_cam[2] - depth_vals) <= depth_tol

    resized_w = pose.get("resized_width", proc_w)
    resized_h = pose.get("resized_height", proc_h)
    crop_left = pose.get("crop_left", 0.0)
    crop_top = pose.get("crop_top", 0.0)

    u = ((u_proc + crop_left) * (image_w / resized_w)).astype(int)
    v = ((v_proc + crop_top) * (image_h / resized_h)).astype(int)

    image_bounds = in_bounds & (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    return u[image_bounds], v[image_bounds]


def draw_points_on_black(image_shape, u, v):
    overlay = np.zeros(image_shape, dtype=np.uint8)
    overlay[v, u] = (0, 140, 255)
    return overlay


def render_projected_video(frame_dir, orange_pts_path, poses_path, pts_dir, depth_dir, out_dir, fps):
    if not os.path.exists(orange_pts_path):
        raise FileNotFoundError(f"Missing segmented points: {orange_pts_path}")
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Missing camera poses: {poses_path}")

    pts3d = np.load(orange_pts_path)["pts3d"]
    with open(poses_path) as f:
        poses = {pose["filename"]: pose for pose in json.load(f)}

    projected_dir = os.path.join(out_dir, "projected_vis_frames")
    video_dir = os.path.join(out_dir, "videos")
    if os.path.exists(projected_dir):
        shutil.rmtree(projected_dir)
    os.makedirs(projected_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    frame_names = sorted(name for name in os.listdir(frame_dir) if name.lower().endswith((".jpg", ".jpeg", ".png")))
    for name in tqdm(frame_names, desc="Projecting", unit="frame"):
        if name not in poses:
            continue

        frame = cv2.imread(os.path.join(frame_dir, name), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        pose = poses[name]
        processed_hw = processed_hw_from_pose(pose)
        if processed_hw is None:
            pts_path = os.path.join(pts_dir, f"{os.path.splitext(name)[0]}_pts3d.npz")
            if os.path.exists(pts_path):
                processed_hw = np.load(pts_path)["pts3d"].shape[:2]

        depth = None
        depth_path = os.path.join(depth_dir, f"{os.path.splitext(name)[0]}_depth.npz")
        if os.path.exists(depth_path):
            depth = np.load(depth_path)["depth"]

        u, v = project_pts_to_frame(pts3d, pose, frame.shape[:2], processed_hw=processed_hw, depth=depth)
        projected = draw_points_on_black(frame.shape, u, v)
        combined = np.hstack((frame, projected))
        cv2.imwrite(os.path.join(projected_dir, f"{os.path.splitext(name)[0]}.jpg"), combined)

    video_path = os.path.join(video_dir, "projected_vis.mp4")
    frames_to_video(projected_dir, video_path, fps)
    return video_path


def iter_geometries(loaded):
    if isinstance(loaded, trimesh.Scene):
        yield from loaded.geometry.values()
    else:
        yield loaded


def build_point_scene(glb_path, bg_color=None):
    import pyrender

    scene = pyrender.Scene(bg_color=bg_color or [0.05, 0.05, 0.05, 1.0])
    loaded = trimesh.load(glb_path)

    for geom in iter_geometries(loaded):
        if isinstance(geom, trimesh.points.PointCloud):
            pts = np.asarray(geom.vertices, dtype=np.float32)
            if geom.colors is not None:
                colors = np.asarray(geom.colors[:, :3], dtype=np.float32) / 255.0
            else:
                colors = np.ones((len(pts), 3), dtype=np.float32)
            if len(pts):
                scene.add(pyrender.Mesh(primitives=[
                    pyrender.Primitive(positions=pts, color_0=colors, mode=0)
                ]))

    return scene


def open3d_c2w_to_pyrender(c2w_opencv):
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.asarray(c2w_opencv, dtype=np.float64) @ opencv_to_opengl


def interpolate_positions(times, positions, sample_times):
    if len(times) < 3:
        return np.stack([
            np.interp(sample_times, times, positions[:, axis])
            for axis in range(3)
        ], axis=1)
    return np.stack([
        CubicSpline(times, positions[:, axis])(sample_times)
        for axis in range(3)
    ], axis=1)


def interpolate_rotations(times, rotations, sample_times):
    if len(times) < 3:
        return Slerp(times, rotations)(sample_times).as_matrix()
    return RotationSpline(times, rotations)(sample_times).as_matrix()


def keyframe_source_size(keyframe, config, fallback_width, fallback_height):
    width = keyframe.get("intrinsic_width", config.get("intrinsic_width", config.get("width", fallback_width)))
    height = keyframe.get("intrinsic_height", config.get("intrinsic_height", config.get("height", fallback_height)))
    intrinsic = keyframe.get("intrinsic")
    if intrinsic is not None:
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        if "intrinsic_width" not in keyframe and intrinsic[0, 2] > 0:
            width = max(width, int(round(2.0 * intrinsic[0, 2] + 1.0)))
        if "intrinsic_height" not in keyframe and intrinsic[1, 2] > 0:
            height = max(height, int(round(2.0 * intrinsic[1, 2] + 1.0)))
    return float(width), float(height)


def keyframe_yfov(keyframe, config, render_width, render_height):
    intrinsic = keyframe.get("intrinsic")
    if intrinsic is None:
        return np.pi / 4.0

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    fx = intrinsic[0, 0]
    if fx <= 0:
        return np.pi / 4.0

    source_width, _ = keyframe_source_size(keyframe, config, render_width, render_height)
    source_xfov = 2.0 * np.arctan(source_width / (2.0 * fx))
    render_aspect = render_width / render_height
    return 2.0 * np.arctan(np.tan(source_xfov / 2.0) / render_aspect)


def interpolate_keyframes(keyframes, config, n_frames, render_width, render_height):
    c2ws = np.asarray([kf["c2w_opencv"] for kf in keyframes], dtype=np.float64)
    yfovs = np.asarray([
        keyframe_yfov(kf, config, render_width, render_height)
        for kf in keyframes
    ], dtype=np.float64)

    if len(c2ws) == 1:
        return np.repeat(c2ws, n_frames, axis=0), np.repeat(yfovs, n_frames)

    times = np.arange(len(c2ws), dtype=np.float64)
    sample_times = np.linspace(0.0, float(len(c2ws) - 1), n_frames)
    positions = interpolate_positions(times, c2ws[:, :3, 3], sample_times)
    rotations = interpolate_rotations(times, Rotation.from_matrix(c2ws[:, :3, :3]), sample_times)
    yfovs = np.interp(sample_times, times, yfovs)

    out = np.repeat(np.eye(4)[None, :, :], n_frames, axis=0)
    out[:, :3, :3] = rotations
    out[:, :3, 3] = positions
    return out, yfovs


def render_keyframes(glb_path, segmented_glb_path, keyframe_config, out_dir, width, height, fps, n_frames):
    import pyrender

    keyframes = keyframe_config.get("keyframes", [])
    if not keyframes:
        raise ValueError("No keyframes found in keyframe JSON")

    frame_dir = os.path.join(out_dir, "3d_vis_frames")
    video_dir = os.path.join(out_dir, "videos")
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    half_width = width // 2
    scene_full = build_point_scene(glb_path)
    scene_segmented = build_point_scene(segmented_glb_path)
    c2ws_opencv, yfovs = interpolate_keyframes(keyframes, keyframe_config, n_frames, half_width, height)

    renderer = pyrender.OffscreenRenderer(half_width, height)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    for i, (c2w_opencv, yfov) in enumerate(tqdm(zip(c2ws_opencv, yfovs), total=n_frames, desc="Rendering", unit="frame")):
        cam_pose = open3d_c2w_to_pyrender(c2w_opencv)
        camera = pyrender.PerspectiveCamera(yfov=float(yfov))
        frames = []

        for scene in (scene_full, scene_segmented):
            renderer.point_size = 1.0
            cam_node = scene.add(camera, pose=cam_pose)
            light_node = scene.add(light, pose=cam_pose)
            color, _ = renderer.render(scene)
            frames.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            scene.remove_node(cam_node)
            scene.remove_node(light_node)

        combined = np.hstack(frames)
        cv2.putText(combined, "Full Scene", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(combined, "Segmented", (half_width + 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.jpg"), combined)

    renderer.delete()
    video_path = os.path.join(video_dir, "3d_vis.mp4")
    frames_to_video(frame_dir, video_path, fps)
    return video_path


def main():
    args = parse_args()
    if args.projected_video:
        video_path = render_projected_video(
            args.frame_dir,
            args.orange_pts,
            args.poses,
            args.pts_dir,
            args.depth_dir,
            args.out_dir,
            args.fps,
        )
        print(f"[DONE] Saved {video_path}")
        return

    with open(args.keyframes) as f:
        keyframe_config = json.load(f)

    video_path = render_keyframes(
        args.glb,
        args.segmented_glb,
        keyframe_config,
        args.out_dir,
        args.width,
        args.height,
        args.fps,
        args.frames,
    )
    print(f"[DONE] Saved {video_path}")


if __name__ == "__main__":
    main()

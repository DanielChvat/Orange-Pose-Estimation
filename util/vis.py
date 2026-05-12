import argparse
import json
import os
import shutil
import subprocess
import pyrender
import cv2
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def parse_args():
    parser = argparse.ArgumentParser(description="Render full-vs-segmented GLBs from Open3D keyframes")
    parser.add_argument("--glb", type=str, default="out/dust3r/scene.glb")
    parser.add_argument("--segmented-glb", type=str, default="out/voted_pts_by_id/orange_pts3d.glb")
    parser.add_argument("--shapes-glb", type=str, default="out/fitted_shapes/fitted_meshes.glb")
    parser.add_argument("--keyframes", type=str, default="vis/open3d_keyframes.json")
    parser.add_argument("--projected-video", action="store_true")
    parser.add_argument("--frame-dir", type=str, default="out/frames")
    parser.add_argument("--orange-pts", type=str, default="out/voted_pts/orange_pts3d.npz")
    parser.add_argument("--poses", type=str, default="out/dust3r/camera_poses.json")
    parser.add_argument("--pts-dir", type=str, default="out/dust3r")
    parser.add_argument("--depth-dir", type=str, default="out/dust3r/depths")
    parser.add_argument("--out-dir", type=str, default="vis")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--mask-fps", type=float, default=5.0)
    parser.add_argument("--render-fps", type=float, default=30)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--mask-video", action="store_true")
    parser.add_argument("--all-videos", action="store_true", help="Render both 2D mask and 3D comparison videos")
    parser.add_argument("--mask-dir", type=str, default="out/masks")
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument(
        "--mask-key",
        type=str,
        default="instance_mask",
        help="Preferred mask key to read from .npz files.",
    )
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


def build_render_scene(glb_path, bg_color=None):
    scene = pyrender.Scene(bg_color=bg_color or [0.05, 0.05, 0.05, 1.0])
    loaded = trimesh.load(glb_path)

    # Simple material avoids pyrender trying to upload GLB textures.
    default_mesh_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.55, 0.05, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )

    for geom in iter_geometries(loaded):
        if isinstance(geom, trimesh.points.PointCloud):
            pts = np.asarray(geom.vertices, dtype=np.float32)

            if geom.colors is not None:
                colors = np.asarray(geom.colors[:, :3], dtype=np.float32) / 255.0
            else:
                colors = np.ones((len(pts), 3), dtype=np.float32)

            if len(pts):
                scene.add(
                    pyrender.Mesh(
                        primitives=[
                            pyrender.Primitive(
                                positions=pts,
                                color_0=colors,
                                mode=0,
                            )
                        ]
                    )
                )

        elif isinstance(geom, trimesh.Trimesh) and len(geom.vertices):
            # Copy mesh and remove visual/material texture references.
            mesh = geom.copy()
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile(
                    np.array([[255, 140, 0, 255]], dtype=np.uint8),
                    (len(mesh.vertices), 1),
                ),
            )

            scene.add(
                pyrender.Mesh.from_trimesh(
                    mesh,
                    material=default_mesh_material,
                    smooth=True,
                )
            )

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


def render_keyframes(glb_path, segmented_glb_path, shapes_glb_path, keyframe_config, out_dir, width, height, fps, n_frames):
    keyframes = keyframe_config.get("keyframes", [])
    if not keyframes:
        raise ValueError("No keyframes found in keyframe JSON")

    frame_dir = os.path.join(out_dir, "3d_vis_frames")
    video_dir = os.path.join(out_dir, "videos")
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    scene_specs = [
        ("Full Scene", glb_path),
        ("Segmented Points", segmented_glb_path),
        ("Fitted Spheres", shapes_glb_path),
    ]
    scene_specs = [(label, path) for label, path in scene_specs if path and os.path.exists(path)]

    if len(scene_specs) == 0:
        raise FileNotFoundError("No 3D GLBs found to render")

    panel_width = max(1, width // len(scene_specs))
    scenes = [(label, build_render_scene(path)) for label, path in scene_specs]
    c2ws_opencv, yfovs = interpolate_keyframes(keyframes, keyframe_config, n_frames, panel_width, height)

    renderer = pyrender.OffscreenRenderer(panel_width, height)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    for i, (c2w_opencv, yfov) in enumerate(tqdm(zip(c2ws_opencv, yfovs), total=n_frames, desc="Rendering", unit="frame")):
        cam_pose = open3d_c2w_to_pyrender(c2w_opencv)
        camera = pyrender.PerspectiveCamera(yfov=float(yfov))
        frames = []

        for label, scene in scenes:
            renderer.point_size = 1.0
            cam_node = scene.add(camera, pose=cam_pose)
            light_node = scene.add(light, pose=cam_pose)
            color, _ = renderer.render(scene)
            frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame,
                label,
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(frame)
            scene.remove_node(cam_node)
            scene.remove_node(light_node)

        combined = np.hstack(frames)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.jpg"), combined)

    renderer.delete()
    video_path = os.path.join(video_dir, "3d_vis.mp4")
    frames_to_video(frame_dir, video_path, fps)
    return video_path

def id_to_color(object_id):
    rng = np.random.default_rng(int(object_id) * 9973)

    color = rng.integers(40, 256, size=3, dtype=np.uint8)

    # OpenCV uses BGR
    return tuple(int(c) for c in color)


def colorize_instance_mask(instance_mask):
    """
    Convert HxW integer instance mask into HxWx3 BGR color image.
    0 = background.
    1,2,3,... = different object ids.
    """
    instance_mask = instance_mask.astype(np.int32)

    h, w = instance_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    object_ids = np.unique(instance_mask)
    object_ids = object_ids[object_ids != 0]

    for oid in object_ids:
        color_mask[instance_mask == oid] = id_to_color(oid)

    return color_mask


def load_mask_npz(mask_path, preferred_key="instance_mask"):
    data = np.load(mask_path, allow_pickle=True)

    for key in [preferred_key, "instance_mask", "local_instance_mask", "mask"]:
        if key in data:
            mask = data[key].astype(np.int32)
            break
    else:
        raise KeyError(
            f"{mask_path} does not contain a usable mask key. "
            f"Available keys: {list(data.keys())}"
        )

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    return mask


def draw_label_box(image, text, origin, color):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    pad_x = 6
    pad_y = 5

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - text_h - baseline - pad_y)
    x1 = min(image.shape[1] - 1, x + text_w + pad_x)
    y1 = min(image.shape[0] - 1, y + baseline + pad_y)

    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_instance_labels(image, instance_mask):
    """
    Draw object id labels and outlines on every visible mask component.

    If tracking assigns one global ID to multiple SAM islands in a frame, each
    island gets the same label so split/occluded objects are obvious in video.
    """
    out = image.copy()
    object_ids = np.unique(instance_mask)
    object_ids = object_ids[object_ids != 0]

    for oid in object_ids:
        component_mask = (instance_mask == oid).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            component_mask,
            connectivity=8,
        )
        color = id_to_color(oid)

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < 20:
                continue

            cx, cy = centroids[label_id]
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue

            text = f"id {int(oid)}"
            draw_label_box(out, text, (int(cx), int(cy)), color)

        if num_labels <= 1:
            continue

    return out


def render_mask_video(frame_dir, mask_dir, out_dir, fps, alpha=0.45, mask_key="instance_mask"):
    """
    Render side-by-side:
      left: original frame
      right: frame with instance mask overlay
    """
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Missing frame dir: {frame_dir}")

    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

    mask_vis_dir = os.path.join(out_dir, "mask_vis_frames")
    video_dir = os.path.join(out_dir, "videos")

    if os.path.exists(mask_vis_dir):
        shutil.rmtree(mask_vis_dir)

    os.makedirs(mask_vis_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    frame_names = sorted(
        name for name in os.listdir(frame_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for name in tqdm(frame_names, desc="Rendering masks", unit="frame"):
        frame_path = os.path.join(frame_dir, name)
        stem = os.path.splitext(name)[0]

        # Try common mask names
        candidate_mask_paths = [
            os.path.join(mask_dir, f"{stem}.npz"),
            os.path.join(mask_dir, f"{stem}_mask.npz"),
            os.path.join(mask_dir, f"{stem}_masks.npz"),
        ]

        mask_path = None
        for p in candidate_mask_paths:
            if os.path.exists(p):
                mask_path = p
                break

        if mask_path is None:
            print(f"[WARN] No mask found for frame {name}")
            continue

        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"[WARN] Could not read frame {frame_path}")
            continue

        instance_mask = load_mask_npz(mask_path, preferred_key=mask_key)

        # Resize mask to frame size if needed.
        # Use nearest-neighbor so object ids do not get interpolated.
        if instance_mask.shape[:2] != frame.shape[:2]:
            instance_mask = cv2.resize(
                instance_mask.astype(np.int32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        color_mask = colorize_instance_mask(instance_mask)

        overlay = frame.copy()
        foreground = instance_mask > 0

        overlay = frame.copy()
        foreground = instance_mask > 0

        if np.any(foreground):
            blended = cv2.addWeighted(
                frame,
                1.0 - alpha,
                color_mask,
                alpha,
                0.0,
            )

            overlay[foreground] = blended[foreground]
        else:
            cv2.putText(
                overlay,
                "No SAM mask",
                (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        overlay = draw_instance_labels(overlay, instance_mask)

        combined = np.hstack((frame, overlay))

        cv2.putText(
            combined,
            "Original",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            combined,
            "Object ID Masks",
            (frame.shape[1] + 12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = os.path.join(mask_vis_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, combined)

    video_path = os.path.join(video_dir, "mask_vis.mp4")
    frames_to_video(mask_vis_dir, video_path, fps)

    return video_path

def main():
    args = parse_args()
    mask_fps = args.mask_fps
    render_fps = args.render_fps

    if args.mask_video or args.all_videos:
        mask_video_path = render_mask_video(
            args.frame_dir,
            args.mask_dir,
            args.out_dir,
            args.mask_fps,
            alpha=args.mask_alpha,
            mask_key=args.mask_key,
        )
        print(f"[DONE] Saved {mask_video_path}")
        if not args.all_videos:
            return

    if args.projected_video:
        video_path = render_projected_video(
            args.frame_dir,
            args.orange_pts,
            args.poses,
            args.pts_dir,
            args.depth_dir,
            args.out_dir,
            args.render_fps,
        )
        print(f"[DONE] Saved {video_path}")
        return

    with open(args.keyframes) as f:
        keyframe_config = json.load(f)

    video_path = render_keyframes(
        args.glb,
        args.segmented_glb,
        args.shapes_glb,
        keyframe_config,
        args.out_dir,
        args.width,
        args.height,
        args.render_fps,
        args.frames,
    )

    print(f"[DONE] Saved {video_path}")


if __name__ == "__main__":
    main()

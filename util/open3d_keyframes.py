import argparse
import json
import os

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Record Open3D camera keyframes for 3D visualisation")
    parser.add_argument("--glb", type=str, default="out/dust3r/scene.glb")
    parser.add_argument("--segmented-glb", type=str, default="out/voted_pts/orange_pts3d.glb")
    parser.add_argument("--out", type=str, default="vis/open3d_keyframes.json")
    parser.add_argument("--sample-points", type=int, default=120000)
    parser.add_argument("--move-step", type=float, default=0.02)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def iter_geometries(loaded):
    if isinstance(loaded, trimesh.Scene):
        yield from loaded.geometry.values()
    else:
        yield loaded


def trimesh_to_open3d_geometries(path, sample_points):
    import open3d as o3d

    loaded = trimesh.load(path)
    geometries = []

    for geom in iter_geometries(loaded):
        if isinstance(geom, trimesh.points.PointCloud):
            pts = np.asarray(geom.vertices, dtype=np.float64)
            colors = np.asarray(geom.colors[:, :3], dtype=np.float64) / 255.0
        elif isinstance(geom, trimesh.Trimesh):
            if len(geom.faces) and sample_points > 0:
                pts, face_idx = trimesh.sample.sample_surface(geom, min(sample_points, max(sample_points // 2, 1)))
                visual = getattr(geom.visual, "vertex_colors", None)
                if visual is not None and len(visual) == len(geom.vertices):
                    face_colors = np.asarray(visual[np.asarray(geom.faces)[face_idx, 0], :3], dtype=np.float64) / 255.0
                    colors = face_colors
                else:
                    colors = np.full((len(pts), 3), 0.7, dtype=np.float64)
            else:
                pts = np.asarray(geom.vertices, dtype=np.float64)
                visual = getattr(geom.visual, "vertex_colors", None)
                if visual is not None and len(visual) == len(pts):
                    colors = np.asarray(visual[:, :3], dtype=np.float64) / 255.0
                else:
                    colors = np.full((len(pts), 3), 0.7, dtype=np.float64)
        else:
            continue

        if len(pts) > sample_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), size=sample_points, replace=False)
            pts = pts[idx]
            colors = colors[idx]

        if len(pts):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(cloud)

    return geometries


def camera_to_world_from_view(vis):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    return np.linalg.inv(params.extrinsic)


def camera_params_from_view(vis):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    intrinsic = params.intrinsic
    return {
        "c2w_opencv": np.linalg.inv(params.extrinsic).tolist(),
        "intrinsic": np.asarray(intrinsic.intrinsic_matrix).tolist(),
        "intrinsic_width": int(intrinsic.width),
        "intrinsic_height": int(intrinsic.height),
    }


def set_camera_to_world(vis, c2w):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    params.extrinsic = np.linalg.inv(c2w)
    vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def move_camera(vis, delta_camera, move_step):
    c2w = camera_to_world_from_view(vis)
    rotation = c2w[:3, :3]
    c2w[:3, 3] += rotation @ (np.asarray(delta_camera, dtype=np.float64) * move_step)
    set_camera_to_world(vis, c2w)


def main():
    args = parse_args()

    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise SystemExit("Open3D is not installed. Install it with: pip install open3d") from exc

    geometries = []
    geometries.extend(trimesh_to_open3d_geometries(args.glb, args.sample_points))
    segmented = trimesh_to_open3d_geometries(args.segmented_glb, max(args.sample_points // 2, 1))
    for geom in segmented:
        geom.paint_uniform_color([1.0, 0.45, 0.0])
    geometries.extend(segmented)

    keyframes = []

    vis = o3d.visualization.VisualizerWithKeyCallback()
    created = vis.create_window("Open3D keyframe recorder", width=args.width, height=args.height)
    if not created:
        raise SystemExit(
            "Open3D could not create an OpenGL window. On Wayland, try running with:\n"
            "  XDG_SESSION_TYPE=x11 python util/open3d_keyframes.py\n"
            "or software OpenGL:\n"
            "  LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 python util/open3d_keyframes.py\n"
            "If those fail, log into an X11/Xorg desktop session and run the recorder there."
        )
    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    if render_option is None:
        vis.destroy_window()
        raise SystemExit(
            "Open3D created a window object but failed to initialize its OpenGL context/GLEW.\n"
            "Try one of these:\n"
            "  XDG_SESSION_TYPE=x11 python util/open3d_keyframes.py\n"
            "  LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 python util/open3d_keyframes.py\n"
            "Best option: run from an X11/Xorg session instead of Wayland."
        )
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.05, 0.05, 0.05])

    def add_keyframe(v):
        keyframes.append(camera_params_from_view(v))
        print(f"[INFO] Captured keyframe {len(keyframes)}")
        return False

    def undo_keyframe(_):
        if keyframes:
            keyframes.pop()
        print(f"[INFO] Keyframes: {len(keyframes)}")
        return False

    def save_keyframes(_):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "keyframes": keyframes,
                "width": args.width,
                "height": args.height,
            }, f, indent=2)
        print(f"[DONE] Saved {len(keyframes)} keyframes to {args.out}")
        print(f"PYOPENGL_PLATFORM=egl python util/vis.py --keyframes {args.out} --width {args.width} --height {args.height}")
        return False

    vis.register_key_callback(ord("K"), add_keyframe)
    vis.register_key_callback(ord("U"), undo_keyframe)
    vis.register_key_callback(ord("P"), save_keyframes)
    vis.register_key_callback(ord("W"), lambda v: move_camera(v, [0, 0, 1], args.move_step) or False)
    vis.register_key_callback(ord("S"), lambda v: move_camera(v, [0, 0, -1], args.move_step) or False)
    vis.register_key_callback(ord("A"), lambda v: move_camera(v, [-1, 0, 0], args.move_step) or False)
    vis.register_key_callback(ord("D"), lambda v: move_camera(v, [1, 0, 0], args.move_step) or False)
    vis.register_key_callback(ord("Q"), lambda v: move_camera(v, [0, -1, 0], args.move_step) or False)
    vis.register_key_callback(ord("E"), lambda v: move_camera(v, [0, 1, 0], args.move_step) or False)

    print("Controls:")
    print("  Mouse: rotate/pan/zoom using Open3D controls")
    print("  W/S: move forward/back")
    print("  A/D: move left/right")
    print("  Q/E: move down/up")
    print("  K: capture current camera keyframe")
    print("  U: undo last keyframe")
    print("  P: save keyframes JSON")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Fit spheres to tracked object points")
    parser.add_argument("--points", type=str, default="out/voted_pts_by_id/orange_pts3d.npz")
    parser.add_argument("--out-dir", type=str, default="out/fitted_shapes")
    parser.add_argument("--trim-quantile", type=float, default=0.88, help="Keep this central fraction of points for robust fitting")
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--mesh-alpha", type=int, default=90, help="0-255 alpha for overlay sphere meshes")
    parser.add_argument("--points-alpha", type=int, default=210, help="0-255 alpha for point colors in overlay")
    parser.add_argument("--no-points", action="store_true")
    return parser.parse_args()


def id_color(object_id, alpha=255):
    rng = np.random.default_rng(int(object_id) * 9973)
    rgb = rng.integers(40, 256, size=3, dtype=np.uint8)
    return [int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha)]


def load_points(path):
    data = np.load(path, allow_pickle=True)
    if "pts3d" not in data:
        raise KeyError(f"{path} does not contain pts3d")
    if "object_ids" not in data:
        raise KeyError(
            f"{path} does not contain object_ids. Run segment_points.py with --track-ids first."
        )

    pts = np.asarray(data["pts3d"], dtype=np.float32)
    object_ids = np.asarray(data["object_ids"], dtype=np.int32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected pts3d shape Nx3, got {pts.shape}")
    if len(pts) != len(object_ids):
        raise ValueError("pts3d and object_ids have different lengths")

    finite = np.isfinite(pts).all(axis=1) & (object_ids > 0)
    return pts[finite], object_ids[finite]


def robust_core_points(pts, trim_quantile):
    center = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - center[None, :], axis=1)
    keep = dists <= np.quantile(dists, trim_quantile)
    if int(keep.sum()) < 4:
        return pts
    return pts[keep]


def fit_sphere_lstsq(pts):
    a = np.concatenate([2.0 * pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    b = np.sum(pts * pts, axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = sol[:3]
    radius_sq = float(np.dot(center, center) + sol[3])
    radius = float(np.sqrt(max(radius_sq, 1e-12)))
    return center.astype(np.float32), radius


def fit_robust_sphere(pts, trim_quantile, niter=4):
    core = robust_core_points(pts, trim_quantile)

    if len(core) < 4:
        center = np.median(pts, axis=0)
        radius = float(np.median(np.linalg.norm(pts - center[None, :], axis=1)))
        return center.astype(np.float32), radius, core

    center, radius = fit_sphere_lstsq(core)

    for _ in range(niter):
        dists = np.linalg.norm(pts - center[None, :], axis=1)
        residuals = np.abs(dists - radius)
        keep = residuals <= np.quantile(residuals, trim_quantile)
        if int(keep.sum()) < 4:
            break
        core = pts[keep]
        center, radius = fit_sphere_lstsq(core)

    return center.astype(np.float32), float(radius), core


def object_fit(object_id, pts, trim_quantile):
    center, radius, core = fit_robust_sphere(pts, trim_quantile)
    dists = np.linalg.norm(pts - center[None, :], axis=1)
    residuals = np.abs(dists - radius)

    return {
        "object_id": int(object_id),
        "num_points": int(len(pts)),
        "num_fit_points": int(len(core)),
        "center": center.astype(float).tolist(),
        "radius": float(radius),
        "median_abs_error": float(np.median(residuals)),
        "mean_abs_error": float(np.mean(residuals)),
        "p90_abs_error": float(np.quantile(residuals, 0.90)),
    }


def transform_sphere(mesh, center, radius):
    transform = np.eye(4)
    transform[:3, :3] *= float(radius)
    transform[:3, 3] = center
    mesh.apply_transform(transform)
    return mesh


def export_scene(pts, object_ids, fits, out_path, subdivisions, mesh_alpha, points_alpha, include_points):
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required to export fitted GLBs") from exc

    scene = trimesh.Scene()

    if include_points:
        colors = np.asarray([id_color(oid, alpha=points_alpha) for oid in object_ids], dtype=np.uint8)
        scene.add_geometry(trimesh.PointCloud(pts, colors=colors), geom_name="object_points")

    for fit in fits:
        color = id_color(fit["object_id"], alpha=mesh_alpha)
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        center = np.asarray(fit["center"], dtype=np.float32)
        mesh = transform_sphere(mesh, center=center, radius=fit["radius"])
        mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
        scene.add_geometry(mesh, geom_name=f"sphere_id_{fit['object_id']}")

    scene.export(out_path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pts, object_ids = load_points(args.points)

    fits = []
    for object_id in sorted(int(x) for x in np.unique(object_ids)):
        obj_pts = pts[object_ids == object_id]
        if len(obj_pts) < args.min_points:
            print(f"[WARN] Skipping id {object_id}: only {len(obj_pts)} points")
            continue
        fits.append(object_fit(object_id, obj_pts, args.trim_quantile))

    json_path = os.path.join(args.out_dir, "shapes.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "points": args.points,
                "shape": "sphere",
                "trim_quantile": args.trim_quantile,
                "objects": fits,
            },
            f,
            indent=2,
        )

    glb_path = os.path.join(args.out_dir, "fitted_shapes.glb")
    export_scene(
        pts,
        object_ids,
        fits,
        glb_path,
        subdivisions=args.subdivisions,
        mesh_alpha=args.mesh_alpha,
        points_alpha=args.points_alpha,
        include_points=not args.no_points,
    )

    mesh_glb_path = os.path.join(args.out_dir, "fitted_meshes.glb")
    export_scene(
        pts,
        object_ids,
        fits,
        mesh_glb_path,
        subdivisions=args.subdivisions,
        mesh_alpha=255,
        points_alpha=args.points_alpha,
        include_points=False,
    )

    print(f"[DONE] Saved shape parameters to {json_path}")
    print(f"[DONE] Saved fitted shape GLB to {glb_path}")
    print(f"[DONE] Saved fitted mesh-only GLB to {mesh_glb_path}")


if __name__ == "__main__":
    main()

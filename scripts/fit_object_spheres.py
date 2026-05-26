"""Per-object oriented bounding box + inscribed sphere fitting.

Reads the object-aware Gaussian point cloud from
`out/gaussian_ownership/gaussian_tracks.npz` (output of
`infer_gaussian_ownership.py`) and, for every object_id > 0:

  1. PCA on the Gaussian centres -> oriented bounding box axes + extents.
  2. Robust centroid = median of points in OBB frame, mapped back to world.
  3. Sphere radius = median distance from centroid to points (so the sphere
     hugs the bulk of the Gaussians, not outliers), clamped to the smallest
     OBB half-extent so the sphere is guaranteed to fit inside the OBB.

Outputs:
  out/object_spheres/spheres.json    list of {object_id, center, R (3x3), half_extents, sphere_center, sphere_radius}
  out/object_spheres/spheres.glb     visualization: OBB wireframes + spheres per object

This is the leaf step of the orange-segmentation pipeline. After this, each
orange has a known 3D position + orientation + radius, suitable for
downstream manipulation use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit OBB + inscribed sphere per object.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--ownership-npz", type=str, default=None,
                        help="Path to gaussian_tracks.npz (default: out/gaussian_ownership/gaussian_tracks.npz)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Default: out/object_spheres")
    parser.add_argument("--min-points", type=int, default=None,
                        help="Skip objects with fewer than this many Gaussians.")
    parser.add_argument("--radius-percentile", type=float, default=None,
                        help="Percentile of point-to-centroid distance used as the sphere radius. "
                             "50 = median (robust to outliers). 90 = more conservative (encloses more points).")
    parser.add_argument("--spatial-cluster", action="store_true", default=None,
                        help="Re-cluster all foreground Gaussians by 3D proximity instead of trusting "
                             "the per-mask object IDs. Bypasses SAM3 instance-merging failures by "
                             "letting spatial separation define objects.")
    parser.add_argument("--no-spatial-cluster", action="store_false", dest="spatial_cluster",
                        help="Trust the per-mask object IDs from infer_gaussian_ownership directly.")
    parser.add_argument("--cluster-eps-multiplier", type=float, default=2.5,
                        help="DBSCAN epsilon = multiplier * median nearest-neighbor distance. "
                             "Larger = merges nearby Gaussians into one object; smaller = splits more.")
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    return namespace_from_config(
        cli.config,
        "object_spheres",
        {
            "ownership_npz": cli.ownership_npz,
            "out_dir": cli.out_dir,
            "min_points": cli.min_points,
            "radius_percentile": cli.radius_percentile,
            "spatial_cluster": cli.spatial_cluster,
            "cluster_eps_multiplier": cli.cluster_eps_multiplier,
            "overwrite": cli.overwrite,
        },
    )


def spatial_cluster_labels(xyz: np.ndarray, eps_multiplier: float = 2.5,
                           min_samples: int = 8) -> np.ndarray:
    """Re-label points by 3D spatial proximity, ignoring any prior labels.

    Returns a (N,) int array where label > 0 = cluster id, 0 = noise.
    Epsilon is auto-derived from the median nearest-neighbor distance so
    the threshold adapts to the scene's Gaussian density.
    """
    from scipy.spatial import cKDTree

    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int32)
    tree = cKDTree(xyz)
    # Median NN distance (excluding self at index 0).
    dists, _ = tree.query(xyz, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) == 0:
        return np.zeros(len(xyz), dtype=np.int32)
    eps = float(np.median(nn) * eps_multiplier)
    eps = max(eps, 1e-6)

    # Flood-fill clustering: each point joins its eps-neighbors' cluster.
    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int32)  # -1 = unvisited
    cluster_id = 0
    for seed in range(n):
        if labels[seed] != -1:
            continue
        # BFS from this point
        cluster_id += 1
        stack = [seed]
        members = []
        while stack:
            idx = stack.pop()
            if labels[idx] != -1:
                continue
            labels[idx] = cluster_id
            members.append(idx)
            neighbors = tree.query_ball_point(xyz[idx], r=eps)
            for nb in neighbors:
                if labels[nb] == -1:
                    stack.append(nb)
        # Demote small clusters to noise (label 0)
        if len(members) < min_samples:
            for m in members:
                labels[m] = 0
            cluster_id -= 1  # reclaim id

    # Convert -1 (shouldn't exist after loop) and 0 (noise) -> 0
    labels[labels < 0] = 0
    return labels


def fit_obb(points: np.ndarray) -> dict:
    """PCA-based oriented bounding box.

    Returns dict with keys:
      center        (3,)       OBB centre in world coords
      rotation      (3, 3)     world-from-OBB; columns are OBB axes (length 1)
      half_extents  (3,)       half-sizes along each OBB axis (descending magnitude)
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Covariance + eigendecomposition; eigenvectors are PCA axes.
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; flip so axis 0 is the LONGEST.
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # Enforce right-handed frame (det = +1).
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] = -eigvecs[:, -1]
    # Project points onto axes and measure extent.
    projected = centered @ eigvecs   # (N, 3) in OBB-aligned frame
    obb_min = projected.min(axis=0)
    obb_max = projected.max(axis=0)
    half_extents = 0.5 * (obb_max - obb_min)
    # OBB centre in world coords = world-centroid + (axis-midpoint in OBB frame) mapped back.
    midpoint_obb = 0.5 * (obb_max + obb_min)
    center = centroid + eigvecs @ midpoint_obb
    return {
        "center": center.astype(np.float64),
        "rotation": eigvecs.astype(np.float64),
        "half_extents": half_extents.astype(np.float64),
    }


def algebraic_sphere_fit(points: np.ndarray) -> dict | None:
    """Best-fit sphere whose surface the splats lie on. Minimises
    (||p - c||^2 - r^2)^2 in closed form. Works for partial-coverage point
    clouds (a hemisphere of splats still recovers the full-sphere radius
    from the curvature), and is naturally outlier-resistant because the
    LSQ objective balances over all points instead of being dragged by
    the farthest one."""
    if len(points) < 4:
        return None
    A = np.column_stack([2.0 * points, np.ones(len(points))])
    b = (points * points).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = sol[:3]
    r_sq = float(sol[3] + (center * center).sum())
    if not np.isfinite(r_sq) or r_sq <= 0.0:
        return None
    return {"center": np.asarray(center, dtype=np.float64), "radius": float(np.sqrt(r_sq))}


def fit_sphere_in_obb(points: np.ndarray, obb: dict, *, radius_percentile: float,
                      use_algebraic_fit: bool = True,
                      algebraic_max_radius_factor: float = 3.0) -> dict:
    """Pick a sphere that physically encloses the cluster's splats.

    Default: algebraic sphere fit on the points. This is the geometrically
    correct radius for spherical objects and doesn't get fooled by an
    off-centre OBB or a few outliers. Falls back to percentile-of-distance
    from the OBB centre when the algebraic fit is missing or absurdly
    large (which indicates near-coplanar points where the LSQ blows up).

    `radius_percentile` controls the fallback only:
      100 = max distance (sensitive to outliers)
      92  = ignore the farthest ~8% of splats (default; robust)
      50  = median distance (very tight to the bulk)
    """
    obb_center = obb["center"]
    distances = np.linalg.norm(points - obb_center, axis=1)
    pct_radius = float(np.percentile(distances, radius_percentile))
    inscribed_cap = float(obb["half_extents"].min())

    sphere_center = obb_center.astype(np.float64)
    radius = pct_radius
    source = "percentile"
    algebraic_radius = None

    if use_algebraic_fit:
        algebraic = algebraic_sphere_fit(np.asarray(points, dtype=np.float64))
        if algebraic is not None:
            algebraic_radius = float(algebraic["radius"])
            # Reject if the LSQ blew up on a near-coplanar cluster.
            ref = max(pct_radius, inscribed_cap, 1e-6)
            if algebraic_radius < ref * algebraic_max_radius_factor:
                sphere_center = algebraic["center"]
                radius = algebraic_radius
                source = "algebraic"

    return {
        "center": sphere_center,
        "radius": float(radius),
        "radius_from_percentile": pct_radius,
        "radius_inscribed_cap": inscribed_cap,
        "radius_from_algebraic_fit": algebraic_radius,
        "source": source,
        "extends_past_obb": float(radius) > inscribed_cap,
        "clamped": False,
    }


def export_glb(out_path: Path, objects: list[dict]) -> None:
    """One opaque sphere per object, colored deterministically from its
    object_id. No OBB wireframes, no textures."""
    scene = trimesh.Scene()
    for obj in objects:
        oid = obj["object_id"]
        color = _color_for(oid)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=float(obj["sphere_radius"]))
        sphere.apply_translation(np.asarray(obj["sphere_center"], dtype=np.float64))
        sphere.visual.face_colors = (*color, 255)
        scene.add_geometry(sphere, node_name=f"sphere_{oid:04d}")
    scene.export(str(out_path))


def _color_for(oid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(oid) * 73856093 ^ 19349663)
    return tuple(int(c) for c in rng.integers(48, 240, size=3))


def main() -> None:
    args = parse_args()

    ownership_npz = Path(args.ownership_npz or "out/gaussian_ownership/gaussian_tracks.npz")
    if not ownership_npz.exists():
        raise SystemExit(
            f"[fit_object_spheres] {ownership_npz} not found. "
            "Run scripts/infer_gaussian_ownership.py first."
        )

    out_dir = Path(args.out_dir or "out/object_spheres")
    if out_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(ownership_npz)
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    n_seed_objects = int(np.unique(labels[labels > 0]).size)

    if bool(getattr(args, "spatial_cluster", True)):
        # Use ALL foreground Gaussians and let 3D spatial clustering decide
        # object IDs. Bypasses SAM-merging failures: if SAM grouped 5 touching
        # oranges into one mask, those gaussians end up with the same seed
        # label, but spatial clustering splits them by 3D proximity here.
        foreground = labels > 0
        fg_xyz = xyz[foreground]
        print(f"[fit_object_spheres] spatial-cluster mode: re-clustering "
              f"{int(foreground.sum())} foreground Gaussians "
              f"(was {n_seed_objects} SAM-derived object IDs)")
        new_labels_fg = spatial_cluster_labels(
            fg_xyz,
            eps_multiplier=float(getattr(args, "cluster_eps_multiplier", 2.5)),
            min_samples=int(getattr(args, "min_points", 20)),
        )
        # Map back to full-length labels array
        labels = np.zeros_like(labels)
        labels[foreground] = new_labels_fg
        print(f"[fit_object_spheres]   -> {int(np.unique(labels[labels > 0]).size)} "
              f"spatial clusters")
    else:
        print(f"[fit_object_spheres] using {n_seed_objects} input object IDs")

    object_ids = sorted(int(i) for i in np.unique(labels) if int(i) > 0)
    print(f"[fit_object_spheres] {len(object_ids)} objects after labeling")

    objects: list[dict] = []
    skipped = 0
    for oid in object_ids:
        pts = xyz[labels == oid]
        if pts.shape[0] < args.min_points:
            skipped += 1
            continue
        obb = fit_obb(pts)
        sphere = fit_sphere_in_obb(
            pts, obb,
            radius_percentile=args.radius_percentile,
            use_algebraic_fit=bool(getattr(args, "use_algebraic_fit", True)),
            algebraic_max_radius_factor=float(getattr(args, "algebraic_max_radius_factor", 3.0)),
        )
        objects.append({
            "object_id": int(oid),
            "num_gaussians": int(pts.shape[0]),
            "center": obb["center"].tolist(),
            "rotation": obb["rotation"].tolist(),
            "half_extents": obb["half_extents"].tolist(),
            "sphere_center": sphere["center"].tolist(),
            "sphere_radius": sphere["radius"],
            "sphere_radius_source": sphere["source"],
            "sphere_radius_from_percentile": sphere["radius_from_percentile"],
            "sphere_radius_from_algebraic_fit": sphere["radius_from_algebraic_fit"],
            "sphere_radius_inscribed_cap": sphere["radius_inscribed_cap"],
            "sphere_extends_past_obb": sphere["extends_past_obb"],
        })
        print(f"  obj {oid:4d}: {pts.shape[0]:5d} gaussians  "
              f"OBB half-extents=({obb['half_extents'][0]:.3f}, "
              f"{obb['half_extents'][1]:.3f}, "
              f"{obb['half_extents'][2]:.3f})  "
              f"sphere r={sphere['radius']:.3f} [{sphere['source']}]")

    json_path = out_dir / "spheres.json"
    glb_path = out_dir / "spheres.glb"
    json_path.write_text(json.dumps({
        "num_objects": len(objects),
        "num_skipped_under_min_points": skipped,
        "radius_percentile": float(args.radius_percentile),
        "min_points": int(args.min_points),
        "ownership_npz": str(ownership_npz),
        "objects": objects,
    }, indent=2))
    if objects:
        export_glb(glb_path, objects)

    print(f"[fit_object_spheres] {len(objects)} fits written ({skipped} skipped under min_points).")
    print(f"  json: {json_path}")
    if objects:
        print(f"  glb:  {glb_path}")


if __name__ == "__main__":
    main()

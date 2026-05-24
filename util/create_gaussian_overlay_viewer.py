import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import DEFAULT_CONFIG, namespace_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Create a browser viewer for a 3D Gaussian splat scene with an optional fitted object mesh overlay.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-gaussians", type=str, default=None)
    parser.add_argument("--tracks-npz", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--mesh-dir", type=str, default=None, help="Per-object mesh directory (overrides config)")
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    return namespace_from_config(
        cli.config,
        "viewer",
        {
            "scene_gaussians": cli.scene_gaussians,
            "tracks_npz": cli.tracks_npz,
            "out_dir": cli.out_dir,
            "mesh_dir": cli.mesh_dir,
            "overwrite": cli.overwrite,
        },
    )


def splat_bounds(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) == 0:
        return {"center": [0, 0, 0], "radius": 1, "low": [-1, -1, -1], "high": [1, 1, 1]}
    low = np.percentile(xyz, 1.0, axis=0)
    high = np.percentile(xyz, 99.0, axis=0)
    center = (low + high) * 0.5
    radius = float(np.linalg.norm(high - low) * 0.55)
    return {
        "center": center.astype(float).tolist(),
        "radius": max(radius, 0.05),
        "low": low.astype(float).tolist(),
        "high": high.astype(float).tolist(),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prune_splats(src, dst, min_opacity, max_scale_percentile):
    ply = PlyData.read(src)
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    if not {"opacity", "scale_0", "scale_1", "scale_2"}.issubset(names):
        shutil.copy2(src, dst)
        return len(vertex), len(vertex)

    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))
    scales = np.exp(
        np.column_stack([
            np.asarray(vertex["scale_0"], dtype=np.float64),
            np.asarray(vertex["scale_1"], dtype=np.float64),
            np.asarray(vertex["scale_2"], dtype=np.float64),
        ])
    )
    max_scale = scales.max(axis=1)
    scale_limit = np.percentile(max_scale, max_scale_percentile)
    keep = np.isfinite(opacity) & np.isfinite(max_scale) & (opacity >= min_opacity) & (max_scale <= scale_limit)
    filtered = vertex[keep]
    PlyData([PlyElement.describe(filtered, "vertex")], text=ply.text).write(dst)
    return len(vertex), len(filtered)


def clamp_splat_scales(path, max_scale):
    if max_scale <= 0:
        return 0
    ply = PlyData.read(path)
    vertex = ply["vertex"].data.copy()
    names = vertex.dtype.names or ()
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if not set(scale_names).issubset(names):
        return 0
    scales = np.exp(np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in scale_names]))
    clamped = np.minimum(scales, float(max_scale))
    changed = np.any(clamped < scales, axis=1)
    if not np.any(changed):
        return 0
    log_clamped = np.log(np.maximum(clamped, 1e-8))
    for dim, name in enumerate(scale_names):
        vertex[name] = log_clamped[:, dim]
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply.text).write(path)
    return int(np.count_nonzero(changed))


def write_track_metadata(path, tracks_npz, tracks_json=None):
    world_up = None
    if tracks_json and os.path.exists(tracks_json):
        with open(tracks_json) as f:
            summary = json.load(f)
        wu = summary.get("world_up")
        if isinstance(wu, list) and len(wu) == 3:
            world_up = [float(v) for v in wu]
        objects = []
        for obj in summary.get("objects", []):
            axes = np.asarray(obj.get("axes", []), dtype=np.float64)
            radius = float(max(np.max(axes), 0.01)) if axes.size else float(obj.get("radius", 0.01))
            source_prompts = [str(prompt) for prompt in obj.get("source_prompts", []) if prompt]
            class_label = source_prompts[0] if source_prompts else "Unlabeled"
            obb = obj.get("obb")
            obb_payload = None
            if isinstance(obb, dict) and obb.get("center") and obb.get("rotation") and obb.get("half_extents"):
                obb_payload = {
                    "center": [float(v) for v in obb["center"]],
                    "rotation": [[float(c) for c in row] for row in obb["rotation"]],
                    "half_extents": [float(v) for v in obb["half_extents"]],
                }
                for k in ("sam_align_angle_deg", "sam_align_aspect_score", "sam_align_iou"):
                    if k in obb:
                        obb_payload[k] = float(obb[k])
            objects.append({
                "id": int(obj["id"]),
                "center": [float(v) for v in obj["center"]],
                "radius": radius,
                "axes": [float(v) for v in axes.tolist()] if axes.size else [],
                "num_splats": int(obj.get("num_splats", obj.get("num_fit_splats", 0))),
                "shape_type": obj.get("shape_type", "object"),
                "source_prompts": source_prompts,
                "class_label": class_label,
                "obb": obb_payload,
            })
        if objects:
            with open(path, "w") as f:
                json.dump({"objects": objects, "world_up": world_up}, f)
            return len(objects)

    data = np.load(tracks_npz)
    labels = np.asarray(data["labels"], dtype=np.int32)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    objects = []
    for oid in sorted(int(v) for v in np.unique(labels) if v > 0):
        pts = xyz[labels == oid]
        if len(pts) == 0:
            continue
        center = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        radius = float(max(np.percentile(dist, 88), 0.01))
        objects.append({
            "id": oid,
            "center": center.astype(float).tolist(),
            "radius": radius,
            "num_splats": int(len(pts)),
            "source_prompts": [],
            "class_label": "Unlabeled",
        })
    with open(path, "w") as f:
        json.dump({"objects": objects, "world_up": world_up}, f)
    return len(objects)


def write_refined_class_layers(out_dir, refined_dir, metadata_path, existing_layers):
    if not refined_dir or not os.path.isdir(refined_dir) or not os.path.exists(metadata_path):
        return existing_layers
    with open(metadata_path) as f:
        metadata = json.load(f)
    layer_by_label = {layer["label"]: layer for layer in existing_layers}
    by_class = {}
    for obj in metadata.get("objects", []):
        label = str(obj.get("class_label") or "Unlabeled")
        by_class.setdefault(label, []).append(int(obj["id"]))
    used_slugs = set()
    for label, object_ids in sorted(by_class.items(), key=lambda item: item[0].casefold()):
        vertices = []
        template_ply = None
        for oid in object_ids:
            obj_path = os.path.join(refined_dir, f"object_{oid:04d}_refined.ply")
            if os.path.exists(obj_path):
                p = PlyData.read(obj_path)
                if template_ply is None:
                    template_ply = p
                vertices.append(p["vertex"].data)
        if not vertices or template_ply is None:
            continue
        combined = np.concatenate(vertices)
        slug = class_slug(label)
        base_slug = slug
        suffix = 2
        while slug in used_slugs:
            slug = f"{base_slug}_{suffix}"
            suffix += 1
        used_slugs.add(slug)
        filename = f"refined_{slug}.ply"
        PlyData([PlyElement.describe(combined, "vertex")], text=template_ply.text).write(os.path.join(out_dir, filename))
        if label in layer_by_label:
            layer_by_label[label]["refined_file"] = filename
        else:
            new_layer = {
                "label": label,
                "file": filename,
                "refined_file": filename,
                "object_ids": object_ids,
                "num_splats": int(len(combined)),
            }
            layer_by_label[label] = new_layer
            existing_layers.append(new_layer)
    return existing_layers


def class_slug(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower()).strip("._-")
    return slug[:48] or "unlabeled"


def write_class_splat_layers(out_dir, tracked_ply_path, tracks_npz, metadata_path):
    if not tracks_npz or not os.path.exists(tracks_npz) or not os.path.exists(metadata_path):
        return []
    data = np.load(tracks_npz)
    labels = np.asarray(data["labels"], dtype=np.int32)
    selected_labels = labels[labels > 0]
    ply = PlyData.read(tracked_ply_path)
    vertex = ply["vertex"].data
    if len(vertex) != len(selected_labels):
        return []
    with open(metadata_path) as f:
        metadata = json.load(f)

    by_class = {}
    for obj in metadata.get("objects", []):
        label = str(obj.get("class_label") or "Unlabeled")
        by_class.setdefault(label, []).append(int(obj["id"]))

    layers = []
    used_slugs = set()
    for label, object_ids in sorted(by_class.items(), key=lambda item: item[0].casefold()):
        keep = np.isin(selected_labels, np.asarray(object_ids, dtype=np.int32))
        if not np.any(keep):
            continue
        slug = class_slug(label)
        base_slug = slug
        suffix = 2
        while slug in used_slugs:
            slug = f"{base_slug}_{suffix}"
            suffix += 1
        used_slugs.add(slug)
        filename = f"tracked_{slug}.ply"
        PlyData([PlyElement.describe(vertex[keep].copy(), "vertex")], text=ply.text).write(os.path.join(out_dir, filename))
        layers.append({
            "label": label,
            "file": filename,
            "object_ids": object_ids,
            "num_splats": int(np.count_nonzero(keep)),
        })
    return layers


def quat_to_rot(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def initial_camera_from_colmap(path):
    if not path or not os.path.exists(path):
        sibling = str(Path(path).with_suffix(".bin")) if path else ""
        if sibling and os.path.exists(sibling):
            path = sibling
        else:
            return None
    if path.endswith(".bin"):
        try:
            import sys

            repo = Path("third_party/gaussian-splatting").resolve()
            if repo.exists():
                sys.path.insert(0, str(repo))
            from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary

            images = read_extrinsics_binary(path)
            image = sorted(images.values(), key=lambda item: item.name)[0]
            r = qvec2rotmat(image.qvec)
            t = np.asarray(image.tvec, dtype=np.float64)
            center = -r.T @ t
            up = -r.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
            return {
                "position": center.astype(float).tolist(),
                "up": up.astype(float).tolist(),
            }
        except Exception:
            return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            t = np.array(list(map(float, parts[5:8])), dtype=np.float64)
            r = quat_to_rot(qw, qx, qy, qz)
            center = -r.T @ t
            up = -r.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
            return {
                "position": center.astype(float).tolist(),
                "up": up.astype(float).tolist(),
            }
    return None


def write_html(
    path,
    splat_name,
    alternate_splat_name,
    tracked_splat_name,
    refined_splat_name,
    track_metadata_name,
    fitted_glb_name,
    scene_label,
    alternate_label,
    tracked_label,
    refined_label,
    fitted_label,
    object_name,
    object_opacity,
    fitted_opacity,
    transparent_fitted,
    class_layers,
    bounds,
    initial_camera,
    flip_y,
    flip_z,
    show_helpers,
):
    bounds_json = json.dumps(bounds)
    camera_json = json.dumps(initial_camera)
    build_id = str(int(time.time() * 1000))
    class_layers_json = json.dumps(class_layers)
    fitted_transparent_js = "true" if transparent_fitted else "false"
    flip_value = "true" if flip_y else "false"
    flip_z_value = "true" if flip_z else "false"
    helpers_value = "true" if show_helpers else "false"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <title>3D Gaussian Splat Scene</title>
  <style>
    html, body {{ margin: 0; overflow: hidden; width: 100%; height: 100%; background: #111; }}
    #hint {{ position: fixed; left: 12px; top: 10px; color: #eee; font: 13px/1.35 sans-serif; background: rgba(0,0,0,.45); padding: 8px 10px; border-radius: 6px; z-index: 2; }}
    #status {{ position: fixed; left: 12px; bottom: 12px; color: #ddd; font: 12px/1.35 monospace; background: rgba(0,0,0,.55); padding: 7px 9px; border-radius: 6px; z-index: 2; max-width: min(720px, calc(100vw - 40px)); white-space: pre-wrap; }}
    #hover {{ position: fixed; left: 0; top: 0; transform: translate(12px, 12px); color: #fff; font: 12px/1.35 monospace; background: rgba(0,0,0,.72); padding: 5px 7px; border-radius: 5px; z-index: 3; pointer-events: none; display: none; }}
    #toolbar {{ position: fixed; right: 12px; top: 10px; display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; max-width: 470px; z-index: 2; }}
    button {{ color: #eee; background: rgba(20,20,20,.72); border: 1px solid rgba(255,255,255,.22); border-radius: 6px; padding: 7px 9px; cursor: pointer; }}
    button:hover {{ background: rgba(55,55,55,.84); }}
    select {{ color: #eee; background: rgba(20,20,20,.82); border: 1px solid rgba(255,255,255,.22); border-radius: 6px; padding: 7px 9px; }}
  </style>
</head>
<body>
<div id="hint">3DGS scene{(' + fitted object meshes' if object_name else '')}<br>left drag: free rotate, right drag: pan, wheel: zoom</div>
<div id="toolbar">
  <button id="reset">Reset</button>
  <button id="front">Front</button>
  <button id="back">Back</button>
  <button id="top">Top</button>
  <button id="modeScene">Full Scene</button>
  <button id="modeExtracted">Extracted Splats</button>
  <button id="modeTracked">{tracked_label}</button>
  <button id="modeRefined">{refined_label}</button>
  <button id="modeFitted">Fitted Shapes</button>
  <button id="modeMeshes">Meshes</button>
  <button id="modeOBB">Bounding Boxes</button>
  <button id="modeCameras">Cameras</button>
  <button id="orientWorld">Orient World</button>
  <select id="classFilter" title="Object class"></select>
  <button id="sceneLayer">Scene On</button>
  <button id="flipY">Flip Y</button>
  <button id="flipZ">Flip Z</button>
  <button id="helpers">Helpers</button>
</div>
<div id="status">Loading splats...</div>
<div id="hover"></div>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/",
    "@mkkellogg/gaussian-splats-3d": "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ TrackballControls }} from 'three/addons/controls/TrackballControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';
import {{ OBJLoader }} from 'three/addons/loaders/OBJLoader.js';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const status = document.getElementById('status');

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x151515, 1);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const bounds = {bounds_json};
const initialCamera = {camera_json};
const sceneLabel = {json.dumps(scene_label)};
const alternateLabel = {json.dumps(alternate_label)};
const trackedLabel = {json.dumps(tracked_label)};
const refinedLabel = {json.dumps(refined_label)};
const fittedLabel = {json.dumps(fitted_label)};
const classLayers = {class_layers_json};
const center = new THREE.Vector3(...bounds.center);
const radius = Math.max(bounds.radius, 0.05);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, radius / 2000, radius * 30.0);
camera.up.set(0, 0, 1);

const controls = new TrackballControls(camera, renderer.domElement);
controls.target.copy(center);
controls.rotateSpeed = 3.0;
controls.zoomSpeed = 1.4;
controls.panSpeed = 0.65;
controls.dynamicDampingFactor = 0.12;
controls.noRoll = false;

const world = new THREE.Group();
scene.add(world);
let flipY = {flip_value};
let flipZ = {flip_z_value};
const objectPath = {json.dumps(object_name)};
const fittedPath = {json.dumps(fitted_glb_name)};
const trackMetadataPath = {json.dumps(track_metadata_name)};
const buildId = {json.dumps(build_id)};
let objectRoot = null;
let fittedRoot = null;
let fittedMeshes = [];
let trackObjects = [];
let helpersVisible = {helpers_value};
const hover = document.getElementById('hover');

function assetUrl(name) {{
  if (!name) return name;
  const separator = name.includes('?') ? '&' : '?';
  return `${{name}}${{separator}}v=${{buildId}}`;
}}

function transformed(v) {{
  return new THREE.Vector3(v.x, flipY ? -v.y : v.y, flipZ ? -v.z : v.z);
}}

function applyWorldTransform() {{
  world.scale.set(1, flipY ? -1 : 1, flipZ ? -1 : 1);
  controls.target.copy(transformed(center));
  controls.update();
}}

function lookFrom(offset, up = new THREE.Vector3(0, 0, 1)) {{
  const target = transformed(center);
  camera.position.copy(target).add(offset);
  camera.up.copy(up);
  camera.near = radius / 2000;
  camera.far = radius * 30.0;
  camera.updateProjectionMatrix();
  controls.target.copy(target);
  controls.update();
}}

function resetCamera() {{
  if (initialCamera?.position && initialCamera?.up) {{
    camera.position.copy(transformed(new THREE.Vector3(...initialCamera.position)));
    camera.up.copy(new THREE.Vector3(...initialCamera.up).normalize());
    controls.target.copy(transformed(center));
    controls.update();
  }} else {{
    lookFrom(new THREE.Vector3(0.0, radius * 1.8, radius * 0.75));
  }}
}}
applyWorldTransform();
resetCamera();

scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const light = new THREE.DirectionalLight(0xffffff, 1.7);
light.position.set(0.2, -0.5, 0.5);
scene.add(light);

let detectedWorldUp = null;
let worldOriented = false;
const defaultCameraUp = new THREE.Vector3(0, 0, 1);

const helpers = new THREE.Group();
helpers.visible = helpersVisible;
world.add(helpers);

const helpersContent = new THREE.Group();
helpers.add(helpersContent);

const axes = new THREE.AxesHelper(radius * 0.35);
helpersContent.add(axes);

const grid = new THREE.GridHelper(radius * 2.0, 24, 0x555555, 0x333333);
helpersContent.add(grid);

function applyHelpersOrientation() {{
  // Position helpers at scene center, then rotate so their Y (THREE's default up) aligns with detected world-up.
  helpers.position.copy(center);
  if (!detectedWorldUp) {{
    helpers.quaternion.identity();
    return;
  }}
  const up = new THREE.Vector3(detectedWorldUp[0], detectedWorldUp[1], detectedWorldUp[2]).normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
  helpers.quaternion.copy(q);
}}
applyHelpersOrientation();

function makeSplatViewer() {{
  return new GaussianSplats3D.DropInViewer({{
    selfDrivenMode: false,
    renderer,
    camera,
    gpuAcceleratedSort: false,
    sharedMemoryForWorkers: false,
    ignoreDevicePixelRatio: false,
  }});
}}

const splats = makeSplatViewer();
const alternateSplats = {('makeSplatViewer()' if alternate_splat_name else 'null')};
const trackedSplats = {('makeSplatViewer()' if tracked_splat_name else 'null')};
const refinedSplats = {('makeSplatViewer()' if refined_splat_name else 'null')};
let sceneLayerVisible = true;
let extractedVisible = false;
let trackedVisible = false;
let refinedVisible = false;
let fittedVisible = false;
let obbVisible = false;
let activeClass = '__all__';
const classViewers = new Map();
world.add(splats);
if (alternateSplats) world.add(alternateSplats);
if (trackedSplats) world.add(trackedSplats);
if (refinedSplats) world.add(refinedSplats);

const obbGroup = new THREE.Group();
obbGroup.visible = false;
world.add(obbGroup);

const camerasGroup = new THREE.Group();
camerasGroup.visible = false;
world.add(camerasGroup);

const meshesGroup = new THREE.Group();
meshesGroup.visible = false;
world.add(meshesGroup);

const meshesByClass = new Map();  // class_label -> array of THREE.Mesh

async function loadObjectMeshes(meshList) {{
  while (meshesGroup.children.length) {{
    const child = meshesGroup.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  }}
  meshesByClass.clear();
  if (!Array.isArray(meshList)) return;
  const plyLoader = new PLYLoader();
  const objLoader = new OBJLoader();
  const texLoader = new THREE.TextureLoader();
  for (const m of meshList) {{
    if (!m.file) continue;
    try {{
      const isObj = m.file.toLowerCase().endsWith('.obj');
      let mesh = null;
      if (isObj) {{
        const group = await new Promise((resolve, reject) => objLoader.load(assetUrl(m.file), resolve, undefined, reject));
        // OBJLoader returns a Group; the OBJ has a single object so we take its first child Mesh
        const child = group.children.find((c) => c.isMesh);
        if (!child) {{ console.warn('OBJ has no mesh', m); continue; }}
        const geom = child.geometry;
        geom.computeVertexNormals();
        let texture = null;
        if (m.texture) {{
          texture = await new Promise((resolve) => texLoader.load(assetUrl(m.texture), resolve, undefined, () => resolve(null)));
          if (texture) {{
            texture.colorSpace = THREE.SRGBColorSpace;
            // OBJ stores v-flipped UVs (we flipped at write time), so three.js's default flipY=true works correctly
          }}
        }}
        const mat = new THREE.MeshStandardMaterial({{
          map: texture,
          color: 0xffffff,
          side: THREE.DoubleSide,
          roughness: 0.85,
          metalness: 0.05,
        }});
        mesh = new THREE.Mesh(geom, mat);
      }} else {{
        const geom = await new Promise((resolve, reject) => plyLoader.load(assetUrl(m.file), resolve, undefined, reject));
        geom.computeVertexNormals();
        const hasColor = geom.hasAttribute('color');
        const mat = new THREE.MeshStandardMaterial({{
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : new THREE.Color().setHSL(((Number(m.id || 0) * 137.508) % 360) / 360, 0.7, 0.55),
          side: THREE.DoubleSide,
          roughness: 0.85,
          metalness: 0.05,
        }});
        mesh = new THREE.Mesh(geom, mat);
      }}
      mesh.userData = {{ id: m.id, class_label: m.class_label, source_prompts: m.source_prompts || [], num_splats: m.num_splats }};
      meshesGroup.add(mesh);
      const label = m.class_label || 'Unlabeled';
      if (!meshesByClass.has(label)) meshesByClass.set(label, []);
      meshesByClass.get(label).push(mesh);
    }} catch (err) {{
      console.warn('Could not load object mesh', m, err);
    }}
  }}
  console.log(`Loaded ${{meshesGroup.children.length}} object meshes`);
  applyObjectMeshMetadata();
  applyLayerVisibility();
}}

async function fetchObjectMeshes() {{
  try {{
    const response = await fetch(assetUrl('object_meshes.json'), {{ cache: 'no-store' }});
    if (!response.ok) return;
    const data = await response.json();
    const meshes = Array.isArray(data?.meshes) ? data.meshes : [];
    await loadObjectMeshes(meshes);
  }} catch (err) {{
    console.warn('Could not load object_meshes.json', err);
  }}
}}
fetchObjectMeshes();

function buildCameras(cameraData) {{
  while (camerasGroup.children.length) {{
    const child = camerasGroup.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  }}
  if (!Array.isArray(cameraData)) return;
  // Tiny frustums — meant as position markers, not rays through the scene
  const camSize = radius * 0.025;
  for (const cam of cameraData) {{
    const Rmat = cam.rotation;
    if (!Rmat || Rmat.length !== 3) continue;
    const pos = cam.position || [0, 0, 0];
    // R is world-to-camera; camera-to-world transform is R^T
    // Compute frustum corners (depth = camSize, image plane at z=camSize in camera coords)
    const aspect = (cam.width && cam.height) ? (cam.width / cam.height) : 1.5;
    const halfW = camSize * 0.5 * aspect;
    const halfH = camSize * 0.5;
    const cornersCam = [
      [-halfW, -halfH, camSize],
      [ halfW, -halfH, camSize],
      [ halfW,  halfH, camSize],
      [-halfW,  halfH, camSize],
    ];
    // R^T: rows of R^T are columns of R; matrix-vector = sum R[k][i] * v[k]
    const RT = [
      [Rmat[0][0], Rmat[1][0], Rmat[2][0]],
      [Rmat[0][1], Rmat[1][1], Rmat[2][1]],
      [Rmat[0][2], Rmat[1][2], Rmat[2][2]],
    ];
    const cornersWorld = cornersCam.map((c) => [
      pos[0] + RT[0][0]*c[0] + RT[0][1]*c[1] + RT[0][2]*c[2],
      pos[1] + RT[1][0]*c[0] + RT[1][1]*c[1] + RT[1][2]*c[2],
      pos[2] + RT[2][0]*c[0] + RT[2][1]*c[1] + RT[2][2]*c[2],
    ]);
    const positions = [];
    // 4 lines from camera origin to image-plane corners
    for (const c of cornersWorld) {{
      positions.push(pos[0], pos[1], pos[2], c[0], c[1], c[2]);
    }}
    // 4 lines connecting image-plane corners (the rectangle)
    for (let i = 0; i < 4; i++) {{
      const a = cornersWorld[i];
      const b = cornersWorld[(i + 1) % 4];
      positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }}
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({{ color: 0xffaa33, transparent: true, opacity: 0.35, depthTest: true, depthWrite: false }});
    const wire = new THREE.LineSegments(geom, mat);
    wire.renderOrder = 49;
    camerasGroup.add(wire);
  }}
}}

async function loadCameras() {{
  try {{
    const response = await fetch(assetUrl('cameras.json'), {{ cache: 'no-store' }});
    if (!response.ok) return;
    const data = await response.json();
    const cams = Array.isArray(data?.cameras) ? data.cameras : (Array.isArray(data) ? data : []);
    buildCameras(cams);
    console.log(`Loaded ${{cams.length}} cameras`);
  }} catch (err) {{
    console.warn('Could not load cameras.json', err);
  }}
}}
loadCameras();

function idToColorHSL(id) {{
  const hue = ((id * 137.508) % 360) / 360;
  return new THREE.Color().setHSL(hue, 0.85, 0.6);
}}

function buildOBBs(objects) {{
  while (obbGroup.children.length) {{
    const child = obbGroup.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  }}
  for (const obj of objects) {{
    const obb = obj?.obb;
    if (!obb || !obb.center || !obb.rotation || !obb.half_extents) continue;
    const he = obb.half_extents;
    const geom = new THREE.BoxGeometry(2 * he[0], 2 * he[1], 2 * he[2]);
    const edges = new THREE.EdgesGeometry(geom);
    const color = idToColorHSL(Number(obj.id) || 0);
    const mat = new THREE.LineBasicMaterial({{ color, transparent: true, opacity: 0.95, depthTest: false, depthWrite: false }});
    const wire = new THREE.LineSegments(edges, mat);
    wire.renderOrder = 50;
    const R = obb.rotation;
    const m = new THREE.Matrix4();
    m.set(
      R[0][0], R[0][1], R[0][2], 0,
      R[1][0], R[1][1], R[1][2], 0,
      R[2][0], R[2][1], R[2][2], 0,
      0, 0, 0, 1,
    );
    wire.quaternion.setFromRotationMatrix(m);
    wire.position.set(obb.center[0], obb.center[1], obb.center[2]);
    wire.userData = obj;
    geom.dispose();
    obbGroup.add(wire);
  }}
}}

async function loadSplat(viewer, name, label) {{
  await viewer.addSplatScene(name, {{
    splatAlphaRemovalThreshold: 1,
    showLoadingUI: true,
    position: [0, 0, 0],
    rotation: [0, 0, 0, 1],
    scale: [1, 1, 1],
  }});
  status.textContent = `${{label}} loaded`;
}}

try {{
  await loadSplat(splats, '{splat_name}', sceneLabel);
  if (alternateSplats) {{
    await loadSplat(alternateSplats, '{alternate_splat_name}', alternateLabel);
    alternateSplats.visible = false;
  }}
  if (trackedSplats) {{
    await loadSplat(trackedSplats, '{tracked_splat_name}', trackedLabel);
    trackedSplats.visible = false;
  }}
  if (refinedSplats) {{
    await loadSplat(refinedSplats, '{refined_splat_name}', refinedLabel);
    refinedSplats.visible = false;
  }}
  status.textContent = 'Layers loaded. Mode buttons keep the same camera.';
  setTimeout(() => status.style.display = 'none', 1800);
}} catch (err) {{
  console.error(err);
  status.textContent = 'Failed to load splats:\\n' + (err?.stack || err?.message || String(err));
}}

if (objectPath) {{
  new GLTFLoader().load(assetUrl(objectPath), (gltf) => {{
    gltf.scene.traverse((obj) => {{
      if (obj.isMesh) {{
        obj.material.transparent = true;
        obj.material.opacity = {object_opacity:.3f};
        obj.material.depthWrite = false;
        obj.material.side = THREE.DoubleSide;
        obj.renderOrder = 10;
      }}
    }});
    objectRoot = gltf.scene;
    world.add(objectRoot);
  }});
}}

if (fittedPath) {{
  new GLTFLoader().load(assetUrl(fittedPath), (gltf) => {{
    fittedMeshes = [];
    gltf.scene.traverse((obj) => {{
      if (obj.isMesh) {{
        const map = obj.material?.map || null;
        if (map) map.colorSpace = THREE.SRGBColorSpace;
        const color = map ? new THREE.Color(0xffffff) : (obj.material?.color ? obj.material.color.clone() : new THREE.Color(0xffffff));
        obj.material = new THREE.MeshBasicMaterial({{
          color,
          map,
          vertexColors: !map && Boolean(obj.geometry?.attributes?.color),
          transparent: {fitted_transparent_js} || {fitted_opacity:.3f} < 0.999,
          opacity: {fitted_opacity:.3f},
          depthWrite: false,
          depthTest: false,
          side: THREE.DoubleSide,
        }});
        obj.renderOrder = 40;
        fittedMeshes.push(obj);
      }}
    }});
    fittedRoot = gltf.scene;
    fittedRoot.visible = fittedVisible;
    world.add(fittedRoot);
    applyFittedMetadata();
    setObjectVisibility();
  }}, undefined, (err) => {{
    console.error(err);
    status.style.display = 'block';
    status.textContent = 'Failed to load fitted shapes:\\n' + (err?.message || String(err));
  }});
}}

const pickObjects = [];
const pickRoot = new THREE.Group();
world.add(pickRoot);

function clearHoverHighlight() {{
}}

function setHoverHighlight(target) {{
}}

function metadataForObjectId(id) {{
  const parsed = Number(id);
  if (!Number.isFinite(parsed)) return null;
  return trackObjects.find((obj) => Number(obj.id) === parsed) || null;
}}

function classMatches(obj) {{
  if (activeClass === '__all__') return true;
  const prompts = obj?.source_prompts || [];
  return obj?.class_label === activeClass || prompts.includes(activeClass);
}}

function setObjectVisibility() {{
  const pickVisible = trackedVisible || fittedVisible;
  for (const proxy of pickRoot.children) {{
    proxy.visible = pickVisible && classMatches(proxy.userData);
  }}
  for (const mesh of fittedMeshes) {{
    mesh.visible = fittedVisible && classMatches(mesh.userData);
  }}
}}

function applyObjectMeshMetadata() {{
  for (const mesh of meshesGroup.children) {{
    const existing = mesh.userData || {{}};
    const meta = metadataForObjectId(existing.id);
    mesh.userData = meta ? {{ ...existing, ...meta }} : existing;
    if (!pickObjects.includes(mesh)) pickObjects.push(mesh);
  }}
}}

function setupClassFilter() {{
  const select = document.getElementById('classFilter');
  select.innerHTML = '';
  const all = document.createElement('option');
  all.value = '__all__';
  all.textContent = 'All Classes';
  select.appendChild(all);
  const labels = new Set();
  for (const layer of classLayers) labels.add(layer.label);
  for (const obj of trackObjects) {{
    if (obj.class_label) labels.add(obj.class_label);
  }}
  for (const label of meshesByClass.keys()) labels.add(label);
  for (const label of [...labels].sort((a, b) => a.localeCompare(b))) {{
    const option = document.createElement('option');
    option.value = label;
    option.textContent = label;
    select.appendChild(option);
  }}
  select.value = activeClass;
  select.disabled = select.options.length <= 1;
  select.onchange = async () => {{
    activeClass = select.value;
    await ensureClassLayer(activeClass);
    applyLayerVisibility();
  }};
}}

const refinedClassViewers = new Map();

async function ensureClassLayer(label) {{
  if (label === '__all__') return;
  const layer = classLayers.find((item) => item.label === label);
  if (!layer) return;
  if (!classViewers.has(label) && layer.file) {{
    const viewer = makeSplatViewer();
    viewer.visible = false;
    world.add(viewer);
    classViewers.set(label, viewer);
    try {{
      await loadSplat(viewer, layer.file, `${{layer.label}} tracked splats`);
      viewer.visible = trackedVisible && activeClass === label;
    }} catch (err) {{
      console.warn('Could not load class tracked layer', layer, err);
    }}
  }}
  if (!refinedClassViewers.has(label) && layer.refined_file) {{
    const viewer = makeSplatViewer();
    viewer.visible = false;
    world.add(viewer);
    refinedClassViewers.set(label, viewer);
    try {{
      await loadSplat(viewer, layer.refined_file, `${{layer.label}} refined splats`);
      viewer.visible = refinedVisible && activeClass === label;
    }} catch (err) {{
      console.warn('Could not load class refined layer', layer, err);
    }}
  }}
}}

function applyWorldOrientation() {{
  if (!detectedWorldUp) return;
  if (worldOriented) {{
    const up = new THREE.Vector3(detectedWorldUp[0], detectedWorldUp[1], detectedWorldUp[2]).normalize();
    camera.up.copy(up);
    // Position camera at center + horizontal-offset (perpendicular to up) + small up-offset
    let horizontal = new THREE.Vector3(0, 1, 0);
    if (Math.abs(horizontal.dot(up)) > 0.85) horizontal = new THREE.Vector3(1, 0, 0);
    horizontal.sub(up.clone().multiplyScalar(horizontal.dot(up))).normalize();
    const target = transformed(center);
    camera.position.copy(target).add(horizontal.multiplyScalar(radius * 1.8)).add(up.clone().multiplyScalar(radius * 0.5));
    controls.target.copy(target);
    controls.update();
  }} else {{
    camera.up.copy(defaultCameraUp);
    resetCamera();
  }}
}}

async function loadTrackPickers() {{
  if (!trackMetadataPath) return;
  try {{
    const response = await fetch(assetUrl(trackMetadataPath), {{ cache: 'no-store' }});
    const data = await response.json();
    trackObjects = data.objects || [];
    if (Array.isArray(data.world_up) && data.world_up.length === 3) {{
      detectedWorldUp = data.world_up.map(Number);
      applyHelpersOrientation();
    }}
    for (const obj of trackObjects) {{
      const pickRadius = Math.max(Number(obj.radius || 0.01) * 1.6, radius * 0.008);
      const geometry = new THREE.SphereGeometry(pickRadius, 12, 8);
      const material = new THREE.MeshBasicMaterial({{
        color: 0xffffff,
        transparent: true,
        opacity: 0.0,
        depthWrite: false,
      }});
      const proxy = new THREE.Mesh(geometry, material);
      proxy.position.set(...obj.center);
      proxy.userData = obj;
      pickRoot.add(proxy);
      pickObjects.push(proxy);
    }}
    applyFittedMetadata();
    setupClassFilter();
    setObjectVisibility();
    buildOBBs(trackObjects);
    applyObjectMeshMetadata();
  }} catch (err) {{
    console.warn('Could not load tracked object metadata', err);
  }}
}}
loadTrackPickers();

function metadataForMesh(mesh, fallbackIndex) {{
  const match = String(mesh.name || '').match(/object[_-]?(\\d+)/i);
  if (match) {{
    const parsed = Number(match[1]);
    const direct = trackObjects.find((obj) => Number(obj.id) === parsed);
    if (direct) return direct;
    const unpadded = Number(String(match[1]).replace(/^0+/, '') || '0');
    const byUnpadded = trackObjects.find((obj) => Number(obj.id) === unpadded);
    if (byUnpadded) return byUnpadded;
  }}
  return trackObjects[fallbackIndex] || null;
}}

function applyFittedMetadata() {{
  if (!trackObjects.length || !fittedMeshes.length) return;
  for (let i = 0; i < fittedMeshes.length; i++) {{
    const meta = metadataForMesh(fittedMeshes[i], i);
    if (!meta) continue;
    fittedMeshes[i].userData = meta;
    if (!pickObjects.includes(fittedMeshes[i])) pickObjects.push(fittedMeshes[i]);
  }}
  setObjectVisibility();
}}

const modeButtons = {{
  scene: document.getElementById('modeScene'),
  extracted: document.getElementById('modeExtracted'),
  tracked: document.getElementById('modeTracked'),
  refined: document.getElementById('modeRefined'),
  fitted: document.getElementById('modeFitted'),
  obb: document.getElementById('modeOBB'),
  cameras: document.getElementById('modeCameras'),
  meshes: document.getElementById('modeMeshes'),
}};
let camerasVisible = false;
let meshesVisible = false;

async function toggleLayer(layer) {{
  if (layer === 'scene') {{
    sceneLayerVisible = !sceneLayerVisible;
  }} else if (layer === 'extracted' && alternateSplats) {{
    extractedVisible = !extractedVisible;
  }} else if (layer === 'tracked' && trackedSplats) {{
    trackedVisible = !trackedVisible;
    if (trackedVisible) await ensureClassLayer(activeClass);
  }} else if (layer === 'refined' && refinedSplats) {{
    refinedVisible = !refinedVisible;
    if (refinedVisible) await ensureClassLayer(activeClass);
  }} else if (layer === 'fitted' && fittedPath) {{
    fittedVisible = !fittedVisible;
  }} else if (layer === 'obb') {{
    obbVisible = !obbVisible;
  }} else if (layer === 'cameras') {{
    camerasVisible = !camerasVisible;
  }} else if (layer === 'meshes') {{
    meshesVisible = !meshesVisible;
  }}
  applyLayerVisibility();
}}

function applyLayerVisibility() {{
  splats.visible = sceneLayerVisible;
  if (alternateSplats) alternateSplats.visible = extractedVisible;
  if (trackedSplats) trackedSplats.visible = trackedVisible && activeClass === '__all__';
  if (refinedSplats) refinedSplats.visible = refinedVisible && activeClass === '__all__';
  for (const [label, viewer] of classViewers.entries()) {{
    viewer.visible = trackedVisible && activeClass === label;
  }}
  for (const [label, viewer] of refinedClassViewers.entries()) {{
    viewer.visible = refinedVisible && activeClass === label;
  }}
  if (fittedRoot) fittedRoot.visible = fittedVisible;
  obbGroup.visible = obbVisible;
  for (const child of obbGroup.children) {{
    child.visible = obbVisible && classMatches(child.userData);
  }}
  camerasGroup.visible = camerasVisible;
  meshesGroup.visible = meshesVisible;
  for (const child of meshesGroup.children) {{
    child.visible = meshesVisible && classMatches(child.userData);
  }}
  if (!(trackedVisible || refinedVisible || fittedVisible || meshesVisible)) {{
    clearHoverHighlight();
    hover.style.display = 'none';
  }}
  setObjectVisibility();
  modeButtons.scene.style.background = sceneLayerVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  modeButtons.extracted.style.background = extractedVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  modeButtons.tracked.style.background = trackedVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  modeButtons.refined.style.background = refinedVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  modeButtons.fitted.style.background = fittedVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  modeButtons.obb.style.background = obbVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  modeButtons.cameras.style.background = camerasVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  modeButtons.meshes.style.background = meshesVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
  const sceneLayerButton = document.getElementById('sceneLayer');
  sceneLayerButton.textContent = sceneLayerVisible ? 'Scene On' : 'Scene Off';
  sceneLayerButton.style.background = sceneLayerVisible ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
}}

function setSceneLayerVisible(visible) {{
  sceneLayerVisible = visible;
  applyLayerVisibility();
}}

if (!alternateSplats) modeButtons.extracted.disabled = true;
if (!trackedSplats) modeButtons.tracked.disabled = true;
if (!refinedSplats) modeButtons.refined.disabled = true;
if (!fittedPath) modeButtons.fitted.disabled = true;
modeButtons.scene.onclick = () => toggleLayer('scene');
modeButtons.extracted.onclick = () => toggleLayer('extracted');
modeButtons.tracked.onclick = () => toggleLayer('tracked');
modeButtons.refined.onclick = () => toggleLayer('refined');
modeButtons.fitted.onclick = () => toggleLayer('fitted');
modeButtons.obb.onclick = () => toggleLayer('obb');
modeButtons.cameras.onclick = () => toggleLayer('cameras');
modeButtons.meshes.onclick = () => toggleLayer('meshes');
applyLayerVisibility();
setSceneLayerVisible(true);
setupClassFilter();

document.getElementById('reset').onclick = resetCamera;
document.getElementById('front').onclick = () => lookFrom(new THREE.Vector3(0, radius * 1.8, radius * 0.75));
document.getElementById('back').onclick = () => lookFrom(new THREE.Vector3(0, -radius * 1.8, radius * 0.75));
document.getElementById('top').onclick = () => lookFrom(new THREE.Vector3(0, 0, radius * 2.4), new THREE.Vector3(0, 1, 0));
document.getElementById('flipY').onclick = () => {{ flipY = !flipY; applyWorldTransform(); resetCamera(); }};
document.getElementById('flipZ').onclick = () => {{ flipZ = !flipZ; applyWorldTransform(); resetCamera(); }};
document.getElementById('helpers').onclick = () => {{ helpers.visible = !helpers.visible; }};
document.getElementById('sceneLayer').onclick = () => setSceneLayerVisible(!sceneLayerVisible);
document.getElementById('orientWorld').onclick = () => {{
  worldOriented = !worldOriented;
  applyWorldOrientation();
  document.getElementById('orientWorld').style.background = worldOriented ? 'rgba(78,98,135,.82)' : 'rgba(20,20,20,.72)';
}};

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener('pointermove', (event) => {{
  if (!(trackedVisible || refinedVisible || fittedVisible || meshesVisible) || pickObjects.length === 0) {{
    hover.style.display = 'none';
    clearHoverHighlight();
    return;
  }}
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  let hits = raycaster.intersectObjects(pickObjects, false);
  if (meshesVisible) {{
    const meshHit = hits.find((hit) => hit.object?.parent === meshesGroup);
    if (meshHit) hits = [meshHit];
  }}
  if (!hits.length) {{
    hover.style.display = 'none';
    clearHoverHighlight();
    return;
  }}
  const hitObject = hits[0].object;
  const obj = hitObject.userData || {{}};
  setHoverHighlight(hitObject);
  hover.style.left = `${{event.clientX}}px`;
  hover.style.top = `${{event.clientY}}px`;
  const parts = [`object_id=${{obj.id ?? 'unknown'}}`];
  if (obj.num_splats != null) parts.push(`splats=${{obj.num_splats}}`);
  if (obj.class_label) parts.push(`class=${{obj.class_label}}`);
  hover.textContent = parts.join('  ');
  hover.style.display = 'block';
}});
renderer.domElement.addEventListener('pointerleave', () => {{
  hover.style.display = 'none';
  clearHoverHighlight();
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  controls.handleResize();
}});
</script>
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(html)


def main():
    args = parse_args()
    if os.path.exists(args.out_dir):
        if args.overwrite:
            shutil.rmtree(args.out_dir)
        else:
            raise FileExistsError(f"{args.out_dir} exists. Use --overwrite.")
    os.makedirs(args.out_dir, exist_ok=True)
    asset_stamp = str(int(time.time() * 1000))
    scene_splat_name = f"scene_gaussians_{asset_stamp}.ply"
    alternate_splat_name = f"alternate_gaussians_{asset_stamp}.ply"
    tracked_splat_name = f"tracked_gaussians_{asset_stamp}.ply"
    refined_splat_name = f"refined_gaussians_{asset_stamp}.ply"
    splat_dst = os.path.join(args.out_dir, scene_splat_name)
    alternate_dst = os.path.join(args.out_dir, alternate_splat_name)
    tracked_dst = os.path.join(args.out_dir, tracked_splat_name)
    refined_dst = os.path.join(args.out_dir, refined_splat_name)
    track_metadata_dst = os.path.join(args.out_dir, "tracked_objects.json")
    fitted_dst = os.path.join(args.out_dir, "fitted_shapes.glb")
    alternate_name = ""
    tracked_name = ""
    refined_name = ""
    track_metadata_name = ""
    fitted_name = ""
    class_layers = []
    object_dst = os.path.join(args.out_dir, "object_meshes.glb")
    object_name = ""
    if args.no_prune_splats:
        shutil.copy2(args.scene_gaussians, splat_dst)
    else:
        original_count, filtered_count = prune_splats(
            args.scene_gaussians,
            splat_dst,
            args.min_splat_opacity,
            args.max_splat_scale_percentile,
        )
        print(f"[INFO] Viewer splat pruning: {original_count:,} -> {filtered_count:,}")
    if args.object_glb:
        if not os.path.exists(args.object_glb):
            raise FileNotFoundError(args.object_glb)
        shutil.copy2(args.object_glb, object_dst)
        object_name = "object_meshes.glb"
    if args.alternate_gaussians:
        if not os.path.exists(args.alternate_gaussians):
            raise FileNotFoundError(args.alternate_gaussians)
        if args.no_prune_splats:
            shutil.copy2(args.alternate_gaussians, alternate_dst)
        else:
            original_count, filtered_count = prune_splats(
                args.alternate_gaussians,
                alternate_dst,
                args.min_splat_opacity,
                args.max_splat_scale_percentile,
            )
            print(f"[INFO] Alternate splat pruning: {original_count:,} -> {filtered_count:,}")
        clamped_count = clamp_splat_scales(alternate_dst, args.max_overlay_splat_scale)
        if clamped_count:
            print(f"[INFO] Alternate overlay scale clamp adjusted {clamped_count:,} splats")
        alternate_name = alternate_splat_name
    if args.tracked_gaussians:
        if os.path.exists(args.tracked_gaussians):
            if args.no_prune_splats:
                shutil.copy2(args.tracked_gaussians, tracked_dst)
            else:
                original_count, filtered_count = prune_splats(
                    args.tracked_gaussians,
                    tracked_dst,
                    args.min_splat_opacity,
                    args.max_splat_scale_percentile,
                )
                print(f"[INFO] Tracked splat pruning: {original_count:,} -> {filtered_count:,}")
            clamped_count = clamp_splat_scales(tracked_dst, args.max_overlay_splat_scale)
            if clamped_count:
                print(f"[INFO] Tracked overlay scale clamp adjusted {clamped_count:,} splats")
            tracked_name = tracked_splat_name
            tracks_npz = args.tracks_npz
            if not tracks_npz:
                sibling = os.path.join(os.path.dirname(args.tracked_gaussians), "gaussian_tracks.npz")
                if os.path.exists(sibling):
                    tracks_npz = sibling
            if tracks_npz:
                if not os.path.exists(tracks_npz):
                    print(f"[WARN] Tracked metadata not found, skipping hover IDs: {tracks_npz}")
                else:
                    tracks_json = os.path.join(os.path.dirname(tracks_npz), "tracks.json")
                    num_objects = write_track_metadata(track_metadata_dst, tracks_npz, tracks_json)
                    print(f"[INFO] Wrote tracked hover metadata for {num_objects} objects")
                    track_metadata_name = "tracked_objects.json"
                    class_layers = write_class_splat_layers(args.out_dir, tracked_dst, tracks_npz, track_metadata_dst)
                    if class_layers:
                        print(f"[INFO] Wrote {len(class_layers)} class-filtered tracked splat layer(s)")
                    cameras_src = os.path.join(os.path.dirname(tracks_npz), "cameras.json")
                    if os.path.exists(cameras_src):
                        shutil.copy2(cameras_src, os.path.join(args.out_dir, "cameras.json"))
                        print(f"[INFO] Copied camera metadata for viewer")
                    # Per-object meshes (if generated by extract_object_meshes.py)
                    mesh_dir = getattr(args, "mesh_dir", "") or os.path.join(os.path.dirname(os.path.dirname(tracks_npz)), "object_meshes")
                    mesh_summary_path = os.path.join(mesh_dir, "summary.json")
                    if os.path.exists(mesh_summary_path):
                        try:
                            with open(mesh_summary_path) as f:
                                mesh_summary = json.load(f)
                            mesh_entries = []
                            with open(track_metadata_dst) as f:
                                tracked_meta = json.load(f)
                            obj_by_id = {int(o["id"]): o for o in tracked_meta.get("objects", [])}
                            for rec in mesh_summary.get("meshes", []):
                                obj_id_i = int(rec["id"])
                                # Prefer textured OBJ + PNG when available, else fall back to PLY
                                obj_src = rec.get("mesh_obj", "")
                                tex_src = rec.get("texture_png", "")
                                mtl_src = rec.get("mesh_mtl", "")
                                fname = None
                                tex_fname = None
                                if obj_src and os.path.exists(obj_src) and tex_src and os.path.exists(tex_src):
                                    fname = f"object_{obj_id_i:04d}_mesh.obj"
                                    tex_fname = f"object_{obj_id_i:04d}_texture.png"
                                    mtl_fname = f"object_{obj_id_i:04d}_mesh.mtl"
                                    shutil.copy2(obj_src, os.path.join(args.out_dir, fname))
                                    shutil.copy2(tex_src, os.path.join(args.out_dir, tex_fname))
                                    if mtl_src and os.path.exists(mtl_src):
                                        shutil.copy2(mtl_src, os.path.join(args.out_dir, mtl_fname))
                                else:
                                    src = rec.get("mesh_ply", "")
                                    if not src or not os.path.exists(src):
                                        continue
                                    fname = f"object_{obj_id_i:04d}_mesh.ply"
                                    shutil.copy2(src, os.path.join(args.out_dir, fname))
                                obj_meta = obj_by_id.get(obj_id_i, {})
                                mesh_entries.append({
                                    "id": obj_id_i,
                                    "file": fname,
                                    "texture": tex_fname,
                                    "n_vertices": int(rec.get("n_vertices", 0)),
                                    "n_triangles": int(rec.get("n_triangles", 0)),
                                    "num_splats": obj_meta.get("num_splats"),
                                    "class_label": obj_meta.get("class_label"),
                                    "source_prompts": obj_meta.get("source_prompts", []),
                                })
                            with open(os.path.join(args.out_dir, "object_meshes.json"), "w") as f:
                                json.dump({"meshes": mesh_entries}, f)
                            print(f"[INFO] Copied {len(mesh_entries)} object meshes for viewer")
                        except Exception as exc:
                            print(f"[WARN] Could not copy object meshes: {exc}")
        else:
            print(f"[WARN] Tracked splats not found, skipping layer: {args.tracked_gaussians}")
    if getattr(args, "refined_gaussians", ""):
        if os.path.exists(args.refined_gaussians):
            if (
                getattr(args, "tracks_npz", "")
                and os.path.exists(args.tracks_npz)
                and os.path.getmtime(args.refined_gaussians) < os.path.getmtime(args.tracks_npz)
            ):
                print(
                    f"[WARN] Refined splats are older than current tracks, skipping stale layer: "
                    f"{args.refined_gaussians}"
                )
            else:
                if args.no_prune_splats:
                    shutil.copy2(args.refined_gaussians, refined_dst)
                else:
                    original_count, filtered_count = prune_splats(
                        args.refined_gaussians,
                        refined_dst,
                        args.min_splat_opacity,
                        args.max_splat_scale_percentile,
                    )
                    print(f"[INFO] Refined splat pruning: {original_count:,} -> {filtered_count:,}")
                clamped_count = clamp_splat_scales(refined_dst, args.max_overlay_splat_scale)
                if clamped_count:
                    print(f"[INFO] Refined overlay scale clamp adjusted {clamped_count:,} splats")
                refined_name = refined_splat_name
                # Per-class refined splat layers (mirrors tracked class layers)
                if track_metadata_name and class_layers:
                    refined_object_dir = os.path.dirname(args.refined_gaussians)
                    class_layers = write_refined_class_layers(
                        args.out_dir,
                        refined_object_dir,
                        track_metadata_dst,
                        class_layers,
                    )
                    refined_class_count = sum(1 for l in class_layers if l.get("refined_file"))
                    print(f"[INFO] Wrote {refined_class_count} class-filtered refined splat layer(s)")
        else:
            print(f"[WARN] Refined splats not found, skipping layer: {args.refined_gaussians}")
    if args.fitted_glb:
        if not os.path.exists(args.fitted_glb):
            raise FileNotFoundError(args.fitted_glb)
        shutil.copy2(args.fitted_glb, fitted_dst)
        fitted_name = "fitted_shapes.glb"
    write_html(
        os.path.join(args.out_dir, "index.html"),
        scene_splat_name,
        alternate_name,
        tracked_name,
        refined_name,
        track_metadata_name,
        fitted_name,
        args.scene_label,
        args.alternate_label,
        args.tracked_label,
        getattr(args, "refined_label", "Dense Segmented Gaussians"),
        args.fitted_label,
        object_name,
        args.object_opacity,
        args.fitted_opacity,
        args.transparent_fitted,
        class_layers,
        splat_bounds(splat_dst),
        initial_camera_from_colmap(args.colmap_images),
        args.flip_y,
        args.flip_z,
        not args.no_helpers,
    )
    print(f"[DONE] Wrote Gaussian overlay viewer to {os.path.join(args.out_dir, 'index.html')}")


if __name__ == "__main__":
    main()

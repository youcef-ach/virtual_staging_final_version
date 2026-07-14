"""Headless Blender G-buffer renderer (Stage 7 / G4).

Bakes per-face 32-bit EXRs from the proxy (shell + details GLBs):
    <out>/<sweep_uuid>/<face>_depth.exr    Cycles Depth pass (ray length, m)
    <out>/<sweep_uuid>/<face>_normal.exr   Cycles Normal pass (world space,
                                           Blender Z-up == Matterport frame)
plus .npy siblings of each (loaded back from the saved EXR via bpy --
headless-safe -- and flipped to cv2 top-left orientation), because Windows
opencv-python builds often ship without OpenEXR support.

Blender 4/5 compatible: uses the compositing_node_group API on 5.x and
scene.node_tree on 4.x (same dual-API pattern as render_edges.py). Passes
route to the scene render output (write_still=True); no File Output nodes.

The glTF importer's Y-up -> Z-up conversion lands the runtime-Y-up proxy
exactly in the Matterport frame, so gbuffer_jobs.json camera matrices
(from vs_pipeline.stages.gbuffer prepare) apply directly.

Usage:
    blender -b -P pipeline/blender/render_gbuffer.py -- \
        --jobs gbuffer_jobs.json --shell proxy_out/proxy_shell.glb \
        --details details_out/proxy_details.glb --out gbuffer_renders \
        [--limit N]

Resumable: sweeps whose outputs all exist are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shell", type=Path, required=True)
    parser.add_argument("--details", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def setup_scene(size: int):
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True

    if hasattr(scene, "compositing_node_group"):
        # Blender 5.x compositor node group API (render_edges.py pattern).
        if scene.compositing_node_group is None:
            scene.compositing_node_group = bpy.data.node_groups.new(
                "Compositor", "CompositorNodeTree"
            )
        tree = scene.compositing_node_group
        for node in list(tree.nodes):
            tree.nodes.remove(node)
        rl = tree.nodes.new("CompositorNodeRLayers")
        group_output = tree.nodes.new("NodeGroupOutput")
        tree.interface.new_socket(
            name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
        )
        out_socket = group_output.inputs["Image"]
    else:
        # Blender 4.x compositor API.
        scene.use_nodes = True
        tree = scene.node_tree
        for node in list(tree.nodes):
            tree.nodes.remove(node)
        rl = tree.nodes.new("CompositorNodeRLayers")
        composite = tree.nodes.new("CompositorNodeComposite")
        out_socket = composite.inputs["Image"]

    cam_data = bpy.data.cameras.new("gbuffer_cam")
    cam_data.type = "PERSP"
    cam_data.angle = math.radians(90.0)
    cam_data.clip_start = 0.01
    cam_data.clip_end = 200.0
    cam = bpy.data.objects.new("gbuffer_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return scene, cam, tree, rl, out_socket


def route_pass(scene, tree, rl, out_socket, pass_name: str, color_mode: str) -> None:
    for link in list(tree.links):
        tree.links.remove(link)
    tree.links.new(rl.outputs[pass_name], out_socket)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = color_mode


def import_glb(path: Path) -> None:
    bpy.ops.import_scene.gltf(filepath=str(path))


def dump_npy(exr_path: Path, depth: bool) -> None:
    """Round-trip the saved EXR through bpy image loading -> .npy sibling.

    Headless-safe (reads from disk, not the render result). Blender pixel
    rows start bottom-left; flip to cv2's top-left orientation.
    """
    img = bpy.data.images.load(str(exr_path))
    try:
        w, h = img.size
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)
        arr = np.flipud(arr)
        data = arr[..., 0] if depth else arr[..., :3]
        np.save(str(exr_path.with_suffix(".npy")), data)
    finally:
        bpy.data.images.remove(img)


def main() -> None:
    args = parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    size = int(jobs.get("render_size", 2048))
    scene, cam, tree, rl, out_socket = setup_scene(size)
    import_glb(args.shell)
    if args.details is not None and Path(args.details).is_file():
        import_glb(Path(args.details))

    depth_socket = "Depth" if "Depth" in rl.outputs else "Z"

    # Frame-lock sanity: cameras must sit INSIDE the imported proxy bounds.
    mins = [1e12, 1e12, 1e12]
    maxs = [-1e12, -1e12, -1e12]
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Matrix.Translation(corner).to_translation()
            for i in range(3):
                mins[i] = min(mins[i], world[i])
                maxs[i] = max(maxs[i], world[i])
    print(
        f"proxy bounds (Blender frame): "
        f"({mins[0]:+.2f},{mins[1]:+.2f},{mins[2]:+.2f}) .. "
        f"({maxs[0]:+.2f},{maxs[1]:+.2f},{maxs[2]:+.2f})"
    )

    out_root = Path(args.out).resolve()
    sweeps = jobs["sweeps"]
    if args.limit and args.limit > 0:
        sweeps = sweeps[: args.limit]
    out_root.mkdir(parents=True, exist_ok=True)

    for entry in sweeps:
        sweep = entry["sweep_uuid"]
        first_face = next(iter(entry["faces"].values()))
        cx, cy, cz = first_face[0][3], first_face[1][3], first_face[2][3]
        inside = all(
            mins[i] - 1.0 <= v <= maxs[i] + 1.0
            for i, v in enumerate((cx, cy, cz))
        )
        print(
            f"[{sweep}] camera ({cx:+.2f},{cy:+.2f},{cz:+.2f}) "
            f"{'INSIDE' if inside else 'OUTSIDE'} proxy bounds"
        )
        sweep_dir = out_root / sweep
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for face, matrix in entry["faces"].items():
            depth_target = sweep_dir / f"{face}_depth.exr"
            normal_target = sweep_dir / f"{face}_normal.exr"
            if depth_target.is_file() and normal_target.is_file():
                continue
            cam.matrix_world = Matrix(matrix)
            route_pass(scene, tree, rl, out_socket, depth_socket, "BW")
            scene.render.filepath = str(depth_target)
            bpy.ops.render.render(write_still=True)
            dump_npy(depth_target, depth=True)
            route_pass(scene, tree, rl, out_socket, "Normal", "RGB")
            scene.render.filepath = str(normal_target)
            bpy.ops.render.render(write_still=True)
            dump_npy(normal_target, depth=False)
        print(f"[{sweep}] done")
    print("gbuffer render complete")


if __name__ == "__main__":
    main()

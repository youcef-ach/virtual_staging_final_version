"""Headless Blender G-buffer renderer (Stage 7 / G4).

Bakes per-face 32-bit EXRs from the proxy (shell + details GLBs):
    <out>/<sweep_uuid>/<face>_depth.exr    Cycles Depth pass (ray length, m)
    <out>/<sweep_uuid>/<face>_normal.exr   Cycles Normal pass (world space,
                                           Blender Z-up == Matterport frame)

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
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shell", type=Path, required=True)
    parser.add_argument("--details", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def setup_scene(size: int) -> tuple[object, object, object]:
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True

    cam_data = bpy.data.cameras.new("gbuffer_cam")
    cam_data.type = "PERSP"
    cam_data.angle = 1.5707963267948966  # 90 degrees
    cam_data.clip_start = 0.01
    cam_data.clip_end = 200.0
    cam = bpy.data.objects.new("gbuffer_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    out_depth = tree.nodes.new("CompositorNodeOutputFile")
    out_depth.name = "gbuffer_depth_out"
    out_depth.format.file_format = "OPEN_EXR"
    out_depth.format.color_depth = "32"
    out_depth.format.color_mode = "BW"
    out_normal = tree.nodes.new("CompositorNodeOutputFile")
    out_normal.name = "gbuffer_normal_out"
    out_normal.format.file_format = "OPEN_EXR"
    out_normal.format.color_depth = "32"
    out_normal.format.color_mode = "RGB"
    depth_socket = "Depth" if "Depth" in rl.outputs else "Z"
    tree.links.new(rl.outputs[depth_socket], out_depth.inputs[0])
    tree.links.new(rl.outputs["Normal"], out_normal.inputs[0])
    return cam, out_depth, out_normal


def import_glb(path: Path) -> None:
    bpy.ops.import_scene.gltf(filepath=str(path))


def finalize_output(node: object, target: Path) -> None:
    """File Output nodes append the frame number; rename to the target."""
    frame = bpy.context.scene.frame_current
    base = Path(node.base_path)
    slot = node.file_slots[0].path
    produced = base / f"{slot}{frame:04d}.exr"
    if produced.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        produced.rename(target)


def main() -> None:
    args = parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    size = int(jobs.get("render_size", 2048))
    cam, out_depth, out_normal = setup_scene(size)
    import_glb(args.shell)
    if args.details is not None and Path(args.details).is_file():
        import_glb(Path(args.details))

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

    sweeps = jobs["sweeps"]
    if args.limit is not None:
        sweeps = sweeps[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_current = 1

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
        sweep_dir = args.out / sweep
        targets = [
            (face, sweep_dir / f"{face}_depth.exr", sweep_dir / f"{face}_normal.exr")
            for face in entry["faces"]
        ]
        if all(d.is_file() and n.is_file() for _, d, n in targets):
            print(f"[{sweep}] complete -> skip")
            continue
        for face, depth_target, normal_target in targets:
            if depth_target.is_file() and normal_target.is_file():
                continue
            cam.matrix_world = Matrix(entry["faces"][face])
            tmp = sweep_dir / "_tmp"
            out_depth.base_path = str(tmp)
            out_depth.file_slots[0].path = f"{face}_depth_"
            out_normal.base_path = str(tmp)
            out_normal.file_slots[0].path = f"{face}_normal_"
            bpy.ops.render.render(write_still=False)
            finalize_output(out_depth, depth_target)
            finalize_output(out_normal, normal_target)
        print(f"[{sweep}] done")
    print("gbuffer render complete")


if __name__ == "__main__":
    main()

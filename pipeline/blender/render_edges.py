"""Headless Blender renderer for Gate 1 pose QA.

Renders a normalized depth image (16-bit PNG) per sweep face using 90-degree
pinhole cameras from render_jobs.json (produced by vs_pipeline.stages.pose_qa
prepare). Geometry discontinuities in these depth maps are compared against
panorama edges by the scorer.

Requires Blender 4.x. Usage (Windows: use the full path to blender.exe if not
on PATH):

    blender --background --python pipeline/blender/render_edges.py -- \
        --obj path/to/matterpak.obj --jobs render_jobs.json --out qa_renders \
        [--limit 10] [--obj-up Y|Z]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def _script_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Render depth faces for pose QA.")
    parser.add_argument("--obj", required=True, help="path to the Matterpak .obj")
    parser.add_argument("--jobs", required=True, help="render_jobs.json")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=0, help="render only the first N sweeps")
    parser.add_argument(
        "--obj-up",
        choices=["Y", "Z"],
        default="Y",
        help="up axis of the OBJ file (Matterpak default: Y)",
    )
    args = parser.parse_args(_script_argv())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if args.obj_up == "Z":
        bpy.ops.wm.obj_import(filepath=args.obj, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.wm.obj_import(filepath=args.obj)  # Blender default: -Z forward, Y up

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    size = int(jobs.get("render_size", 512))

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "16"

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True

    # Compositor: Depth -> Normalize -> Composite (per-image normalized depth;
    # only discontinuities matter for edge scoring, not metric values).
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    normalize = tree.nodes.new("CompositorNodeNormalize")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Depth"], normalize.inputs[0])
    tree.links.new(normalize.outputs[0], composite.inputs["Image"])

    cam_data = bpy.data.cameras.new("qa_cam")
    cam_data.angle = math.radians(90.0)  # cube-face FOV
    cam_data.clip_start = 0.05
    cam_data.clip_end = 200.0
    cam = bpy.data.objects.new("qa_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    out_root = Path(args.out)
    sweeps = jobs["sweeps"]
    if args.limit > 0:
        sweeps = sweeps[: args.limit]

    total = len(sweeps)
    for i, sweep in enumerate(sweeps):
        sweep_dir = out_root / sweep["sweep_uuid"]
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for face, matrix_rows in sweep["faces"].items():
            cam.matrix_world = Matrix(matrix_rows)
            scene.render.filepath = str(sweep_dir / f"{face}.png")
            bpy.ops.render.render(write_still=True)
        print(f"[{i + 1}/{total}] {sweep['sweep_uuid']}")

    print(f"done: {total} sweep(s) -> {out_root}")


if __name__ == "__main__":
    main()

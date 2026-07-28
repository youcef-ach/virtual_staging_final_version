"""Stage for Catalog Asset Normalization (P1).

Normalizes raw glTF/GLB models for catalog use:
- Validates model structure and geometry.
- Scales model to meters (if source model uses mm/cm or non-standard scale).
- Recenters origin to the bottom-center of the model's bounding box.
- Generates LOD meshes (LOD0 original, LOD1 decimated, LOD2 low-poly).
- Applies Draco geometry compression & KTX2/Basis UASTC texture processing via gltf-transform CLI when available.
- Computes AABB bounding box, dimensions (in mm), polygon counts.
- Generates thumbnail PNG.
- Emits normalized metadata.json + LOD files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def apply_gltf_transform_optimizations(file_path: Path) -> None:
    """Apply gltf-transform draco compression and KTX2 texture processing if CLI is available."""
    try:
        tmp_out = file_path.with_name(f"{file_path.stem}_opt.glb")
        cmd = [
            "npx", "-y", "@gltf-transform/cli@3.10.1", "optimize", 
            str(file_path), str(tmp_out), 
            "--compress", "draco", 
            "--texture-compress", "ktx2"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if res.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
            shutil.move(tmp_out, file_path)
            logger.info(f"Draco and KTX2 compression applied via gltf-transform: {file_path.name}")
        else:
            if tmp_out.exists():
                tmp_out.unlink()
            logger.debug(f"gltf-transform skipped or returned non-zero code ({res.returncode}): {res.stderr}")
    except Exception as err:
        logger.debug(f"gltf-transform optimization unavailable: {err}")


def generate_thumbnail(scene: trimesh.Scene, output_path: Path) -> None:
    """Generate a 512x512 thumbnail image for a catalog asset."""
    try:
        png_data = scene.save_image(resolution=(512, 512))
        if png_data:
            with open(output_path, "wb") as f:
                f.write(png_data)
            return
    except Exception as err:
        logger.debug(f"Offscreen render unavailable ({err}), generating fallback thumbnail")

    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (512, 512), (240, 242, 245, 255))
        draw = ImageDraw.Draw(img)

        # Draw border & subtle gradient box
        draw.rectangle([16, 16, 496, 496], outline=(200, 205, 215), width=2)
        draw.rectangle([32, 32, 480, 480], fill=(255, 255, 255, 255), outline=(220, 225, 235), width=1)

        bounds = scene.bounds
        if bounds is not None:
            min_c, max_c = bounds[0], bounds[1]
            dx = max_c[0] - min_c[0]
            dy = max_c[1] - min_c[1]
            dz = max_c[2] - min_c[2]

            cx, cy = 256, 280
            scale = 160.0 / max(max(dx, dy), dz, 0.001)

            def iso_proj(x: float, y: float, z: float) -> tuple[float, float]:
                px = cx + (x - z) * 0.866 * scale
                py = cy - y * scale + (x + z) * 0.5 * scale * 0.5
                return (px, py)

            hx, hy, hz = dx / 2.0, dy, dz / 2.0
            p000 = iso_proj(-hx, 0, -hz)
            p100 = iso_proj(hx, 0, -hz)
            p101 = iso_proj(hx, 0, hz)
            p001 = iso_proj(-hx, 0, hz)
            p010 = iso_proj(-hx, hy, -hz)
            p110 = iso_proj(hx, hy, -hz)
            p111 = iso_proj(hx, hy, hz)
            p011 = iso_proj(-hx, hy, hz)

            draw.polygon([p010, p110, p111, p011], fill=(220, 230, 245), outline=(90, 120, 180), width=2)
            draw.polygon([p001, p101, p111, p011], fill=(190, 205, 230), outline=(90, 120, 180), width=2)
            draw.polygon([p100, p101, p111, p110], fill=(170, 190, 220), outline=(90, 120, 180), width=2)

        draw.text((48, 440), "CATALOG ASSET", fill=(100, 110, 130))
        img.save(output_path, "PNG")
    except Exception as err:
        logger.warning(f"Could not save thumbnail: {err}")


def normalize_catalog_asset(
    input_path: str | Path,
    output_dir: str | Path,
    sku: str = "ASSET-001",
    category: str = "furniture",
    manufacturer: str = "In-House",
    product_name: str = "Catalog Item",
    price: float | None = None,
    scale_factor: float | None = None,
    optimize: bool = True,
) -> dict[str, Any]:
    """Normalize a glTF/GLB asset and produce LODs, thumbnail, and metadata JSON."""
    input_p = Path(input_path).resolve()
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    if not input_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_p}")

    logger.info(f"Normalizing catalog asset: {input_p.name} -> SKU {sku}")

    # Load geometry using trimesh
    scene = trimesh.load(str(input_p), force="scene")
    bounds = scene.bounds
    if bounds is None:
        raise ValueError("Could not compute bounding box for asset geometry")

    min_corner, max_corner = bounds[0], bounds[1]
    extents = max_corner - min_corner

    # Auto-detect scale factor if not specified
    max_extent = float(np.max(extents))
    if scale_factor is None:
        if max_extent > 100.0:
            scale_factor = 0.001  # mm -> meters
        elif max_extent > 20.0:
            scale_factor = 0.01   # cm -> meters
        else:
            scale_factor = 1.0    # meters

    # Scale scene if needed
    if abs(scale_factor - 1.0) > 1e-6:
        transform = trimesh.transformations.scale_matrix(scale_factor)
        scene.apply_transform(transform)
        bounds = scene.bounds
        min_corner, max_corner = bounds[0], bounds[1]
        extents = max_corner - min_corner

    # Bottom-center pivot alignment:
    # Target pivot is bottom-center of AABB at Y=0, X=0, Z=0.
    center_x = (min_corner[0] + max_corner[0]) / 2.0
    bottom_y = min_corner[1]
    center_z = (min_corner[2] + max_corner[2]) / 2.0

    translation = np.array([-center_x, -bottom_y, -center_z])
    translation_matrix = trimesh.transformations.translation_matrix(translation)
    scene.apply_transform(translation_matrix)

    # Re-evaluate bounds after alignment
    bounds = scene.bounds
    min_corner, max_corner = bounds[0], bounds[1]
    extents = max_corner - min_corner

    # Polygon count calculation
    poly_count = 0
    for geometry in scene.geometry.values():
        if hasattr(geometry, "faces"):
            poly_count += len(geometry.faces)

    # Export LOD0
    lod0_path = out_p / "lod0.glb"
    scene.export(str(lod0_path), file_type="glb")

    # Generate LOD1 and LOD2 decimation
    lod1_path = out_p / "lod1.glb"
    lod2_path = out_p / "lod2.glb"

    try:
        scene_lod1 = scene.copy()
        for name, geom in list(scene_lod1.geometry.items()):
            if hasattr(geom, "simplify_quadratic_decimation") and len(geom.faces) > 2000:
                target_faces = max(1000, int(len(geom.faces) * 0.4))
                try:
                    scene_lod1.geometry[name] = geom.simplify_quadratic_decimation(target_faces)
                except Exception:
                    pass
        scene_lod1.export(str(lod1_path), file_type="glb")
    except Exception as err:
        logger.warning(f"LOD1 decimation fallback: {err}")
        shutil.copy(lod0_path, lod1_path)

    try:
        scene_lod2 = scene.copy()
        for name, geom in list(scene_lod2.geometry.items()):
            if hasattr(geom, "simplify_quadratic_decimation") and len(geom.faces) > 1000:
                target_faces = max(500, int(len(geom.faces) * 0.15))
                try:
                    scene_lod2.geometry[name] = geom.simplify_quadratic_decimation(target_faces)
                except Exception:
                    pass
        scene_lod2.export(str(lod2_path), file_type="glb")
    except Exception as err:
        logger.warning(f"LOD2 decimation fallback: {err}")
        shutil.copy(lod0_path, lod2_path)

    # Optional gltf-transform Draco compression
    if optimize:
        apply_gltf_transform_optimizations(lod0_path)
        apply_gltf_transform_optimizations(lod1_path)
        apply_gltf_transform_optimizations(lod2_path)

    # Generate Thumbnail PNG
    thumb_path = out_p / "thumbnail.png"
    generate_thumbnail(scene, thumb_path)

    # Metadata payload
    dimensions_mm = {
        "x": round(float(extents[0] * 1000.0), 1),
        "y": round(float(extents[1] * 1000.0), 1),
        "z": round(float(extents[2] * 1000.0), 1),
    }

    aabb = {
        "min": [round(float(min_corner[i]), 4) for i in range(3)],
        "max": [round(float(max_corner[i]), 4) for i in range(3)],
    }

    metadata: dict[str, Any] = {
        "sku": sku,
        "manufacturer": manufacturer,
        "product_name": product_name,
        "category": category,
        "price": price,
        "polygon_count": poly_count,
        "dimensions_mm": dimensions_mm,
        "bounding_box": aabb,
        "lod_urls": {
            "lod0": "lod0.glb",
            "lod1": "lod1.glb",
            "lod2": "lod2.glb",
        },
        "thumbnail_url": "thumbnail.png",
    }

    meta_path = out_p / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Catalog asset successfully normalized: {sku} ({poly_count} tris) -> {out_p}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize catalog asset glTF/GLB")
    parser.add_argument("--input", "-i", required=True, help="Input glTF or GLB file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--sku", default="SOFA-001", help="Stock Keeping Unit identifier")
    parser.add_argument("--category", default="sofas", help="Asset category (sofa, chair, table, etc.)")
    parser.add_argument("--manufacturer", default="Nordic Living", help="Manufacturer name")
    parser.add_argument("--product-name", default="Modern Sofa", help="Product name")
    parser.add_argument("--price", type=float, default=899.0, help="Item price")
    parser.add_argument("--scale", type=float, default=None, help="Explicit scale factor to meters")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    normalize_catalog_asset(
        input_path=args.input,
        output_dir=args.output_dir,
        sku=args.sku,
        category=args.category,
        manufacturer=args.manufacturer,
        product_name=args.product_name,
        price=args.price,
        scale_factor=args.scale,
    )


if __name__ == "__main__":
    main()

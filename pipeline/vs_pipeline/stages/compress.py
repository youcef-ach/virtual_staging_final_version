"""Stage 10: Compress and Publish.

Generates compressed artifacts (KTX2/Draco) with graceful degradation (passthrough)
if CLI tools (gltf-transform, toktx) are missing. Uploads the final bundle using
the configured StorageProvider (S3 or local filesystem) and emits a manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import concurrent.futures
from pathlib import Path

from vs_pipeline.storage import get_storage_provider, StorageProvider

logger = logging.getLogger(__name__)


def check_tool(name: str, test_arg: str = "--version") -> bool:
    """Check if a CLI tool is available."""
    try:
        subprocess.run(
            [name, test_arg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        return True
    except (FileNotFoundError, PermissionError):
        return False


def compress_glb(input_path: Path, output_path: Path, has_draco: bool) -> bool:
    """Compress GLB using gltf-transform if available."""
    if not has_draco:
        return False
    try:
        cmd = ["gltf-transform", "optimize", str(input_path), str(output_path), "--compress", "draco"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        logger.warning("gltf-transform failed for %s, falling back to uncompressed: %s", input_path.name, e)
        return False


def compress_texture_ktx2(input_path: Path, output_path: Path, has_ktx2: bool) -> bool:
    """Compress EXR or PNG texture to KTX2 UASTC using toktx."""
    if not has_ktx2:
        return False
    try:
        # --t2 for KTX2, --uastc for universal ASTC, --zcmp for supercompression
        cmd = ["toktx", "--t2", "--uastc", "3", "--zcmp", "15", str(output_path), str(input_path)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        logger.warning("toktx failed for %s: %s", input_path.name, e)
        return False

def _process_cubemap_face(args):
    face_img, output_dir, has_ktx2, sweep_name, storage = args
    compressed_path = output_dir / f"{sweep_name}_{face_img.stem}.ktx2"
    if compress_texture_ktx2(face_img, compressed_path, has_ktx2):
        url = storage.upload_file(compressed_path, f"cubemaps/{sweep_name}/{face_img.stem}.ktx2", "image/ktx2")
        return face_img.stem, f"{face_img.stem}.ktx2", url
    else:
        url = storage.upload_file(face_img, f"cubemaps/{sweep_name}/{face_img.name}", "image/jpeg")
        return face_img.stem, face_img.name, url


def publish_property(property_dir: Path, output_dir: Path) -> None:
    """Execute the compress & publish stage."""
    start_t = time.monotonic()

    proxy_dir = property_dir / "proxy_out"
    details_dir = property_dir / "details_out"
    env_dir = property_dir / "env_out"
    cubemaps_dir = property_dir / "extracted_cubemaps"
    
    if not proxy_dir.exists():
        raise FileNotFoundError(f"Missing proxy_out in {property_dir}")

    has_draco = check_tool("gltf-transform")
    has_ktx2 = check_tool("toktx")

    logger.info("Compression tools - Draco (gltf-transform): %s, KTX2 (toktx): %s", has_draco, has_ktx2)

    storage = get_storage_provider()
    manifest: dict[str, str] = {}

    # 1. Publish Proxy Shell
    proxy_shell_src = proxy_dir / "proxy_shell.glb"
    if proxy_shell_src.exists():
        upload_path = proxy_shell_src
        compressed_path = output_dir / "proxy_shell_draco.glb"
        if compress_glb(proxy_shell_src, compressed_path, has_draco):
            upload_path = compressed_path
            logger.info("Compressed proxy shell")
        
        manifest["proxy_shell"] = storage.upload_file(upload_path, "proxy_shell.glb", "model/gltf-binary")
    
    # 2. Publish Details Mesh
    proxy_details_src = details_dir / "proxy_details.glb"
    if proxy_details_src.exists():
        upload_path = proxy_details_src
        compressed_path = output_dir / "proxy_details_draco.glb"
        if compress_glb(proxy_details_src, compressed_path, has_draco):
            upload_path = compressed_path
            logger.info("Compressed details mesh")

        manifest["proxy_details"] = storage.upload_file(upload_path, "proxy_details.glb", "model/gltf-binary")

    # 3. Publish IBL envs (passing through .env files for Babylon, and compressing .exr/.png to .ktx2)
    if env_dir.exists():
        for env_file in env_dir.glob("*.*"):
            if env_file.suffix == ".env":
                url = storage.upload_file(env_file, f"env/{env_file.name}", "application/octet-stream")
                manifest[f"env_{env_file.stem}"] = url
            elif env_file.suffix in (".exr", ".png"):
                compressed_path = output_dir / f"{env_file.stem}.ktx2"
                if compress_texture_ktx2(env_file, compressed_path, has_ktx2):
                    url = storage.upload_file(compressed_path, f"env/{compressed_path.name}", "image/ktx2")
                    manifest[f"ktx2_{env_file.stem}"] = url
                else:
                    mime = "image/png" if env_file.suffix == ".png" else "image/x-exr"
                    url = storage.upload_file(env_file, f"env/{env_file.name}", mime)
                    manifest[f"raw_{env_file.stem}"] = url

    # 4. Publish Cubemaps (compressing .jpg to .ktx2)
    if cubemaps_dir.exists():
        manifest_path_in = cubemaps_dir / "manifest.json"
        if manifest_path_in.exists():
            pano_manifest = json.loads(manifest_path_in.read_text(encoding="utf-8"))
            
            # Prepare jobs
            jobs = []
            for sweep in pano_manifest.get("sweeps", []):
                sweep_uuid = sweep["sweep_uuid"]
                sweep_dir = cubemaps_dir / sweep_uuid
                if not sweep_dir.exists():
                    continue
                for face_img in sweep_dir.glob("*.jpg"):
                    jobs.append((face_img, output_dir, has_ktx2, sweep_uuid, storage))
                    
            logger.info("Compressing and uploading %d cubemap faces (parallel)...", len(jobs))
            
            # Execute in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(_process_cubemap_face, jobs))
                
            # Update manifest with new paths and CDN urls
            results_dict = {} # key: (sweep_uuid, face_stem) -> (new_file_name, url)
            for j, res in zip(jobs, results):
                sweep_uuid = j[3]
                face_stem, new_file_name, url = res
                results_dict[(sweep_uuid, face_stem)] = (new_file_name, url)
                manifest[f"skybox_{sweep_uuid}_{face_stem}"] = url
                
            for sweep in pano_manifest.get("sweeps", []):
                sweep_uuid = sweep["sweep_uuid"]
                for face_key, old_path in list(sweep["faces"].items()):
                    # old_path is something like "uuid/nx.jpg"
                    face_stem = Path(old_path).stem
                    if (sweep_uuid, face_stem) in results_dict:
                        new_file_name, _ = results_dict[(sweep_uuid, face_stem)]
                        sweep["faces"][face_key] = f"{sweep_uuid}/{new_file_name}"
            
            # Upload the modified panorama manifest as well so frontend can get it
            modified_manifest_path = output_dir / "manifest.json"
            modified_manifest_path.write_text(json.dumps(pano_manifest, indent=2))
            url = storage.upload_file(modified_manifest_path, "manifest.json", "application/json")
            manifest["panorama_manifest"] = url
            logger.info("Updated and published panorama manifest")

    # Write local manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "published_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    duration = time.monotonic() - start_t
    logger.info("Published %d files in %.1fs", len(manifest), duration)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Compress & Publish pipeline stage")
    parser.add_argument("command", choices=["publish"])
    parser.add_argument("--input", required=True, type=Path, help="Property pipeline dir")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Local scratch output dir")
    
    args = parser.parse_args()
    try:
        publish_property(args.input, args.output)
        return 0
    except Exception as e:
        logger.error("Publish failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

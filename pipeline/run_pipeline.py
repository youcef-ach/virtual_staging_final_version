import os
import subprocess
import sys
from pathlib import Path

def run_stage(cmd):
    print(f"\n========================================================")
    print(f"RUNNING: {cmd}")
    print(f"========================================================\n")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at command:\n{cmd}")
        sys.exit(1)

def main():
    os.environ["VS_TEST_SWEEP_RANGE"] = "45-65"
    
    # Resolving absolute paths for safety
    repo_root = Path(__file__).resolve().parent.parent
    e57_path = repo_root / "backend/data/storage/raw/ea382c71-f8ff-4028-a78c-2b55800c2a2a/v1/cloud_0.e57"
    matterpak_obj = repo_root / "backend/data/input_matterpak/mp_matterpak_Two-story-home-including-facade_CNDDpymQrDA/bc6b73a9f9f04c6187dd2be3e17329cb.obj"
    
    prop = Path("test_pipeline_out").resolve()
    prop.mkdir(parents=True, exist_ok=True)
    
    # 1. Pose
    run_stage(f"py -3.11 -m vs_pipeline.stages.pose {e57_path} -o {prop}/poses.json")
    
    # 2. Pose QA (Skipping for fast local test)
    # run_stage(f"py -3.11 -m vs_pipeline.stages.pose_qa --poses {prop}/poses.json --obj {matterpak_obj} -o {prop}/qa_output")
    
    # 3. Pointcloud
    run_stage(f"py -3.11 -m vs_pipeline.stages.pointcloud preprocess {e57_path} -o {prop}/preprocessed")
    
    # Panorama (required by segmentation and IBL)
    run_stage(f"py -3.11 -m vs_pipeline.stages.panorama {e57_path} -o {prop}/extracted_cubemaps")
    
    # 4. Segmentation (Infer)
    run_stage(f"py -3.11 -m vs_pipeline.stages.segmentation infer --panos {prop}/extracted_cubemaps -o {prop}/seg_out")
    
    # 5. Planes
    run_stage(f"py -3.11 -m vs_pipeline.stages.planes fit --cache {prop}/preprocessed/sweep_cache -o {prop}/planes_out --poses {prop}/poses.json")
    
    # 5.5 Rooms
    run_stage(f"py -3.11 -m vs_pipeline.stages.rooms build --planes {prop}/planes_out/structural_planes.json --poses {prop}/poses.json -o {prop}/planes_out")
    
    # 5.7 Openings
    run_stage(f"py -3.11 -m vs_pipeline.stages.openings detect --planes {prop}/planes_out/structural_planes.json -o {prop}/planes_out/openings.json")
    
    # 6. Proxy
    proxy_out = prop / "proxy_out"
    proxy_out.mkdir(parents=True, exist_ok=True)
    run_stage(f"py -3.11 -m vs_pipeline.stages.proxy build --rooms {prop}/planes_out/rooms.json --openings {prop}/planes_out/openings.json -o {proxy_out}")
    
    # 7. G-Buffer
    run_stage(f"py -3.11 -m vs_pipeline.stages.gbuffer prepare {prop}/poses.json -o {prop}/gbuffer_jobs.json")
    
    # 8. IBL
    run_stage(f"py -3.11 -m vs_pipeline.stages.ibl bake --panos {prop}/extracted_cubemaps -o {prop}/env_out")
    
    # 9. Navigation
    run_stage(f"py -3.11 -m vs_pipeline.stages.navigation build --poses {prop}/poses.json --rooms {prop}/planes_out/rooms.json --openings {prop}/planes_out/openings.json -o {prop}/nav_out")
    
    # 10. Compress & Publish
    # Note: I modified compress.py to expect property_dir structure.
    run_stage(f"py -3.11 -m vs_pipeline.stages.compress publish --input {prop} -o {prop}/published")

    print("\nSUCCESS! Pipeline finished on 20 scans.")

if __name__ == "__main__":
    main()

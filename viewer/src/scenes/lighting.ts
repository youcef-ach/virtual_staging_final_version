import "@babylonjs/core/Materials/Textures/Loaders/envTextureLoader";
import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial";
import { CreateSphere } from "@babylonjs/core/Meshes/Builders/sphereBuilder";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import type { Matrix } from "@babylonjs/core/Maths/math.vector";
import type { Scene } from "@babylonjs/core/scene";
import type { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import { GROUP_ASSET } from "./staging";

/** Babylon .env magic bytes (environmentTextureTools.js -- source-verified). */
const ENV_MAGIC = [0x86, 0x16, 0x87, 0x96, 0xf6, 0xd6, 0x96, 0x36] as const;

/** The G7.1 stand-in lights created by StagingLayer (looked up by name). */
const TEMP_LIGHT_NAMES = ["temp-hemi-7_1", "temp-sun-7_1"] as const;

const CHROME_SPHERE_DISTANCE_M = 1.2;
const ENV_LOAD_TIMEOUT_MS = 15000;

/**
 * G7.2 -- per-sweep IBL: swaps scene.environmentTexture to the sweep's
 * exported `.env` (G5 products via ibl_export.py) on every sweep jump.
 *
 * Orientation: the .env cubemap faces are the SAME stored canonical faces as
 * the pano skybox (sweep-LOCAL frame), so the environment texture gets the
 * SAME reflection matrix as the skybox (TourViewer.getPanoOrientation()).
 * Babylon's PBR shader applies the reflection matrix to specular sampling AND
 * to the normal used for the SH-polynomial irradiance, so one matrix aligns
 * both light paths with the photograph.
 *
 * Fallback: sweeps without an exported .env fall back to the G7.1 stand-in
 * hemi+sun lights (honest, clearly reported) instead of keeping a stale
 * neighbor environment.
 */
export class IblLighting {
  private envTexture: CubeTexture | null = null;
  private chromeSphere: Mesh | null = null;
  private requestSeq = 0;

  constructor(private readonly scene: Scene) {}

  private setTempLights(on: boolean): void {
    for (const name of TEMP_LIGHT_NAMES) {
      this.scene.getLightByName(name)?.setEnabled(on);
    }
  }

  private clearEnvironment(): void {
    const previous = this.envTexture;
    this.scene.environmentTexture = null;
    this.envTexture = null;
    previous?.dispose();
  }

  /**
   * Load and activate the sweep's environment. Returns a short HUD status.
   * VRAM discipline: only the active sweep's env stays resident. A stale
   * response (operator already jumped again) is disposed, not applied.
   */
  async setForSweep(uuid: string, url: string, orientation: Matrix): Promise<string> {
    const seq = ++this.requestSeq;
    try {
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const buf = new Uint8Array(await resp.arrayBuffer());
      for (let i = 0; i < ENV_MAGIC.length; i += 1) {
        if (buf[i] !== ENV_MAGIC[i]) {
          throw new Error("not a Babylon .env (magic mismatch)");
        }
      }
      const blobUrl = URL.createObjectURL(new Blob([buf]));
      try {
        const texture = CubeTexture.CreateFromPrefilteredData(blobUrl, this.scene, ".env");
        await new Promise<void>((resolve, reject) => {
          const timer = window.setTimeout(
            () => reject(new Error("env load timeout")),
            ENV_LOAD_TIMEOUT_MS,
          );
          texture.onLoadObservable.addOnce(() => {
            window.clearTimeout(timer);
            resolve();
          });
        });
        if (seq !== this.requestSeq) {
          texture.dispose(); // stale: the operator already jumped to another sweep
          return "IBL (stale load discarded)";
        }
        texture.setReflectionTextureMatrix(orientation);
        const previous = this.envTexture;
        this.scene.environmentTexture = texture;
        this.envTexture = texture;
        previous?.dispose();
        this.setTempLights(false);
        console.info(`[ibl] environment active for sweep ${uuid}`);
        return "IBL \u2713";
      } finally {
        URL.revokeObjectURL(blobUrl);
      }
    } catch (error) {
      if (seq !== this.requestSeq) {
        return "IBL (stale load discarded)";
      }
      this.clearEnvironment();
      this.setTempLights(true);
      console.info(
        `[ibl] no environment for sweep ${uuid} (${error instanceof Error ? error.message : error}) -- stand-in lights`, 
      );
      return "IBL \u2717 fallback lights";
    }
  }

  /**
   * Chrome-sphere debug probe (C key) -- the G8 gate instrument: a metallic
   * roughness-0 sphere must show the room's window in the direction of the
   * REAL window (orientation proof for the whole IBL chain). Repositioned
   * ~1.2 m ahead of the camera on every toggle-on.
   */
  toggleChromeSphere(camera: FreeCamera): boolean {
    if (!this.chromeSphere) {
      const sphere = CreateSphere(
        "chrome-sphere",
        { diameter: 0.4, segments: 48 },
        this.scene,
      );
      const material = new PBRMaterial("chrome-sphere-mat", this.scene);
      material.metallic = 1.0;
      material.roughness = 0.0;
      material.albedoColor = new Color3(1, 1, 1);
      sphere.material = material;
      sphere.renderingGroupId = GROUP_ASSET;
      sphere.isPickable = false;
      this.chromeSphere = sphere;
    }
    const sphere = this.chromeSphere;
    const next = !sphere.isEnabled();
    if (next) {
      const dir = camera.getForwardRay(1).direction.clone();
      dir.normalize();
      sphere.position.copyFrom(camera.position.add(dir.scale(CHROME_SPHERE_DISTANCE_M)));
      console.info(
        this.envTexture
          ? "[ibl] chrome sphere ON (IBL environment active)"
          : "[ibl] chrome sphere ON -- NO environment loaded, stand-in lights only",
      );
    }
    sphere.setEnabled(next);
    return next;
  }
}

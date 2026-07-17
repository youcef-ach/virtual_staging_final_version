import "@babylonjs/loaders/glTF";
import { ImportMeshAsync } from "@babylonjs/core/Loading/sceneLoader";
import { Ray } from "@babylonjs/core/Culling/ray"; // side effect: enables scene.pick*
import type { Scene } from "@babylonjs/core/scene";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { DirectionalLight } from "@babylonjs/core/Lights/directionalLight";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";

/**
 * Rendering groups implement the three-layer contract (PLAN section 1):
 * skybox (Layer 1, photographic truth) -> proxy depth (Layer 2, metric truth)
 * -> staged assets (Layer 3). The depth buffer written by the proxy group is
 * PRESERVED into the asset group, so real geometry occludes virtual furniture
 * while Layer 1 pixels stay untouched (the proxy never writes color).
 */
export const GROUP_SKYBOX = 0;
export const GROUP_PROXY = 1;
export const GROUP_ASSET = 2;

/** Matterport tripods stand ~1.4 m above the floor they rest on (F2/G2 evidence). */
const TRIPOD_HEIGHT_M = 1.4;
const INITIAL_PLACE_DISTANCE_M = 1.8;

const isFloor = (mesh: AbstractMesh): boolean =>
  mesh.isPickable && mesh.name.includes("floor");

/**
 * G7 staging layer: invisible depth-only proxy occluders + the first
 * furniture piece with placement tooling.
 *
 * Slice 7.1 uses TEMPORARY stand-in lighting (hemi + sun); slice 7.2
 * replaces it with the per-sweep IBL products (G5 irradiance + GGX chain).
 */
export class StagingLayer {
  private readonly scene: Scene;
  private readonly occluder: StandardMaterial;
  private assetRoot: TransformNode | null = null;
  private userPlaced = false;

  constructor(scene: Scene) {
    this.scene = scene;
    // Layer 3 must depth-test against the proxy depth written in GROUP_PROXY:
    // stop Babylon from clearing the depth buffer between the two groups.
    scene.setRenderingAutoClearDepthStencil(GROUP_ASSET, false);

    // Invisible depth-only occluder: writes depth, never color. Wherever the
    // proxy is closer than the asset, the photographic Layer 1 pixel survives.
    const occluder = new StandardMaterial("proxy-occluder", scene);
    occluder.disableColorWrite = true;
    occluder.disableLighting = true;
    occluder.backFaceCulling = false; // room shells have inward-facing normals
    this.occluder = occluder;

    // TEMPORARY stand-in lighting -- slice 7.1 only, replaced by IBL in 7.2.
    const hemi = new HemisphericLight("temp-hemi-7_1", new Vector3(0, 1, 0), scene);
    hemi.intensity = 0.9;
    const sun = new DirectionalLight("temp-sun-7_1", new Vector3(-0.4, -1, -0.3), scene);
    sun.intensity = 0.6;
  }

  get hasAsset(): boolean {
    return this.assetRoot !== null;
  }

  /**
   * Load proxy GLBs (shell + details) as invisible depth-only occluders.
   * Missing files are reported, not fatal (the tour must keep working).
   * Floors stay pickable as placement targets; walls/ceilings/details only
   * occlude.
   */
  async loadProxy(urls: string[]): Promise<{ count: number; failed: string[] }> {
    let count = 0;
    const failed: string[] = [];
    for (const url of urls) {
      try {
        const result = await ImportMeshAsync(url, this.scene);
        for (const mesh of result.meshes) {
          mesh.renderingGroupId = GROUP_PROXY;
          if (mesh.getTotalVertices() > 0) {
            mesh.material = this.occluder;
            mesh.isPickable = mesh.name.includes("floor");
            mesh.freezeWorldMatrix();
            count += 1;
          } else {
            mesh.isPickable = false;
          }
        }
      } catch {
        failed.push(url);
      }
    }
    return { count, failed };
  }

  /** Load the Layer 3 test asset (real-scale GLB, meters; P1 normalizes later). */
  async loadAsset(url: string): Promise<void> {
    const result = await ImportMeshAsync(url, this.scene);
    const root = new TransformNode("asset-root", this.scene);
    for (const mesh of result.meshes) {
      if (!mesh.parent) {
        mesh.parent = root;
      }
      mesh.renderingGroupId = GROUP_ASSET;
      mesh.isPickable = false; // placement raycasts must reach the floor
    }
    this.assetRoot = root;
  }

  /** Double-click placement: pick the proxy FLOOR under canvas coords (x, y). */
  placeFromPick(x: number, y: number): boolean {
    if (!this.assetRoot) return false;
    const hit = this.scene.pick(x, y, isFloor);
    if (!hit?.hit || !hit.pickedPoint) return false;
    this.assetRoot.position.copyFrom(hit.pickedPoint);
    this.userPlaced = true;
    return true;
  }

  /**
   * Initial placement: on the floor ~1.8 m in front of the camera (downward
   * ray onto the proxy floor; tripod-height fallback when no proxy loaded).
   * Skipped once the operator has placed the asset manually.
   */
  placeInFrontOf(camera: FreeCamera): void {
    if (!this.assetRoot || this.userPlaced) return;
    const dir = camera.getForwardRay(1).direction.clone();
    dir.y = 0;
    if (dir.lengthSquared() < 1e-6) {
      dir.set(0, 0, 1);
    }
    dir.normalize();
    const target = camera.position.add(dir.scale(INITIAL_PLACE_DISTANCE_M));
    const ray = new Ray(
      new Vector3(target.x, camera.position.y + 0.5, target.z),
      Vector3.Down(),
      6,
    );
    const hit = this.scene.pickWithRay(ray, isFloor);
    const y =
      hit?.hit && hit.pickedPoint
        ? hit.pickedPoint.y
        : camera.position.y - TRIPOD_HEIGHT_M;
    this.assetRoot.position.set(target.x, y, target.z);
  }

  /** Keyboard nudge on the ground plane (world X/Z, meters). */
  nudge(dx: number, dz: number): void {
    if (!this.assetRoot) return;
    this.assetRoot.position.x += dx;
    this.assetRoot.position.z += dz;
    this.userPlaced = true;
  }

  /** Vertical adjust (meters) -- diagnosis aid, T6 locks Y to the floor. */
  lift(dy: number): void {
    if (!this.assetRoot) return;
    this.assetRoot.position.y += dy;
    this.userPlaced = true;
  }

  /** Rotate around Y (radians). */
  rotate(rad: number): void {
    if (!this.assetRoot) return;
    this.assetRoot.rotation.y += rad;
  }
}

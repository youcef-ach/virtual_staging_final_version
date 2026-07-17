import "@babylonjs/loaders/glTF";
import { ImportMeshAsync } from "@babylonjs/core/Loading/sceneLoader";
import { Ray } from "@babylonjs/core/Culling/ray"; // side effect: enables scene.pick*
import type { Scene } from "@babylonjs/core/scene";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Color3 } from "@babylonjs/core/Maths/math.color";
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
  private readonly xrayShell: StandardMaterial;
  private readonly xrayDetail: StandardMaterial;
  private readonly shellMeshes: AbstractMesh[] = [];
  private readonly detailMeshes: AbstractMesh[] = [];
  private xray = false;
  private detailsEnabled = true;
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

    // G7.1 X-ray debug materials: judge proxy<->photo alignment directly over
    // the photography (LESSONS L5). Forced depth write keeps occlusion live
    // while X-ray is on (alpha-blended meshes skip depth writes by default).
    // Shells (walls/floors/ceilings) are GREEN; detail proxies (OBJ-clipped
    // stairs/counters/furniture fragments) are ORANGE so their intentionally
    // ragged silhouettes are not mistaken for misplaced walls.
    const xrayShell = new StandardMaterial("proxy-xray-shell", scene);
    xrayShell.emissiveColor = new Color3(0.15, 0.9, 0.4);
    xrayShell.disableLighting = true;
    xrayShell.alpha = 0.3;
    xrayShell.forceDepthWrite = true;
    xrayShell.backFaceCulling = false;
    this.xrayShell = xrayShell;
    const xrayDetail = new StandardMaterial("proxy-xray-detail", scene);
    xrayDetail.emissiveColor = new Color3(0.95, 0.55, 0.15);
    xrayDetail.disableLighting = true;
    xrayDetail.alpha = 0.3;
    xrayDetail.forceDepthWrite = true;
    xrayDetail.backFaceCulling = false;
    this.xrayDetail = xrayDetail;

    // TEMPORARY stand-in lighting -- slice 7.1 only, replaced by IBL in 7.2.
    const hemi = new HemisphericLight("temp-hemi-7_1", new Vector3(0, 1, 0), scene);
    hemi.intensity = 0.9;
    const sun = new DirectionalLight("temp-sun-7_1", new Vector3(-0.4, -1, -0.3), scene);
    sun.intensity = 0.6;
  }

  get hasAsset(): boolean {
    return this.assetRoot !== null;
  }

  /** Toggle the X-ray debug view on the proxy (X key). Returns the new state. */
  toggleXray(): boolean {
    this.xray = !this.xray;
    for (const mesh of this.shellMeshes) {
      mesh.material = this.xray ? this.xrayShell : this.occluder;
    }
    for (const mesh of this.detailMeshes) {
      mesh.material = this.xray ? this.xrayDetail : this.occluder;
    }
    return this.xray;
  }

  /**
   * Toggle detail proxies (D key): hides them visually AND as occluders --
   * isolates shell-vs-detail occlusion faults (e.g. OPEN_PROBLEMS #14b,
   * a detail box enclosing a sweep position). Returns the new state.
   */
  toggleDetails(): boolean {
    this.detailsEnabled = !this.detailsEnabled;
    for (const mesh of this.detailMeshes) {
      mesh.setEnabled(this.detailsEnabled);
    }
    console.info(`[staging] detail proxies ${this.detailsEnabled ? "ON" : "OFF"}`);
    return this.detailsEnabled;
  }

  /**
   * Load proxy GLBs (shell + details) as invisible depth-only occluders.
   * Missing files are reported, not fatal (the tour must keep working).
   * Floors stay pickable as placement targets; walls/ceilings/details only
   * occlude.
   */
  async loadProxy(
    urls: string[],
  ): Promise<{ count: number; floors: number; failed: string[] }> {
    let count = 0;
    let floors = 0;
    const failed: string[] = [];
    for (const url of urls) {
      try {
        const result = await ImportMeshAsync(url, this.scene);
        // L1 diagnostic: settle what Babylon's RH glTF import does to __root__
        // (identity expected in a right-handed scene; anything else means the
        // PROXY is the transformed party, not the pano).
        const rootNode = result.meshes.find((m) => m.name === "__root__");
        if (rootNode) {
          console.info(
            `[staging] ${url} __root__ scaling=${rootNode.scaling.toString()} ` +
              `rotationQuaternion=${rootNode.rotationQuaternion?.toString() ?? "null"} ` +
              `position=${rootNode.position.toString()}`,
          );
        }
        const isDetails = url.includes("details");
        for (const mesh of result.meshes) {
          mesh.renderingGroupId = GROUP_PROXY;
          if (mesh.getTotalVertices() > 0) {
            const xrayMat = isDetails ? this.xrayDetail : this.xrayShell;
            mesh.material = this.xray ? xrayMat : this.occluder;
            mesh.isPickable = !isDetails && mesh.name.includes("floor");
            if (mesh.isPickable) {
              floors += 1;
            }
            mesh.freezeWorldMatrix();
            (isDetails ? this.detailMeshes : this.shellMeshes).push(mesh);
            count += 1;
          } else {
            mesh.isPickable = false;
          }
        }
      } catch {
        failed.push(url);
      }
    }
    return { count, floors, failed };
  }

  /**
   * Load the Layer 3 test asset (real-scale GLB, meters).
   * Bottom-center pivot normalization (the P1 catalog convention) happens at
   * runtime for test assets: ad-hoc GLBs usually have a CENTERED origin,
   * which sinks them half their height into the floor when the root lands
   * at pickedPoint. Shift the hierarchy so the bounding-box bottom sits
   * exactly at the root origin.
   */
  async loadAsset(url: string): Promise<void> {
    const result = await ImportMeshAsync(url, this.scene);
    const root = new TransformNode("asset-root", this.scene);
    const pivot = new TransformNode("asset-pivot", this.scene);
    pivot.parent = root;
    for (const mesh of result.meshes) {
      if (!mesh.parent) {
        mesh.parent = pivot;
      }
      mesh.renderingGroupId = GROUP_ASSET;
      mesh.isPickable = false; // placement raycasts must reach the floor
    }
    let minY = Number.POSITIVE_INFINITY;
    for (const mesh of result.meshes) {
      if (mesh.getTotalVertices() > 0) {
        mesh.computeWorldMatrix(true);
        minY = Math.min(minY, mesh.getBoundingInfo().boundingBox.minimumWorld.y);
      }
    }
    if (Number.isFinite(minY) && minY !== 0) {
      pivot.position.y = -minY;
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

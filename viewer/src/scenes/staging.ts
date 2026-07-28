import "@babylonjs/loaders/glTF";
import { ImportMeshAsync } from "@babylonjs/core/Loading/sceneLoader";
import { Ray } from "@babylonjs/core/Culling/ray"; // side effect: enables scene.pick*
import type { Scene } from "@babylonjs/core/scene";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import type { Material } from "@babylonjs/core/Materials/material";
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

/** X-ray debug view scope (X key cycles off -> room -> local -> full). */
export type XrayMode = "off" | "room" | "local" | "full";
/** LOCAL X-ray: only meshes whose bounding box comes within this range show. */
const LOCAL_XRAY_RADIUS_M = 5;

/** Proxy sub-meshes are named "<storey>_room_<NN>/<floor|ceiling|wall_NN>". */
const roomKeyOf = (name: string): string => {
  const idx = name.indexOf("/");
  return idx > 0 ? name.slice(0, idx) : name;
};

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
  private readonly seethruMeshes: AbstractMesh[] = [];
  private readonly seethruIdle: StandardMaterial;
  private readonly xraySeethru: StandardMaterial;
  private xrayMode: XrayMode = "off";
  private detailsEnabled = true;
  private highlightedMesh: AbstractMesh | null = null;
  private highlightedPrev: Material | null = null;
  private readonly xrayPicked: StandardMaterial;
  private assetRoot: TransformNode | null = null;
  private userPlaced = false;

  /** T7: get asset position (or null if no asset loaded). */
  getAssetPosition(): { x: number; y: number; z: number } | null {
    if (!this.assetRoot) return null;
    const p = this.assetRoot.position;
    return { x: p.x, y: p.y, z: p.z };
  }

  /** T7: get asset Y rotation in radians (or null). */
  getAssetRotationY(): number | null {
    return this.assetRoot?.rotation.y ?? null;
  }

  /** T7: restore asset state from a saved project. */
  restoreAssetState(position: [number, number, number], rotationY: number): void {
    if (!this.assetRoot) return;
    this.assetRoot.position.set(position[0], position[1], position[2]);
    this.assetRoot.rotation.y = rotationY;
    this.userPlaced = true;
  }

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
    // Identify highlight (Ctrl+click): the picked proxy mesh turns MAGENTA so
    // "that rogue wall" becomes a concrete sub-mesh name in console + HUD.
    const xrayPicked = new StandardMaterial("proxy-xray-picked", scene);
    xrayPicked.emissiveColor = new Color3(0.95, 0.2, 0.85);
    xrayPicked.disableLighting = true;
    xrayPicked.alpha = 0.5;
    xrayPicked.forceDepthWrite = true;
    xrayPicked.backFaceCulling = false;
    this.xrayPicked = xrayPicked;
    // See-through walls (#23): glass closure walls / open-passage dividers
    // exist for ROOM FORMATION only -- at runtime they must not occlude
    // (the pano sees through glass). Inert in normal mode (no color, no
    // depth write; still Ctrl+click identifiable); CYAN in X-ray, never
    // depth-writing so they cannot hide the asset even while debugging.
    const seethruIdle = new StandardMaterial("proxy-seethru-idle", scene);
    seethruIdle.disableColorWrite = true;
    seethruIdle.disableDepthWrite = true;
    seethruIdle.disableLighting = true;
    this.seethruIdle = seethruIdle;
    const xraySeethru = new StandardMaterial("proxy-xray-seethru", scene);
    xraySeethru.emissiveColor = new Color3(0.2, 0.85, 0.95);
    xraySeethru.disableLighting = true;
    xraySeethru.alpha = 0.3;
    xraySeethru.backFaceCulling = false;
    this.xraySeethru = xraySeethru;

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
   * Cycle the X-ray debug view (X key): off -> room -> local -> full -> off.
   * ROOM shows ONLY the shell of the room under the camera (by sub-mesh
   * name) plus nearby detail proxies -- the G3 defect-triage view: one
   * room's planes, zero soup. LOCAL shows all geometry within ~5 m.
   * Non-scoped meshes keep occluding invisibly, so runtime behavior stays
   * honest while the view stays readable.
   */
  cycleXray(cameraPos: Vector3): XrayMode {
    const order: XrayMode[] = ["off", "room", "local", "full"];
    this.xrayMode = order[(order.indexOf(this.xrayMode) + 1) % order.length];
    this.applyXray(cameraPos);
    return this.xrayMode;
  }

  /** Re-apply the LOCAL X-ray scope after the camera moved (sweep jump). */
  refreshXray(cameraPos: Vector3): void {
    if (this.xrayMode !== "off") {
      this.applyXray(cameraPos);
    }
  }

  private applyXray(cameraPos: Vector3): void {
    this.clearHighlight();
    const roomKey = this.xrayMode === "room" ? this.currentRoomKey(cameraPos) : null;
    if (this.xrayMode === "room") {
      console.info(
        `[staging] ROOM x-ray scope: ${roomKey ?? "no floor resolved -- radius fallback"}`,
      );
    }
    const near = (mesh: AbstractMesh): boolean => {
      const bb = mesh.getBoundingInfo().boundingBox;
      const dx = Math.max(bb.minimumWorld.x - cameraPos.x, 0, cameraPos.x - bb.maximumWorld.x);
      const dy = Math.max(bb.minimumWorld.y - cameraPos.y, 0, cameraPos.y - bb.maximumWorld.y);
      const dz = Math.max(bb.minimumWorld.z - cameraPos.z, 0, cameraPos.z - bb.maximumWorld.z);
      return dx * dx + dy * dy + dz * dz <= LOCAL_XRAY_RADIUS_M * LOCAL_XRAY_RADIUS_M;
    };
    const inScope = (mesh: AbstractMesh, isShell: boolean): boolean => {
      if (this.xrayMode === "full") return true;
      if (this.xrayMode === "room" && isShell && roomKey !== null) {
        return roomKeyOf(mesh.name) === roomKey;
      }
      return near(mesh);
    };
    for (const mesh of this.shellMeshes) {
      mesh.material =
        this.xrayMode !== "off" && inScope(mesh, true) ? this.xrayShell : this.occluder;
    }
    for (const mesh of this.detailMeshes) {
      mesh.material =
        this.xrayMode !== "off" && inScope(mesh, false) ? this.xrayDetail : this.occluder;
    }
    for (const mesh of this.seethruMeshes) {
      mesh.material =
        this.xrayMode !== "off" && inScope(mesh, true)
          ? this.xraySeethru
          : this.seethruIdle;
    }
  }

  /** Resolve the room key under the camera: floor below, else nearest floor. */
  private currentRoomKey(cameraPos: Vector3): string | null {
    const ray = new Ray(cameraPos, Vector3.Down(), 8);
    const hit = this.scene.pickWithRay(ray, isFloor);
    if (hit?.hit && hit.pickedMesh) {
      return roomKeyOf(hit.pickedMesh.name);
    }
    // No floor directly below (stair void -- or the floor itself is MISSING,
    // one of the defects under triage): nearest floor bbox wins; the console
    // scope line above tells the operator which room got resolved.
    let bestKey: string | null = null;
    let bestD2 = Number.POSITIVE_INFINITY;
    for (const mesh of this.shellMeshes) {
      if (!mesh.name.includes("floor")) continue;
      const bb = mesh.getBoundingInfo().boundingBox;
      const dx = Math.max(bb.minimumWorld.x - cameraPos.x, 0, cameraPos.x - bb.maximumWorld.x);
      const dy = Math.max(bb.minimumWorld.y - cameraPos.y, 0, cameraPos.y - bb.maximumWorld.y);
      const dz = Math.max(bb.minimumWorld.z - cameraPos.z, 0, cameraPos.z - bb.maximumWorld.z);
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestKey = roomKeyOf(mesh.name);
      }
    }
    return bestKey;
  }

  /**
   * Identify the proxy mesh under canvas coords (Ctrl+click): logs the exact
   * sub-mesh name + room key + hit point + bbox, highlights it MAGENTA and
   * returns a short HUD label. Works in ANY X-ray mode including off -- you
   * can identify the invisible occluder that wrongly hides the asset, and
   * clicking where a floor SHOULD be proves its absence.
   */
  identify(x: number, y: number): string | null {
    const hit = this.scene.pick(
      x,
      y,
      (mesh) =>
        mesh.isEnabled() &&
        (this.shellMeshes.includes(mesh) ||
          this.detailMeshes.includes(mesh) ||
          this.seethruMeshes.includes(mesh)),
    );
    if (!hit?.hit || !hit.pickedMesh || !hit.pickedPoint) {
      console.info(
        "[identify] no proxy mesh under cursor (proxy ABSENT here -- absence is also a finding)",
      );
      return "identify: no proxy here";
    }
    const mesh = hit.pickedMesh;
    const kind = this.detailMeshes.includes(mesh)
      ? "detail"
      : this.seethruMeshes.includes(mesh)
        ? "seethru"
        : "shell";
    const bb = mesh.getBoundingInfo().boundingBox;
    const size = bb.maximumWorld.subtract(bb.minimumWorld);
    const p = hit.pickedPoint;
    console.info(
      `[identify] ${mesh.name} [${kind}] room=${roomKeyOf(mesh.name)} ` +
        `hit=(${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}) ` +
        `dist=${hit.distance.toFixed(2)} m ` +
        `bbox=${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)} m`,
    );
    this.clearHighlight();
    this.highlightedPrev = mesh.material;
    this.highlightedMesh = mesh;
    mesh.material = this.xrayPicked;
    return `${mesh.name} [${kind}]`;
  }

  private clearHighlight(): void {
    if (this.highlightedMesh) {
      this.highlightedMesh.material = this.highlightedPrev;
      this.highlightedMesh = null;
      this.highlightedPrev = null;
    }
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

  private refRoot: TransformNode | null = null;
  private refZup = false;
  private readonly refMeshes: AbstractMesh[] = [];
  private readonly refMats: (Material | null)[] = [];
  private refOccluding = false;

  /** T2: expose reference meshes for the projective transition shader. */
  getReferenceMeshes(): readonly AbstractMesh[] {
    return this.refMeshes;
  }

  /**
   * G7.1 ground-truth reference: the textured MatterPak mesh (OBJ exported
   * as GLB). Its textures DERIVE from the panos, so with a correct pano
   * orientation the photo and the mesh must match nearly feature-for-feature
   * -- the strongest in-viewer alignment evidence (F5 ghosting philosophy).
   * Hidden until toggled (B); N applies rotX-90 if the GLB kept the
   * MatterPak Z-up frame.
   */
  async loadReference(url: string): Promise<boolean> {
    try {
      const result = await ImportMeshAsync(url, this.scene);
      const root = new TransformNode("reference-root", this.scene);
      for (const mesh of result.meshes) {
        if (!mesh.parent) {
          mesh.parent = root;
        }
        mesh.renderingGroupId = GROUP_PROXY;
        mesh.isPickable = false;
        if (mesh.getTotalVertices() > 0) {
          this.refMeshes.push(mesh);
          this.refMats.push(mesh.material);
        }
      }
      // The MatterPak GLB is Z-up; rotate -90° around X to align with
      // Babylon's Y-up scene (empirically confirmed: rotX=270° = -PI/2).
      root.rotation.x = -Math.PI / 2;
      this.refZup = true;
      root.setEnabled(false); // B shows it
      this.refRoot = root;
      console.info(
        `[staging] reference mesh loaded (${result.meshes.length} meshes, Z-up→Y-up applied)`,
      );
      return true;
    } catch {
      return false;
    }
  }

  /** Toggle the textured reference mesh (B key). Returns the new state. */
  toggleReference(): boolean {
    if (!this.refRoot) {
      console.info("[staging] no reference.glb loaded (copy it into viewer/public/data/)");
      return false;
    }
    const next = !this.refRoot.isEnabled();
    this.refRoot.setEnabled(next);
    return next;
  }

  /**
   * A/B occlusion experiment (G key): the textured MatterPak mesh becomes
   * the INVISIBLE depth-only occluder while shell walls/ceilings, details
   * and seethru meshes stand down (floors stay enabled: placement raycasts
   * + floor occlusion). The OBJ carries the real captured geometry --
   * worktops, open passages, stairs -- so this tests whether Decision 6
   * should be amended to a hybrid occluder. Exiting restores everything.
   */
  toggleReferenceOccluder(): boolean {
    if (!this.refRoot) {
      console.info("[staging] no reference.glb loaded -- cannot A/B occluders");
      return false;
    }
    this.refOccluding = !this.refOccluding;
    if (this.refOccluding) {
      this.refRoot.setEnabled(true);
      for (const mesh of this.refMeshes) {
        mesh.material = this.occluder;
      }
      for (const mesh of [...this.shellMeshes, ...this.detailMeshes]) {
        if (!mesh.name.includes("floor")) {
          mesh.setEnabled(false);
        }
      }
      for (const mesh of this.seethruMeshes) {
        mesh.setEnabled(false);
      }
      console.info(
        "[staging] REFERENCE-OCCLUDER mode ON: MatterPak mesh depth-writes " +
          "(invisible); shell walls/ceilings + details + seethru OFF; floors kept",
      );
    } else {
      this.refRoot.setEnabled(false);
      for (let i = 0; i < this.refMeshes.length; i += 1) {
        this.refMeshes[i].material = this.refMats[i];
      }
      for (const mesh of this.shellMeshes) {
        mesh.setEnabled(true);
      }
      for (const mesh of this.detailMeshes) {
        mesh.setEnabled(this.detailsEnabled);
      }
      for (const mesh of this.seethruMeshes) {
        mesh.setEnabled(true);
      }
      console.info("[staging] REFERENCE-OCCLUDER mode OFF: shells + details restored");
    }
    return this.refOccluding;
  }

  /** Toggle the Z-up -> Y-up conversion on the reference root (N key). */
  toggleReferenceFrame(): boolean {
    if (!this.refRoot) return false;
    this.refZup = !this.refZup;
    this.refRoot.rotation.x = this.refZup ? -Math.PI / 2 : 0;
    console.info(
      `[staging] reference frame: ${this.refZup ? "Z-up source (rotX-90 applied)" : "as-authored (Y-up)"}`,
    );
    return this.refZup;
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
            const isSeethru = !isDetails && mesh.name.includes("seethru");
            mesh.material = isSeethru ? this.seethruIdle : this.occluder;
            mesh.isPickable = !isDetails && mesh.name.includes("floor");
            if (mesh.isPickable) {
              floors += 1;
            }
            mesh.freezeWorldMatrix();
            (isSeethru
              ? this.seethruMeshes
              : isDetails
                ? this.detailMeshes
                : this.shellMeshes
            ).push(mesh);
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
  async loadAsset(
    url: string,
    options?: { isPreNormalized?: boolean; lod?: "lod0" | "lod1" | "lod2" },
  ): Promise<void> {
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

    if (!options?.isPreNormalized) {
      let minY = Number.POSITIVE_INFINITY;
      for (const mesh of result.meshes) {
        if (mesh.getTotalVertices() > 0) {
          mesh.computeWorldMatrix(true);
          minY = Math.min(minY, mesh.getBoundingInfo().boundingBox.minimumWorld.y);
        }
      }
      if (Number.isFinite(minY) && Math.abs(minY) > 0.001) {
        pivot.position.y = -minY;
      }
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
    // Grounding check: proxy floor Y is the G2-elected floor plane (LiDAR
    // truth); tripods stand ~1.4 m above the floor they rest on, so
    // (cameraY - 1.4) - floorY should be ~0. |delta| > ~0.05 m flags a
    // suspect floor election (#10 thin-coverage rooms).
    const camY = this.scene.activeCamera?.position.y;
    const delta =
      camY !== undefined ? camY - TRIPOD_HEIGHT_M - hit.pickedPoint.y : null;
    console.info(
      `[place] floor=${hit.pickedMesh?.name ?? "?"} y=${hit.pickedPoint.y.toFixed(3)}` +
        (delta !== null
          ? ` | grounding delta (cameraY-1.4)-floorY = ${delta.toFixed(3)} m (expect ~0 +/- 0.05)`
          : ""),
    );
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
    const usedFloor = !!(hit?.hit && hit.pickedPoint);
    const y = usedFloor
      ? hit!.pickedPoint!.y
      : camera.position.y - TRIPOD_HEIGHT_M;
    console.info(
      `[place] initial ${
        usedFloor
          ? `floor=${hit!.pickedMesh?.name ?? "?"}`
          : "NO FLOOR HIT -- tripod-height fallback (missing floor mesh here?)"
      } y=${y.toFixed(3)}`,
    );
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

  // ---------------------------------------------------------------------------
  // T6: drag-drop staging editor
  // ---------------------------------------------------------------------------

  private _isDragging = false;

  /** True while the asset is being dragged. */
  get isDragging(): boolean {
    return this._isDragging;
  }

  /** Start drag: record that we're in drag mode. */
  dragStart(): void {
    if (!this.assetRoot) return;
    this._isDragging = true;
  }

  /**
   * Drag move: raycast to floor under cursor and move asset there.
   * Returns true if the asset was repositioned.
   */
  dragMove(x: number, y: number): boolean {
    if (!this._isDragging || !this.assetRoot) return false;
    const hit = this.scene.pick(x, y, isFloor);
    if (!hit?.hit || !hit.pickedPoint) return false;
    this.assetRoot.position.x = hit.pickedPoint.x;
    this.assetRoot.position.z = hit.pickedPoint.z;
    // Keep Y locked to the floor surface.
    this.assetRoot.position.y = hit.pickedPoint.y;
    this.userPlaced = true;
    return true;
  }

  /** End drag. Optionally auto-snap to nearest wall. */
  dragEnd(autoSnap: boolean = true): void {
    this._isDragging = false;
    if (autoSnap) {
      this.snapToNearestWall();
    }
  }

  /**
   * T6 wall snap: if the asset is within SNAP_DISTANCE of a wall mesh,
   * push it flush against the wall and rotate to face along the wall normal.
   */
  private static readonly SNAP_DISTANCE_M = 0.4;

  snapToNearestWall(): boolean {
    if (!this.assetRoot) return false;
    const pos = this.assetRoot.position;

    // Cast rays in 8 compass directions from the asset position to find walls.
    const directions = [
      new Vector3(1, 0, 0),
      new Vector3(-1, 0, 0),
      new Vector3(0, 0, 1),
      new Vector3(0, 0, -1),
      new Vector3(0.707, 0, 0.707),
      new Vector3(-0.707, 0, 0.707),
      new Vector3(0.707, 0, -0.707),
      new Vector3(-0.707, 0, -0.707),
    ];

    const isWall = (m: AbstractMesh): boolean =>
      m.isPickable && m.name.includes("wall") && m.renderingGroupId === GROUP_PROXY;

    let bestDist = StagingLayer.SNAP_DISTANCE_M;
    let bestNormal: Vector3 | null = null;
    let bestPoint: Vector3 | null = null;

    for (const dir of directions) {
      const ray = new Ray(pos, dir, StagingLayer.SNAP_DISTANCE_M);
      const hit = this.scene.pickWithRay(ray, isWall);
      if (hit?.hit && hit.pickedPoint && hit.distance < bestDist) {
        bestDist = hit.distance;
        bestNormal = hit.getNormal(true) ?? null;
        bestPoint = hit.pickedPoint.clone();
      }
    }

    if (!bestNormal || !bestPoint) return false;

    // Push asset flush against the wall (offset by a small margin).
    const margin = 0.02;
    const pushDir = bestNormal.scale(margin);
    this.assetRoot.position.x = bestPoint.x + pushDir.x;
    this.assetRoot.position.z = bestPoint.z + pushDir.z;
    // Keep Y locked.

    // Rotate asset to face outward from the wall (normal direction).
    const angle = Math.atan2(bestNormal.x, bestNormal.z);
    this.assetRoot.rotation.y = angle;

    console.info(
      `[staging] wall snap: dist=${bestDist.toFixed(3)} m, angle=${((angle * 180) / Math.PI).toFixed(1)}°`,
    );
    return true;
  }
}

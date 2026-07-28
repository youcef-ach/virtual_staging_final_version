import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import { Scene } from "@babylonjs/core/scene";
import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { Matrix, Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { CreateBox } from "@babylonjs/core/Meshes/Builders/boxBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import type { SweepTransition } from "./transition";
import type { TextureCache } from "./textureCache";
import type { SweepData, TourData } from "../data/types";

/** Babylon CubeTexture file order: +x, +y, +z, -x, -y, -z. */
const FACE_ORDER = ["px", "py", "pz", "nx", "ny", "nz"] as const;

/** Drag sensitivity (radians per pixel) and pitch clamp for the pano camera. */
const DRAG_SENSITIVITY = 0.0025;
const PITCH_LIMIT = Math.PI / 2 - 0.02;

/**
 * Pano orientation chain -- LOCKED 2026-07-17 (G7.1), harness stripped
 * 2026-07-26 after G7.1 closed (see LESSONS L13/L14 and the PLAN changelog):
 *
 * The extracted canonical faces live in each sweep's LOCAL tripod frame.
 * poses.json's pose_rt rotation maps local -> world; the viewer rotates every
 * sampled direction world -> local PER SWEEP before the canonical-layout
 * conversion. Chain (Babylon row-vector convention, v' = v * M,
 * A.multiply(B) = A then B):
 *
 *   sampleDir = worldDir * R_pose (verbatim fill = R^T, world->local)
 *                        * SKYBOX_BASE (local -> cube layout, F4.4)
 *
 * Residual yaw = 0 (validated against the textured MatterPak reference,
 * the detail proxies and the photography simultaneously; rotation source =
 * E57 SCAN headers, legacy project confirmed identical).
 *
 * The SAME matrix orients the per-sweep IBL environment (G7.2): the .env
 * cubemap is built from the same stored canonical faces, and Babylon's PBR
 * shader applies the reflection matrix to both specular sampling and the
 * SH-polynomial irradiance normal.
 */
const SKYBOX_BASE: Matrix = Matrix.Scaling(-1, 1, 1).multiply(
  Matrix.RotationX(-Math.PI / 2),
);

/**
 * Layer 1 renderer: the photographic layer. The active sweep's cubemap is an
 * unlit skybox, rendered as-is, never modified (visual truth).
 *
 * Camera: inside-out look-around (FreeCamera + deterministic drag handler).
 * All orientation goes through applyLook() -- one code path for the initial
 * view, sweep jumps and drags (avoids RH TargetCamera path discrepancies).
 * Yaw/pitch persist across sweep jumps for continuity.
 *
 * VRAM discipline: only the active sweep's cubemap stays resident.
 */
export class TourViewer {
  readonly scene: Scene;
  onSweepShown?: (uuid: string) => void;

  private readonly camera: FreeCamera;
  private readonly skyboxMaterial: StandardMaterial;
  private activeTexture: CubeTexture | null = null;
  private activeUuid: string | null = null;
  private yaw = 0;
  private pitch = 0;
  private currentPoseRotation: Matrix = Matrix.Identity();
  private detachControls: (() => void) | null = null;
  /** T2: transition controller (set after reference mesh loads). */
  private transition: SweepTransition | null = null;
  /** T2: cubemap cache for neighbor prefetch. */
  private textureCache: TextureCache | null = null;

  constructor(
    engine: AbstractEngine,
    private readonly tour: TourData,
  ) {
    this.scene = new Scene(engine);
    // The runtime frame (Coordinate Lock Rule) is right-handed Y-up -- match it.
    this.scene.useRightHandedSystem = true;
    this.scene.clearColor = new Color4(0, 0, 0, 1);

    this.camera = new FreeCamera("camera", Vector3.Zero(), this.scene);
    this.camera.minZ = 0.01;
    this.camera.fov = 1.15;
    this.camera.inputs.clear(); // deterministic pano controls below
    this.attachPanoControls(engine.getRenderingCanvas());
    this.applyLook();

    const skybox = CreateBox("skybox", { size: 512 }, this.scene);
    skybox.infiniteDistance = true; // always centered on the camera
    const material = new StandardMaterial("skybox-material", this.scene);
    material.backFaceCulling = false;
    material.disableLighting = true;
    material.diffuseColor = Color3.Black();
    material.specularColor = Color3.Black();
    skybox.material = material;
    this.skyboxMaterial = material;
  }

  get activeSweepUuid(): string | null {
    return this.activeUuid;
  }

  /** The pano camera -- staging (Layer 2/3) needs it for placement helpers. */
  getCamera(): FreeCamera {
    return this.camera;
  }

  /**
   * The locked per-sweep orientation matrix (pose world->local x SKYBOX_BASE).
   * The IBL environment texture (G7.2) applies this SAME matrix so lighting
   * directions match the photograph exactly like the skybox does.
   */
  getPanoOrientation(): Matrix {
    return this.currentPoseRotation.multiply(SKYBOX_BASE);
  }

  /**
   * T2 alignment fix: the pure pose rotation (R^T, world→scanner-local)
   * WITHOUT Babylon's SKYBOX_BASE. The projective shader samples via
   * textureCube() directly — it needs the scanner-local direction, not the
   * Babylon skybox reflection convention.
   */
  getPoseRotation(): Matrix {
    return this.currentPoseRotation.clone();
  }

  /** T2: set the transition controller (call after reference mesh is loaded). */
  setTransition(transition: SweepTransition): void {
    this.transition = transition;
  }

  /** T2: set the texture cache for prefetch. */
  setTextureCache(cache: TextureCache): void {
    this.textureCache = cache;
  }

  /** T2: true while a projective transition is animating. */
  get isTransitioning(): boolean {
    return this.transition?.isTransitioning ?? false;
  }

  /** Toggle the pano skybox (P key): proxy-only debug view. Returns new state. */
  toggleSkybox(): boolean {
    const skybox = this.scene.getMeshByName("skybox");
    if (!skybox) return true;
    const next = !skybox.isEnabled();
    skybox.setEnabled(next);
    return next;
  }

  /**
   * Load a sweep's cubemap and move the camera to its pose position.
   * T2: if a transition controller is set, animates via projective morph;
   * otherwise instant-jumps (the G7 behavior).
   */
  async showSweep(uuid: string): Promise<void> {
    if (this.transition?.isTransitioning) return; // block during animation

    const sweep = this.tour.sweeps.find((s) => s.uuid === uuid);
    if (!sweep) {
      throw new Error(`unknown sweep: ${uuid}`);
    }

    const files = FACE_ORDER.map((face) => `${this.tour.baseUrl}/${sweep.faces[face]}`);

    // T2: use cache if available, otherwise load directly.
    const texture = this.textureCache
      ? await this.textureCache.get(sweep.uuid, files)
      : await this.loadCubeTexture(`sweep:${sweep.uuid}`, files);
    texture.coordinatesMode = Texture.SKYBOX_MODE;

    const destPoseRotation = TourViewer.rotationFromPose(sweep);
    const destOrientation = destPoseRotation.multiply(SKYBOX_BASE);
    texture.setReflectionTextureMatrix(destOrientation);

    const destPos = new Vector3(sweep.position[0], sweep.position[1], sweep.position[2]);

    // T2: attempt animated transition if we have a departure sweep and
    // the transition controller is ready.
    const canTransition =
      this.transition &&
      this.activeTexture &&
      this.activeUuid &&
      this.activeUuid !== uuid;

    if (canTransition) {
      const departureCubemap = this.activeTexture!;
      // T2 alignment: pass pure pose rotation (world→scanner-local) — NOT
      // the combined pano orientation which includes Babylon's SKYBOX_BASE.
      const departureOrientation = this.getPoseRotation();
      const departurePos = this.camera.position.clone();

      // The midpoint callback swaps the skybox to the destination.
      const onMidpoint = () => {
        this.currentPoseRotation = destPoseRotation;
        this.skyboxMaterial.reflectionTexture = texture;
        this.activeTexture = texture;
        this.camera.position.copyFrom(destPos);
        this.applyLook();
        this.activeUuid = uuid;
        this.onSweepShown?.(uuid);
        // Detach the departure texture from cache (now owned by skybox).
        this.textureCache?.detach(uuid);
      };

      await this.transition!.flyTo(
        departureCubemap,
        departureOrientation,
        departurePos,
        destPos,
        this.camera,
        onMidpoint,
      );

      // Dispose the old departure texture AFTER the transition completes.
      if (departureCubemap !== texture) {
        departureCubemap.dispose();
      }

      // Prefetch neighbors of the new sweep.
      this.prefetchNeighbors(uuid);
      return;
    }

    // Instant jump (no transition available or first sweep).
    const previous = this.activeTexture;
    this.currentPoseRotation = destPoseRotation;
    this.skyboxMaterial.reflectionTexture = texture;
    this.activeTexture = texture;
    if (previous && previous !== texture) {
      previous.dispose(); // VRAM discipline
    }
    this.textureCache?.detach(uuid);

    this.camera.position.copyFrom(destPos);
    this.applyLook();
    this.activeUuid = uuid;
    this.onSweepShown?.(uuid);

    // Prefetch neighbors of the new sweep.
    this.prefetchNeighbors(uuid);
  }

  /** Simple jump to the previous / next sweep by capture index. */
  async showNeighbor(offset: number): Promise<void> {
    const current = this.tour.sweeps.findIndex((s) => s.uuid === this.activeUuid);
    if (current < 0) return;
    const next = this.tour.sweeps[current + offset];
    if (next) {
      await this.showSweep(next.uuid);
    }
  }

  dispose(): void {
    this.transition?.abort();
    this.detachControls?.();
    this.detachControls = null;
    this.textureCache?.dispose();
    this.activeTexture?.dispose();
    this.scene.dispose();
  }

  /**
   * T2: prefetch cubemaps for the nav-graph neighbors of the given sweep.
   * Uses the tour's sweep data to build file URLs.
   */
  private prefetchNeighbors(uuid: string): void {
    if (!this.textureCache || !this.tour.nav) return;
    // Build adjacency from the nav graph edges (reuse the edge list).
    const neighbors: string[] = [];
    for (const edge of this.tour.nav.edges) {
      if (edge.from === uuid) neighbors.push(edge.to);
      else if (edge.to === uuid) neighbors.push(edge.from);
    }
    // Sort by distance (edges are already sorted in the JSON), take top 3.
    const toFetch = neighbors.slice(0, 3);
    const items = toFetch
      .map((nUuid) => {
        const sw = this.tour.sweeps.find((s) => s.uuid === nUuid);
        if (!sw) return null;
        return { uuid: nUuid, baseUrl: this.tour.baseUrl, faces: sw.faces };
      })
      .filter((x): x is NonNullable<typeof x> => x !== null);
    this.textureCache.prefetch(items);
  }

  /**
   * Pose rotation block -> Babylon matrix. pose_rt is column-convention
   * (d_world = R * d_local); filling Babylon row-major storage VERBATIM makes
   * v * M act as R^T * v = world -> local, exactly the sampling map we need.
   */
  private static rotationFromPose(sweep: SweepData): Matrix {
    const r = sweep.rotation;
    if (!r || r.length !== 3) return Matrix.Identity();
    return Matrix.FromValues(
      r[0][0], r[0][1], r[0][2], 0,
      r[1][0], r[1][1], r[1][2], 0,
      r[2][0], r[2][1], r[2][2], 0,
      0, 0, 0, 1,
    );
  }

  /** The one and only place camera orientation is applied (all frames, all paths). */
  private applyLook(): void {
    const cosPitch = Math.cos(this.pitch);
    const direction = new Vector3(
      Math.sin(this.yaw) * cosPitch,
      Math.sin(this.pitch),
      Math.cos(this.yaw) * cosPitch,
    );
    this.camera.setTarget(this.camera.position.add(direction));
  }

  private attachPanoControls(canvas: HTMLCanvasElement | null): void {
    if (!canvas) return;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const onPointerDown = (event: PointerEvent) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      // Grab-the-world: the view rotates opposite to the drag direction.
      this.yaw -= dx * DRAG_SENSITIVITY;
      this.pitch = Math.max(
        -PITCH_LIMIT,
        Math.min(PITCH_LIMIT, this.pitch - dy * DRAG_SENSITIVITY),
      );
      this.applyLook();
    };
    const onPointerUp = (event: PointerEvent) => {
      dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    this.detachControls = () => {
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
    };
  }

  private loadCubeTexture(cacheKey: string, files: string[]): Promise<CubeTexture> {
    return new Promise((resolve, reject) => {
      const texture = new CubeTexture(
        cacheKey,
        this.scene,
        null,
        false,
        files,
        () => resolve(texture),
        (message) => reject(new Error(message ?? "cubemap load failed")),
      );
    });
  }
}

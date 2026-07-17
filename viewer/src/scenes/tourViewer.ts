import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import { Scene } from "@babylonjs/core/scene";
import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { Matrix, Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { CreateBox } from "@babylonjs/core/Meshes/Builders/boxBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import type { SweepData, TourData } from "../data/types";

/** Babylon CubeTexture file order: +x, +y, +z, -x, -y, -z. */
const FACE_ORDER = ["px", "py", "pz", "nx", "ny", "nz"] as const;

/** Drag sensitivity (radians per pixel) and pitch clamp for the pano camera. */
const DRAG_SENSITIVITY = 0.0025;
const PITCH_LIMIT = Math.PI / 2 - 0.02;

/**
 * Pano orientation chain (G7.1 root-cause discovery):
 *
 * The extracted canonical faces live in each sweep's LOCAL tripod frame --
 * NOT the world frame. poses.json's pose_rt rotation maps local -> world;
 * the viewer must therefore rotate every sampled direction world -> local
 * PER SWEEP before the canonical-layout conversion. F4.4's global constant
 * could never encode that (its criteria -- seams, up/down, readability,
 * continuity -- are all satisfiable per-sweep because tripods are level);
 * F5/G4/fusion never noticed because their render cameras are built FROM
 * the pose rotations, i.e. they compare in the local frame. The viewer is
 * the first world-aligned consumer of the panos.
 *
 * Chain (Babylon row-vector convention, v' = v * M, A.multiply(B) = A then B):
 *   sampleDir = worldDir * R_pose (acts as R^T = world->local)
 *                        * RotationY(yawCandidate)   (residual discrete offset)
 *                        * SKYBOX_BASE               (local -> cube layout, F4.4)
 *
 * The O key cycles the RESIDUAL yaw candidate (0/90/180/270): level tripods
 * mean any leftover offset between the canonical layout and the local frame
 * is a pure k*90-deg yaw on top of the F4.4 mirror parity. The I key flips
 * the rotation interpretation (transpose insurance) in case the convention
 * analysis is off by an inverse. Lock the winners and strip the harness.
 */
const SKYBOX_BASE: Matrix = Matrix.Scaling(-1, 1, 1).multiply(
  Matrix.RotationX(-Math.PI / 2),
);
const SKYBOX_YAW_CANDIDATES_DEG = [0, 90, 180, 270] as const;
// 180 deg predicted by the legacy 3d-viewer protocol (q_sweep * q_yawLocal(180)
// about local Z-up = RotationY(180) in the local Y-up frame) AND by the F5
// face-mapping +-X/+-Y-swap signature. Confirm visually, then lock.
const DEFAULT_YAW_INDEX = 2;

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
  private yawIndex = DEFAULT_YAW_INDEX;
  private yawFineDeg = 0;
  private currentPoseRotation: Matrix = Matrix.Identity();
  private invertRotation = false;
  private detachControls: (() => void) | null = null;

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

  /** Current residual yaw candidate (degrees, G7.1 calibration harness). */
  get skyboxYawDeg(): number {
    return SKYBOX_YAW_CANDIDATES_DEG[this.yawIndex];
  }

  /** Cycle the residual yaw candidate (O key). Returns the new value in degrees. */
  cycleSkyboxYaw(): number {
    this.yawIndex = (this.yawIndex + 1) % SKYBOX_YAW_CANDIDATES_DEG.length;
    this.applySkyboxOrientation(this.activeTexture);
    return this.skyboxYawDeg;
  }

  /**
   * Flip the pose-rotation interpretation (I key, transpose insurance).
   * After toggling, re-cycle O -- the residual yaw winner may differ.
   */
  toggleRotationInverse(): boolean {
    this.invertRotation = !this.invertRotation;
    console.info(
      `[tour] pose rotation interpretation: ${this.invertRotation ? "INVERTED (R, local->world)" : "normal (R^T, world->local)"} -- re-cycle O`,
    );
    this.applySkyboxOrientation(this.activeTexture);
    return this.invertRotation;
  }

  /** Toggle the pano skybox (P key): proxy-only debug view. Returns new state. */
  toggleSkybox(): boolean {
    const skybox = this.scene.getMeshByName("skybox");
    if (!skybox) return true;
    const next = !skybox.isEnabled();
    skybox.setEnabled(next);
    return next;
  }

  /** Load a sweep's cubemap and move the camera to its pose position. */
  async showSweep(uuid: string): Promise<void> {
    const sweep = this.tour.sweeps.find((s) => s.uuid === uuid);
    if (!sweep) {
      throw new Error(`unknown sweep: ${uuid}`);
    }
    const files = FACE_ORDER.map((face) => `${this.tour.baseUrl}/${sweep.faces[face]}`);
    // Unique cache key per sweep: Babylon caches cube textures by rootUrl.
    const texture = await this.loadCubeTexture(`sweep:${sweep.uuid}`, files);
    texture.coordinatesMode = Texture.SKYBOX_MODE;
    this.currentPoseRotation = TourViewer.rotationFromPose(sweep);
    this.applySkyboxOrientation(texture);

    const previous = this.activeTexture;
    this.skyboxMaterial.reflectionTexture = texture;
    this.activeTexture = texture;
    previous?.dispose(); // VRAM discipline

    this.camera.position.set(sweep.position[0], sweep.position[1], sweep.position[2]);
    this.applyLook();
    this.activeUuid = uuid;
    this.onSweepShown?.(uuid);
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
    this.detachControls?.();
    this.detachControls = null;
    this.activeTexture?.dispose();
    this.scene.dispose();
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

  /** Per-sweep pose rotation, then residual yaw candidate, then F4.4 base. */
  private applySkyboxOrientation(texture: CubeTexture | null): void {
    if (!texture) return;
    const rotation = this.invertRotation
      ? Matrix.Transpose(this.currentPoseRotation)
      : this.currentPoseRotation;
    const yawRad = (this.skyboxYawDeg * Math.PI) / 180;
    texture.setReflectionTextureMatrix(
      rotation.multiply(Matrix.RotationY(yawRad)).multiply(SKYBOX_BASE),
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

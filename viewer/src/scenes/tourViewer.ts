import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import { Scene } from "@babylonjs/core/scene";
import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { Matrix, Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { CreateBox } from "@babylonjs/core/Meshes/Builders/boxBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import type { TourData } from "../data/types";

/** Babylon CubeTexture file order: +x, +y, +z, -x, -y, -z. */
const FACE_ORDER = ["px", "py", "pz", "nx", "ny", "nz"] as const;

/** Drag sensitivity (radians per pixel) and pitch clamp for the pano camera. */
const DRAG_SENSITIVITY = 0.0025;
const PITCH_LIMIT = Math.PI / 2 - 0.02;

/**
 * F4.4 locked the mirror + Z-up->Y-up part of the pano orientation
 * (mirrorX . rotX-90 on the cube texture's reflection matrix, runtime-only,
 * Layer 1 pixels untouched).
 *
 * G7.1 finding: every F4.4 acceptance criterion (seams, up/down, text
 * readability, cross-sweep continuity) is INVARIANT under a global yaw about
 * the vertical axis -- the residual yaw only became observable once Layer 2
 * world geometry composited with the photography for the first time.
 *
 * The F5 face-mapping calibration already measured that residual's signature:
 * extracted faces have +-X AND +-Y swapped vs the rendered Matterport-frame
 * faces (= rotZ180 in the scanner frame = 180 deg yaw in the runtime Y-up
 * frame). Default candidate below is therefore 180 deg; the O key cycles
 * 0/90/180/270 for the X-ray calibration pass. Lock the winning value and
 * remove the harness afterwards -- same protocol as F4.4.
 */
const SKYBOX_BASE: Matrix = Matrix.Scaling(-1, 1, 1).multiply(
  Matrix.RotationX(-Math.PI / 2),
);
const SKYBOX_YAW_CANDIDATES_DEG = [0, 90, 180, 270] as const;
const DEFAULT_YAW_INDEX = 2; // 180 deg -- predicted by the F5 face-mapping evidence

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

  /** Current pano yaw-correction candidate (degrees, G7.1 calibration harness). */
  get skyboxYawDeg(): number {
    return SKYBOX_YAW_CANDIDATES_DEG[this.yawIndex];
  }

  /** Cycle the pano yaw candidate (O key). Returns the new value in degrees. */
  cycleSkyboxYaw(): number {
    this.yawIndex = (this.yawIndex + 1) % SKYBOX_YAW_CANDIDATES_DEG.length;
    this.applySkyboxOrientation(this.activeTexture);
    return this.skyboxYawDeg;
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
   * F4.4 base orientation composed with the G7.1 yaw candidate.
   *
   * COMPOSITION ORDER MATTERS (round-1 harness bug): Babylon's A.multiply(B)
   * applies A THEN B to the sampled direction. The yaw must act on the
   * WORLD-space direction BEFORE the world->cube conversion (SKYBOX_BASE),
   * so it PRE-multiplies. Post-multiplying rotated around the cube frame's
   * Y axis -- horizontal after rotX-90 -- pitching the pano down instead of
   * yawing it (observed at yaw 180).
   */
  private applySkyboxOrientation(texture: CubeTexture | null): void {
    if (!texture) return;
    const yawRad = (this.skyboxYawDeg * Math.PI) / 180;
    texture.setReflectionTextureMatrix(
      Matrix.RotationY(yawRad).multiply(SKYBOX_BASE),
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

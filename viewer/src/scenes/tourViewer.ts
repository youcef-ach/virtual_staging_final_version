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
 * F4.4 calibration: the extracted faces are in the scanner Z-up frame; the
 * viewer world is Y-up. The correction is a single global rotation applied to
 * the cube texture's reflection matrix at runtime (never baked into faces).
 * Cycle candidates with the O key (+ M for mirror) and lock the winner.
 */
const ORIENTATIONS: { label: string; matrix: Matrix }[] = [
  { label: "identity", matrix: Matrix.Identity() },
  { label: "rotX+90", matrix: Matrix.RotationX(Math.PI / 2) },
  { label: "rotX-90", matrix: Matrix.RotationX(-Math.PI / 2) },
  { label: "rotZ+90", matrix: Matrix.RotationZ(Math.PI / 2) },
  { label: "rotZ-90", matrix: Matrix.RotationZ(-Math.PI / 2) },
  { label: "rotX180", matrix: Matrix.RotationX(Math.PI) },
  { label: "rotY180", matrix: Matrix.RotationY(Math.PI) },
  {
    label: "rotX+90*rotY180",
    matrix: Matrix.RotationX(Math.PI / 2).multiply(Matrix.RotationY(Math.PI)),
  },
  {
    label: "rotX-90*rotY180",
    matrix: Matrix.RotationX(-Math.PI / 2).multiply(Matrix.RotationY(Math.PI)),
  },
];

/** Analytic guess for Z-up -> Y-up as the starting candidate. */
const DEFAULT_ORIENTATION_INDEX = 1;

/**
 * Layer 1 renderer: the photographic layer. The active sweep's cubemap is an
 * unlit skybox, rendered as-is, never modified (visual truth).
 *
 * Camera: inside-out look-around (FreeCamera + deterministic drag handler).
 * Yaw/pitch persist across sweep jumps for continuity.
 *
 * VRAM discipline: only the active sweep's cubemap stays resident.
 */
export class TourViewer {
  readonly scene: Scene;
  onSweepShown?: (uuid: string) => void;
  onCalibrationChanged?: (label: string) => void;

  private readonly camera: FreeCamera;
  private readonly skyboxMaterial: StandardMaterial;
  private activeTexture: CubeTexture | null = null;
  private activeUuid: string | null = null;
  private yaw = 0;
  private pitch = 0;
  private detachControls: (() => void) | null = null;
  private orientationIndex = DEFAULT_ORIENTATION_INDEX;
  private mirrored = false;

  constructor(
    engine: AbstractEngine,
    private readonly tour: TourData,
  ) {
    this.scene = new Scene(engine);
    // The runtime frame (Coordinate Lock Rule) is right-handed Y-up — match it.
    this.scene.useRightHandedSystem = true;
    this.scene.clearColor = new Color4(0, 0, 0, 1);

    this.camera = new FreeCamera("camera", Vector3.Zero(), this.scene);
    this.camera.minZ = 0.01;
    this.camera.fov = 1.15;
    this.camera.inputs.clear(); // deterministic pano controls below
    this.attachPanoControls(engine.getRenderingCanvas());

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

  get calibrationLabel(): string {
    const base = ORIENTATIONS[this.orientationIndex].label;
    return `${base}${this.mirrored ? " +mirror" : ""} [${this.orientationIndex}]`;
  }

  /** F4.4 calibration: cycle to the next candidate orientation. */
  cycleOrientation(): void {
    this.orientationIndex = (this.orientationIndex + 1) % ORIENTATIONS.length;
    this.applyOrientation();
  }

  /** F4.4 calibration: toggle a mirror on top of the current orientation. */
  toggleMirror(): void {
    this.mirrored = !this.mirrored;
    this.applyOrientation();
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
    texture.setReflectionTextureMatrix(this.currentOrientationMatrix());

    const previous = this.activeTexture;
    this.skyboxMaterial.reflectionTexture = texture;
    this.activeTexture = texture;
    previous?.dispose(); // VRAM discipline

    this.camera.position.set(sweep.position[0], sweep.position[1], sweep.position[2]);
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

  private currentOrientationMatrix(): Matrix {
    const base = ORIENTATIONS[this.orientationIndex].matrix;
    return this.mirrored ? Matrix.Scaling(-1, 1, 1).multiply(base) : base;
  }

  private applyOrientation(): void {
    this.activeTexture?.setReflectionTextureMatrix(this.currentOrientationMatrix());
    this.onCalibrationChanged?.(this.calibrationLabel);
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
      this.camera.rotation.set(this.pitch, this.yaw, 0);
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

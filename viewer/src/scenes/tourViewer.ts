import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import { Scene } from "@babylonjs/core/scene";
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { CreateBox } from "@babylonjs/core/Meshes/Builders/boxBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import type { TourData } from "../data/types";

/** Babylon CubeTexture file order: +x, +y, +z, -x, -y, -z. */
const FACE_ORDER = ["px", "py", "pz", "nx", "ny", "nz"] as const;

/**
 * Layer 1 renderer: the photographic layer. The active sweep's cubemap is an
 * unlit skybox, rendered as-is, never modified (visual truth).
 *
 * VRAM discipline: only the active sweep's cubemap stays resident; the
 * previous texture is disposed right after the swap.
 */
export class TourViewer {
  readonly scene: Scene;
  onSweepShown?: (uuid: string) => void;

  private readonly camera: ArcRotateCamera;
  private readonly skyboxMaterial: StandardMaterial;
  private activeTexture: CubeTexture | null = null;
  private activeUuid: string | null = null;

  constructor(
    engine: AbstractEngine,
    private readonly tour: TourData,
  ) {
    this.scene = new Scene(engine);
    // The runtime frame (Coordinate Lock Rule) is right-handed Y-up — match it.
    this.scene.useRightHandedSystem = true;
    this.scene.clearColor = new Color4(0, 0, 0, 1);

    this.camera = new ArcRotateCamera(
      "camera",
      -Math.PI / 2,
      Math.PI / 2,
      0.05,
      Vector3.Zero(),
      this.scene,
    );
    this.camera.attachControl(true);
    this.camera.lowerRadiusLimit = 0.05;
    this.camera.upperRadiusLimit = 0.05;
    this.camera.panningSensibility = 0;
    this.camera.inputs.removeByType("ArcRotateCameraMouseWheelInput");
    this.camera.minZ = 0.01;
    this.camera.fov = 1.15;

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

  /** Load a sweep's cubemap and move the camera to its pose position. */
  async showSweep(uuid: string): Promise<void> {
    const sweep = this.tour.sweeps.find((s) => s.uuid === uuid);
    if (!sweep) {
      throw new Error(`unknown sweep: ${uuid}`);
    }
    const files = FACE_ORDER.map((face) => `${this.tour.baseUrl}/${sweep.faces[face]}`);
    const texture = await this.loadCubeTexture(files);
    texture.coordinatesMode = Texture.SKYBOX_MODE;

    const previous = this.activeTexture;
    this.skyboxMaterial.reflectionTexture = texture;
    this.activeTexture = texture;
    previous?.dispose(); // VRAM discipline

    this.camera.setTarget(new Vector3(sweep.position[0], sweep.position[1], sweep.position[2]));
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
    this.activeTexture?.dispose();
    this.scene.dispose();
  }

  private loadCubeTexture(files: string[]): Promise<CubeTexture> {
    return new Promise((resolve, reject) => {
      const texture = new CubeTexture(
        "",
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

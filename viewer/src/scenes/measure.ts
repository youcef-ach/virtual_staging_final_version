/**
 * T5 -- Measurement tools: point-to-point distance and area measurement.
 *
 * The user clicks points on the proxy mesh (LiDAR-accurate geometry) to measure
 * real-world distances. Two modes:
 *
 *   1. **Distance mode** (default): click two points → shows distance in meters
 *      with a line connecting them.
 *   2. **Area mode**: click 3+ points to define a polygon → shows area in m².
 *
 * Toggle measurement mode with the **L** key ("line/length").
 * While in measurement mode:
 *   - Click proxy/floor surfaces to add points
 *   - Right-click or Escape to finish the current measurement
 *   - Click L again to exit measurement mode entirely
 *
 * Measurements are rendered as 3D overlays (lines + labels) in GROUP_ASSET
 * so they appear on top of the proxy but are occluded correctly.
 */
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { DynamicTexture } from "@babylonjs/core/Materials/Textures/dynamicTexture";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { Scene } from "@babylonjs/core/scene";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import { GROUP_ASSET } from "./staging";

const POINT_RADIUS = 0.03;
const LINE_WIDTH = 0.01;

export type MeasureMode = "off" | "distance" | "area";

export class MeasureTool {
  private readonly scene: Scene;
  private _mode: MeasureMode = "off";
  private points: Vector3[] = [];
  private pointMeshes: Mesh[] = [];
  private lineMeshes: Mesh[] = [];
  private labelMeshes: Mesh[] = [];
  private matPoint: StandardMaterial;
  private matLine: StandardMaterial;

  constructor(scene: Scene) {
    this.scene = scene;

    this.matPoint = new StandardMaterial("measure-point-mat", scene);
    this.matPoint.diffuseColor = new Color3(1, 0.3, 0.1);
    this.matPoint.emissiveColor = new Color3(0.6, 0.15, 0.05);
    this.matPoint.disableLighting = true;

    this.matLine = new StandardMaterial("measure-line-mat", scene);
    this.matLine.diffuseColor = new Color3(1, 1, 0.2);
    this.matLine.emissiveColor = new Color3(0.5, 0.5, 0.1);
    this.matLine.disableLighting = true;
  }

  get mode(): MeasureMode {
    return this._mode;
  }

  /** Cycle measurement mode: off → distance → area → off. */
  cycleMode(): MeasureMode {
    this.clearCurrent();
    if (this._mode === "off") {
      this._mode = "distance";
    } else if (this._mode === "distance") {
      this._mode = "area";
    } else {
      this._mode = "off";
    }
    console.info(`[measure] mode: ${this._mode}`);
    return this._mode;
  }

  /** Force measurement mode off. */
  disable(): void {
    this.clearCurrent();
    this._mode = "off";
  }

  /**
   * Add a measurement point by picking the proxy/floor at screen coords.
   * Returns a status string for the HUD.
   */
  addPoint(x: number, y: number): string | null {
    if (this._mode === "off") return null;

    // Pick against all meshes (proxy + floor surfaces).
    const pick = this.scene.pick(x, y, (m) => {
      // Only pick proxy/floor meshes (not assets, skybox, markers, etc.)
      return (
        m.renderingGroupId === 1 && // GROUP_PROXY
        m.isPickable &&
        m.isEnabled() &&
        !m.name.startsWith("nav-") &&
        !m.name.startsWith("dh-") &&
        !m.name.startsWith("measure-")
      );
    });

    if (!pick?.hit || !pick.pickedPoint) {
      return null;
    }

    const point = pick.pickedPoint.clone();
    this.points.push(point);

    // Create point marker.
    const sphere = MeshBuilder.CreateSphere(
      `measure-point-${this.pointMeshes.length}`,
      { diameter: POINT_RADIUS * 2, segments: 8 },
      this.scene,
    );
    sphere.position.copyFrom(point);
    sphere.material = this.matPoint;
    sphere.renderingGroupId = GROUP_ASSET;
    sphere.isPickable = false;
    this.pointMeshes.push(sphere);

    // Distance mode: complete after 2 points.
    if (this._mode === "distance" && this.points.length === 2) {
      return this.finishDistance();
    }

    // Area mode: draw lines as we go, complete on right-click/escape.
    if (this._mode === "area" && this.points.length >= 2) {
      this.drawLine(
        this.points[this.points.length - 2],
        this.points[this.points.length - 1],
      );
    }

    if (this._mode === "distance") {
      return `📏 click second point`;
    }
    return `📐 ${this.points.length} points (right-click to finish)`;
  }

  /** Finish the current area measurement (right-click or Escape). */
  finishArea(): string | null {
    if (this._mode !== "area" || this.points.length < 3) {
      return null;
    }

    // Close the polygon.
    this.drawLine(this.points[this.points.length - 1], this.points[0]);

    // Compute area using the Shoelace formula in 3D (project to best-fit plane).
    const area = this.computePolygonArea(this.points);
    const label = `${area.toFixed(2)} m²`;
    this.createLabel(this.centroid(this.points), label);
    console.info(`[measure] area: ${label}`);

    // Keep the visualization, start a new measurement.
    this.points = [];
    this.pointMeshes = [];
    this.lineMeshes = [];
    this.labelMeshes = [];

    return `📐 area: ${label}`;
  }

  private finishDistance(): string {
    const p1 = this.points[0];
    const p2 = this.points[1];
    const dist = Vector3.Distance(p1, p2);
    const label = `${dist.toFixed(3)} m`;

    this.drawLine(p1, p2);
    this.createLabel(Vector3.Lerp(p1, p2, 0.5), label);
    console.info(`[measure] distance: ${label}`);

    // Keep the visualization, start a new measurement.
    this.points = [];
    this.pointMeshes = [];
    this.lineMeshes = [];
    this.labelMeshes = [];

    return `📏 ${label}`;
  }

  private drawLine(from: Vector3, to: Vector3): void {
    const line = MeshBuilder.CreateTube(
      `measure-line-${this.lineMeshes.length}`,
      {
        path: [from, to],
        radius: LINE_WIDTH / 2,
        tessellation: 8,
        updatable: false,
      },
      this.scene,
    );
    line.material = this.matLine;
    line.renderingGroupId = GROUP_ASSET;
    line.isPickable = false;
    this.lineMeshes.push(line);
  }

  private createLabel(position: Vector3, text: string): void {
    // Dynamic texture for the label.
    const texSize = 256;
    const texture = new DynamicTexture(
      `measure-label-tex-${this.labelMeshes.length}`,
      { width: texSize, height: texSize / 4 },
      this.scene,
      false,
    );
    const ctx = texture.getContext() as unknown as CanvasRenderingContext2D;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(0, 0, texSize, texSize / 4);
    ctx.font = "bold 32px Inter, sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, texSize / 2, texSize / 8);
    texture.update();

    const mat = new StandardMaterial(
      `measure-label-mat-${this.labelMeshes.length}`,
      this.scene,
    );
    mat.diffuseTexture = texture;
    mat.emissiveColor = new Color3(1, 1, 1);
    mat.disableLighting = true;
    mat.backFaceCulling = false;
    mat.useAlphaFromDiffuseTexture = true;

    const plane = MeshBuilder.CreatePlane(
      `measure-label-${this.labelMeshes.length}`,
      { width: 0.4, height: 0.1 },
      this.scene,
    );
    plane.position.copyFrom(position);
    plane.position.y += 0.15; // lift above the measurement line
    plane.billboardMode = 7; // always face camera (all axes)
    plane.material = mat;
    plane.renderingGroupId = GROUP_ASSET;
    plane.isPickable = false;
    this.labelMeshes.push(plane);
  }

  /** Clear the current in-progress measurement. */
  private clearCurrent(): void {
    for (const m of this.pointMeshes) m.dispose();
    for (const m of this.lineMeshes) m.dispose();
    for (const m of this.labelMeshes) {
      (m.material as StandardMaterial)?.diffuseTexture?.dispose();
      m.material?.dispose();
      m.dispose();
    }
    this.pointMeshes = [];
    this.lineMeshes = [];
    this.labelMeshes = [];
    this.points = [];
  }

  private centroid(points: Vector3[]): Vector3 {
    const sum = Vector3.Zero();
    for (const p of points) sum.addInPlace(p);
    return sum.scale(1 / points.length);
  }

  /**
   * Compute the area of a 3D polygon using the Shoelace formula projected
   * onto the polygon's best-fit plane (via the cross-product method).
   */
  private computePolygonArea(points: Vector3[]): number {
    if (points.length < 3) return 0;

    // Compute the polygon normal via Newell's method.
    const normal = Vector3.Zero();
    for (let i = 0; i < points.length; i++) {
      const curr = points[i];
      const next = points[(i + 1) % points.length];
      normal.x += (curr.y - next.y) * (curr.z + next.z);
      normal.y += (curr.z - next.z) * (curr.x + next.x);
      normal.z += (curr.x - next.x) * (curr.y + next.y);
    }
    const len = normal.length();
    if (len < 1e-10) return 0;

    // The area = |normal| / 2 (Newell's method gives 2× the area as the
    // magnitude of the accumulated cross product).
    return len / 2;
  }
}

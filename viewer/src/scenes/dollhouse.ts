/**
 * T3 -- Dollhouse view: textured Matterpak OBJ with orbit camera and clickable
 * sweep position markers.
 *
 * Toggle with the H key ("house"). In dollhouse mode:
 *   - The reference mesh (reference.glb) is shown with its original textures.
 *   - The camera switches to ArcRotateCamera orbiting the model centroid.
 *   - Small sphere markers appear at each sweep position (clickable → jump).
 *   - The active sweep marker is highlighted.
 *   - All other layers (skybox, proxy, asset, shadow, hotspots) are hidden.
 *
 * Exiting dollhouse mode restores the pano view exactly as it was.
 */
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { Scene } from "@babylonjs/core/scene";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import type { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import type { SweepData } from "../data/types";
import { GROUP_PROXY } from "./staging";

const MARKER_RADIUS = 0.12;
const ACTIVE_MARKER_RADIUS = 0.18;

export class DollhouseView {
  private readonly scene: Scene;
  private orbitCamera: ArcRotateCamera | null = null;
  private readonly markers: Mesh[] = [];
  private readonly markerUuids: string[] = [];
  private matDefault: StandardMaterial;
  private matActive: StandardMaterial;
  private _active = false;

  /** Meshes/nodes that were visible before entering dollhouse (to restore). */
  private savedVisibility = new Map<TransformNode, boolean>();
  private previousClearColor: Color4 | null = null;

  constructor(scene: Scene) {
    this.scene = scene;

    this.matDefault = new StandardMaterial("dh-marker-default", scene);
    this.matDefault.diffuseColor = new Color3(0.2, 0.6, 1.0);
    this.matDefault.emissiveColor = new Color3(0.1, 0.3, 0.5);
    this.matDefault.alpha = 0.85;

    this.matActive = new StandardMaterial("dh-marker-active", scene);
    this.matActive.diffuseColor = new Color3(1.0, 0.4, 0.1);
    this.matActive.emissiveColor = new Color3(0.6, 0.2, 0.05);
    this.matActive.alpha = 1.0;
  }

  get isActive(): boolean {
    return this._active;
  }

  /** Initialize sweep markers. Call once after tour data is loaded. */
  initMarkers(sweeps: SweepData[]): void {
    for (const sweep of sweeps) {
      const marker = MeshBuilder.CreateSphere(
        `dh-marker-${sweep.index}`,
        { diameter: MARKER_RADIUS * 2, segments: 8 },
        this.scene,
      );
      marker.position.set(sweep.position[0], sweep.position[1], sweep.position[2]);
      marker.material = this.matDefault;
      marker.isPickable = true;
      marker.setEnabled(false); // hidden until dollhouse mode
      this.markers.push(marker);
      this.markerUuids.push(sweep.uuid);
    }
  }

  /** Highlight the active sweep marker. */
  setActiveSweep(uuid: string): void {
    for (let i = 0; i < this.markers.length; i++) {
      if (this.markerUuids[i] === uuid) {
        this.markers[i].material = this.matActive;
        this.markers[i].scaling.setAll(ACTIVE_MARKER_RADIUS / MARKER_RADIUS);
      } else {
        this.markers[i].material = this.matDefault;
        this.markers[i].scaling.setAll(1);
      }
    }
  }

  /**
   * Enter dollhouse mode: show the reference mesh with orbit camera.
   * Returns false if no reference mesh is available.
   */
  enter(panoCamera: FreeCamera): boolean {
    const refRoot = this.scene.getTransformNodeByName("reference-root");
    if (!refRoot) {
      console.info("[dollhouse] no reference.glb loaded");
      return false;
    }

    this._active = true;
    this.previousClearColor = this.scene.clearColor.clone();
    this.scene.clearColor = new Color4(0.87, 0.89, 0.85, 1);

    this.savedVisibility.clear();
    const hide = (node: TransformNode) => {
      this.savedVisibility.set(node, node.isEnabled(false));
      node.setEnabled(false);
    };
    const asset = this.scene.getTransformNodeByName("asset-root");
    if (asset) hide(asset);
    for (const mesh of this.scene.meshes) {
      if (mesh.name === "skybox" || mesh.name.startsWith("nav-marker-") ||
        (mesh.renderingGroupId === GROUP_PROXY && !mesh.isDescendantOf(refRoot))) hide(mesh);
    }

    // Show reference mesh.
    this.savedVisibility.set(refRoot, refRoot.isEnabled(false));
    refRoot.setEnabled(true);

    // Show sweep markers.
    for (const marker of this.markers) {
      marker.setEnabled(true);
    }

    // Deactivate pano camera.
    panoCamera.detachControl();

    // Create orbit camera targeting the reference mesh centroid.
    const bounds = this.computeRefBounds(refRoot);
    const center = bounds.center;
    const radius = bounds.radius * 2.1;

    if (!this.orbitCamera) {
      this.orbitCamera = new ArcRotateCamera(
        "dh-orbit-camera",
        -Math.PI / 4,     // alpha (azimuth)
        Math.PI / 4,      // beta (elevation)
        radius,
        center,
        this.scene,
      );
      this.orbitCamera.minZ = 0.1;
      this.orbitCamera.wheelPrecision = 10;
      this.orbitCamera.lowerRadiusLimit = radius * 0.3;
      this.orbitCamera.upperRadiusLimit = radius * 3;
      this.orbitCamera.panningSensibility = 50;
    } else {
      this.orbitCamera.setTarget(center);
      this.orbitCamera.radius = radius;
    }

    this.scene.activeCamera = this.orbitCamera;
    const canvas = this.scene.getEngine().getRenderingCanvas();
    if (canvas) this.orbitCamera.attachControl(canvas, true);

    return true;
  }

  /** Exit dollhouse mode: restore pano view. */
  exit(panoCamera: FreeCamera): void {
    if (!this._active) return;
    this._active = false;

    // Detach orbit camera.
    this.orbitCamera?.detachControl();

    // Restore pano camera.
    this.scene.activeCamera = panoCamera;
    const canvas = this.scene.getEngine().getRenderingCanvas();
    if (canvas) panoCamera.attachControl(canvas, false);

    // Hide sweep markers.
    for (const marker of this.markers) {
      marker.setEnabled(false);
    }

    // Restore saved visibility.
    for (const [node, wasEnabled] of this.savedVisibility) node.setEnabled(wasEnabled);
    this.savedVisibility.clear();
    if (this.previousClearColor) this.scene.clearColor = this.previousClearColor;
    this.previousClearColor = null;
  }

  /** Pick a sweep marker at screen coords. Returns the target UUID, or null. */
  pick(x: number, y: number): string | null {
    const hit = this.scene.pick(x, y, (m) => this.markers.includes(m as Mesh));
    if (!hit?.hit || !hit.pickedMesh) return null;
    const idx = this.markers.indexOf(hit.pickedMesh as Mesh);
    if (idx < 0) return null;
    return this.markerUuids[idx];
  }

  private computeRefBounds(refRoot: TransformNode): { center: Vector3; radius: number } {
    let min = new Vector3(Infinity, Infinity, Infinity);
    let max = new Vector3(-Infinity, -Infinity, -Infinity);
    let found = false;

    const visit = (node: TransformNode) => {
      if ("getBoundingInfo" in node) {
        const mesh = node as AbstractMesh;
        if (mesh.getTotalVertices() > 0) {
          mesh.computeWorldMatrix(true);
          const bi = mesh.getBoundingInfo();
          const wMin = bi.boundingBox.minimumWorld;
          const wMax = bi.boundingBox.maximumWorld;
          min = Vector3.Minimize(min, wMin);
          max = Vector3.Maximize(max, wMax);
          found = true;
        }
      }
      for (const child of node.getChildren()) {
        visit(child as TransformNode);
      }
    };
    visit(refRoot);

    if (!found) {
      return { center: Vector3.Zero(), radius: 10 };
    }
    const center = Vector3.Lerp(min, max, 0.5);
    const radius = Vector3.Distance(min, max) / 2;
    return { center, radius };
  }
}

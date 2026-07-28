/**
 * T2 -- projective morphing transition: animates the camera from sweep A to
 * sweep B along a cubic Bézier, with the departure pano projected onto the
 * reference mesh (MatterPak OBJ/GLB) and a final cross-fade to the
 * destination skybox.
 *
 * Timeline (default 800 ms):
 *   t=0 → t=crossFadeStart: fly phase (camera moves, departure pano projected)
 *   t=crossFadeStart → t=end: cross-fade (destination skybox alpha 0→1,
 *     projected mesh fades out)
 *
 * Fallback: if no reference mesh is loaded, the transition degrades to an
 * instant jump (the existing showSweep behavior).
 */
import { ShaderMaterial } from "@babylonjs/core/Materials/shaderMaterial";
import { Effect } from "@babylonjs/core/Materials/effect";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { Matrix } from "@babylonjs/core/Maths/math.vector";
import type { Scene } from "@babylonjs/core/scene";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import type { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import type { Material } from "@babylonjs/core/Materials/material";
import type { Observer } from "@babylonjs/core/Misc/observable";

const DEFAULT_DURATION_MS = 800;
const CROSS_FADE_MS = 150;
/** Bézier control points are lifted this much above the straight A→B line. */
const BEZIER_LIFT_M = 0.3;

// ---------------------------------------------------------------------------
// Projective cubemap shader (inline GLSL)
// ---------------------------------------------------------------------------

const VERTEX_SHADER = `
precision highp float;

attribute vec3 position;
uniform mat4 world;
uniform mat4 worldViewProjection;

varying vec3 vWorldPos;

void main() {
  vWorldPos = (world * vec4(position, 1.0)).xyz;
  gl_Position = worldViewProjection * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = `
precision highp float;

uniform vec3 sweepPosition;
uniform samplerCube departureTex;
uniform mat4 orientationMatrix;
uniform float alpha;

varying vec3 vWorldPos;

void main() {
  vec3 dir = normalize(vWorldPos - sweepPosition);
  // Step 1: Pose rotation → left-handed Y-up scanner-local direction.
  // (The E57 rotation_matrix is in Matterport's left-handed Z-up frame;
  //  after the pipeline's similarity transform it becomes LH Y-up.)
  dir = (orientationMatrix * vec4(dir, 0.0)).xyz;
  // Step 2+3 combined: LH Y-up → RH Z-up (matches cubemap face layout).
  //   Y-up→Z-up: (x,y,z) → (x, -z, y)   then
  //   LH→RH:     (x,y,z) → (-x, -y, z)
  //   Combined:   (a,b,c) → (-a, c, b)
  dir = vec3(-dir.x, dir.z, dir.y);
  vec4 color = textureCube(departureTex, dir);
  gl_FragColor = vec4(color.rgb, alpha);
}
`;

// Register the shader once.
const SHADER_NAME = "projective-pano";

let _shaderRegistered = false;
function ensureShaderRegistered(): void {
  if (_shaderRegistered) return;
  Effect.ShadersStore[`${SHADER_NAME}VertexShader`] = VERTEX_SHADER;
  Effect.ShadersStore[`${SHADER_NAME}FragmentShader`] = FRAGMENT_SHADER;
  _shaderRegistered = true;
}

// ---------------------------------------------------------------------------
// Cubic Bézier helper
// ---------------------------------------------------------------------------

function cubicBezier(
  p0: Vector3,
  p1: Vector3,
  p2: Vector3,
  p3: Vector3,
  t: number,
): Vector3 {
  const u = 1 - t;
  const tt = t * t;
  const uu = u * u;
  return p0
    .scale(uu * u)
    .add(p1.scale(3 * uu * t))
    .add(p2.scale(3 * u * tt))
    .add(p3.scale(tt * t));
}

/** Smooth ease-in-out (Hermite). */
function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------
// SweepTransition
// ---------------------------------------------------------------------------

export class SweepTransition {
  private readonly scene: Scene;
  private readonly referenceMeshes: AbstractMesh[];
  private readonly savedMaterials: (Material | null)[] = [];
  private projMaterial: ShaderMaterial | null = null;
  private animObserver: Observer<Scene> | null = null;
  private _isTransitioning = false;

  constructor(scene: Scene, referenceMeshes: AbstractMesh[]) {
    this.scene = scene;
    this.referenceMeshes = referenceMeshes;
    // Save original materials for restoration.
    for (const mesh of referenceMeshes) {
      this.savedMaterials.push(mesh.material);
    }
  }

  get isTransitioning(): boolean {
    return this._isTransitioning;
  }

  /**
   * Animate from the current camera position (departure) to the destination
   * sweep. The departure cubemap is projected onto the reference mesh; the
   * destination cubemap is cross-faded in at the end.
   *
   * @param departureCubemap The active sweep's cubemap texture.
   * @param departureOrientation The pure pose rotation (world→scanner-local, no SKYBOX_BASE).
   * @param departurePosition The departure sweep's camera position.
   * @param destinationPosition The destination sweep's camera position.
   * @param camera The scene camera.
   * @param onMidpoint Called when the camera reaches the destination position
   *   (before cross-fade ends) so the caller can swap skybox/IBL/shadows.
   * @param durationMs Total transition duration (default 800 ms).
   */
  flyTo(
    departureCubemap: CubeTexture,
    departureOrientation: Matrix,
    departurePosition: Vector3,
    destinationPosition: Vector3,
    camera: FreeCamera,
    onMidpoint: () => void,
    durationMs: number = DEFAULT_DURATION_MS,
  ): Promise<void> {
    if (this._isTransitioning) {
      return Promise.resolve();
    }
    if (this.referenceMeshes.length === 0) {
      return Promise.resolve();
    }

    this._isTransitioning = true;

    return new Promise<void>((resolve) => {
      ensureShaderRegistered();

      // Create the projective material.
      const mat = new ShaderMaterial(
        "projective-mat",
        this.scene,
        { vertex: SHADER_NAME, fragment: SHADER_NAME },
        {
          attributes: ["position"],
          uniforms: [
            "world",
            "worldViewProjection",
            "sweepPosition",
            "orientationMatrix",
            "alpha",
          ],
          samplers: ["departureTex"],
        },
      );
      mat.setVector3("sweepPosition", departurePosition);
      mat.setMatrix("orientationMatrix", departureOrientation);
      mat.setTexture("departureTex", departureCubemap);
      mat.setFloat("alpha", 1.0);
      mat.backFaceCulling = false;
      this.projMaterial = mat;

      // Apply projective material to reference meshes and make them visible.
      for (const mesh of this.referenceMeshes) {
        mesh.material = mat;
      }
      // Show the reference root (it may be hidden via B toggle).
      const refRoot = this.scene.getTransformNodeByName("reference-root");
      const refWasEnabled = refRoot?.isEnabled() ?? false;
      refRoot?.setEnabled(true);

      // Hide proxy occluder meshes (GROUP_PROXY = 1) that aren't part of the
      // reference mesh hierarchy. Their depth-only writes would block the
      // projective mesh, creating black patches.
      const hiddenProxies: AbstractMesh[] = [];
      for (const mesh of this.scene.meshes) {
        if (mesh.renderingGroupId === 1 && mesh.isEnabled()) {
          // Skip any mesh that is a descendant of the reference root
          // (includes __root__ nodes and actual geometry meshes).
          let isRef = false;
          let node: import("@babylonjs/core/node").Node | null = mesh;
          while (node) {
            if (node.name === "reference-root") { isRef = true; break; }
            node = node.parent;
          }
          if (!isRef) {
            mesh.setEnabled(false);
            hiddenProxies.push(mesh);
          }
        }
      }

      // Compute Bézier control points.
      const p0 = departurePosition.clone();
      const p3 = destinationPosition.clone();
      const mid = Vector3.Lerp(p0, p3, 0.5);
      mid.y += BEZIER_LIFT_M;
      const p1 = Vector3.Lerp(p0, mid, 0.66);
      const p2 = Vector3.Lerp(mid, p3, 0.33);

      const crossFadeStart = durationMs - CROSS_FADE_MS;
      const startTime = performance.now();
      let midpointFired = false;

      // Get the skybox mesh for cross-fade alpha control.
      const skybox = this.scene.getMeshByName("skybox");

      // Animation loop.
      this.animObserver = this.scene.onBeforeRenderObservable.add(() => {
        const elapsed = performance.now() - startTime;
        const t = Math.min(elapsed / durationMs, 1.0);
        const eased = smoothstep(t);

        // Move camera along Bézier.
        const pos = cubicBezier(p0, p1, p2, p3, eased);
        camera.position.copyFrom(pos);

        // Cross-fade phase.
        if (elapsed >= crossFadeStart) {
          const fadeT = Math.min(
            (elapsed - crossFadeStart) / CROSS_FADE_MS,
            1.0,
          );
          // Fade out projected mesh.
          mat.setFloat("alpha", 1.0 - fadeT);
          // Fade in skybox (destination pano).
          if (skybox) {
            skybox.visibility = fadeT;
          }

          // Fire midpoint callback once (swap skybox/IBL to destination).
          if (!midpointFired) {
            midpointFired = true;
            // Temporarily hide skybox while we swap its texture.
            if (skybox) skybox.visibility = 0;
            onMidpoint();
          }
        } else {
          // During fly phase: hide the destination skybox.
          if (skybox) skybox.visibility = 0;
        }

        // Transition complete.
        if (t >= 1.0) {
          this.cleanup(refWasEnabled, hiddenProxies);
          resolve();
        }
      });
    });
  }

  /** Abort a running transition immediately (e.g. on dispose). */
  abort(): void {
    if (this._isTransitioning) {
      this.cleanup(false, []);
    }
  }

  private cleanup(restoreRefVisibility: boolean, hiddenProxies: AbstractMesh[]): void {
    // Remove the animation observer.
    if (this.animObserver) {
      this.scene.onBeforeRenderObservable.remove(this.animObserver);
      this.animObserver = null;
    }

    // Restore reference mesh materials.
    for (let i = 0; i < this.referenceMeshes.length; i++) {
      this.referenceMeshes[i].material = this.savedMaterials[i];
    }

    // Restore skybox full visibility.
    const skybox = this.scene.getMeshByName("skybox");
    if (skybox) skybox.visibility = 1.0;

    // Restore reference root visibility to what it was before the transition.
    const refRoot = this.scene.getTransformNodeByName("reference-root");
    if (refRoot && !restoreRefVisibility) {
      refRoot.setEnabled(false);
    }

    // Restore proxy occluder meshes that were hidden.
    for (const mesh of hiddenProxies) {
      mesh.setEnabled(true);
    }

    // Dispose projective material.
    this.projMaterial?.dispose();
    this.projMaterial = null;

    this._isTransitioning = false;
  }
}

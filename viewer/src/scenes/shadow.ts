import "@babylonjs/core/Lights/Shadows/shadowGeneratorSceneComponent";
import { DirectionalLight } from "@babylonjs/core/Lights/directionalLight";
import { ShadowGenerator } from "@babylonjs/core/Lights/Shadows/shadowGenerator";
import { ShadowOnlyMaterial } from "@babylonjs/materials/shadowOnly/shadowOnlyMaterial";
import { CreateGround } from "@babylonjs/core/Meshes/Builders/groundBuilder";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import type { Scene } from "@babylonjs/core/scene";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import type { BaseTexture } from "@babylonjs/core/Materials/Textures/baseTexture";
import { GROUP_ASSET } from "./staging";

/** Catcher plane size around the asset (meters) and z-fighting lift. */
const CATCHER_SIZE_M = 4;
const CATCHER_LIFT_M = 0.003;
const SHADOW_MAP_SIZE = 1024;
/** Plan formula: final = background * (0.35 + 0.65 * shadowVisibility). */
const SHADOW_FLOOR = 0.35;
/** Shadow light must point meaningfully DOWN for a floor contact shadow. */
const MIN_DOWN_COMPONENT = 0.35;
const FALLBACK_DIRECTION = new Vector3(-0.4, -1.0, -0.3);
const LUMA = { r: 0.2126, g: 0.7152, b: 0.0722 } as const;

/**
 * G7.3 -- contact shadow: the missing grounding cue (the G7.1 grounding delta
 * proved the asset IS at LiDAR floor level; it FELT floating because nothing
 * anchored it visually).
 *
 * Design:
 * - A shadow-catcher ground is PARENTED to asset-root, so it follows every
 *   placement/nudge with zero event wiring; ShadowOnlyMaterial shows nothing
 *   but the received shadow, multiplying the photographic Layer 1 exactly per
 *   the plan formula (ShadowGenerator darkness = 0.35 keeps 35% floor).
 * - The shadow light DIRECTION derives from the active sweep's IBL: the SH
 *   polynomial's linear band (x, y, z terms, luminance-weighted) points toward
 *   the dominant light -- negate it, clamp downward, and the shadow falls away
 *   from the real window. Sweeps without an .env use a constant fallback.
 * - No double-lighting by construction: light.includedOnlyMeshes = [catcher],
 *   so the IBL environment stays the asset's ONLY light source; asset meshes
 *   are shadow CASTERS only (the caster render list ignores inclusion lists).
 * - GROUP_ASSET keeps the catcher depth-tested against the proxy: a shadow
 *   plane behind a real wall is occluded like the asset itself.
 */
export class ContactShadow {
  private readonly light: DirectionalLight;
  private readonly generator: ShadowGenerator;
  private catcher: Mesh | null = null;
  private enabled = true;

  constructor(private readonly scene: Scene) {
    this.light = new DirectionalLight(
      "shadow-sun",
      FALLBACK_DIRECTION.clone().normalize(),
      scene,
    );
    this.light.intensity = 1.0;
    this.light.shadowMinZ = 0.05;
    this.light.shadowMaxZ = 30;
    this.generator = new ShadowGenerator(SHADOW_MAP_SIZE, this.light);
    this.generator.useBlurExponentialShadowMap = true;
    this.generator.blurKernel = 32;
    this.generator.setDarkness(SHADOW_FLOOR);
    this.generator.bias = 0.002;
    // Track the light above the catcher so the ortho shadow frustum always
    // covers the asset wherever it gets placed.
    scene.onBeforeRenderObservable.add(() => {
      if (this.catcher) {
        this.light.position.copyFrom(
          this.catcher.getAbsolutePosition().subtract(this.light.direction.scale(6)),
        );
      }
    });
  }

  /**
   * Create the catcher under the loaded asset (call after loadAsset resolved).
   * Looks the asset up by its stable node name -- staging.ts stays untouched.
   */
  attachToAsset(): boolean {
    const root = this.scene.getTransformNodeByName("asset-root");
    if (!root) {
      console.info("[shadow] no asset-root in scene -- catcher not created");
      return false;
    }
    if (!this.catcher) {
      const catcher = CreateGround(
        "shadow-catcher",
        { width: CATCHER_SIZE_M, height: CATCHER_SIZE_M },
        this.scene,
      );
      const material = new ShadowOnlyMaterial("shadow-catcher-mat", this.scene);
      material.activeLight = this.light;
      material.shadowColor = Color3.Black();
      catcher.material = material;
      catcher.receiveShadows = true;
      catcher.renderingGroupId = GROUP_ASSET;
      catcher.isPickable = false; // placement raycasts must reach the proxy floor
      this.catcher = catcher;
      // The shadow light affects ONLY the catcher (no asset double-lighting).
      this.light.includedOnlyMeshes.push(catcher);
    }
    this.catcher.parent = root;
    this.catcher.position.set(0, CATCHER_LIFT_M, 0);
    let casters = 0;
    for (const mesh of root.getChildMeshes()) {
      if (mesh.getTotalVertices() > 0) {
        this.generator.addShadowCaster(mesh, false);
        casters += 1;
      }
    }
    console.info(`[shadow] catcher attached under asset-root (${casters} casters)`);
    return true;
  }

  /**
   * Point the shadow light along the sweep's dominant IBL direction.
   * The SH polynomial linear terms (x, y, z), luminance-weighted, point
   * TOWARD the brightest hemisphere; the light direction is the negation,
   * clamped to a minimum downward component so the floor always catches.
   */
  updateFromEnvironment(env: BaseTexture | null): void {
    const poly = env?.sphericalPolynomial ?? null;
    const dir = FALLBACK_DIRECTION.clone();
    let source = "fallback";
    if (poly) {
      const luma = (v: Vector3): number => LUMA.r * v.x + LUMA.g * v.y + LUMA.b * v.z;
      const dominant = new Vector3(luma(poly.x), luma(poly.y), luma(poly.z));
      if (dominant.length() > 1e-4) {
        dir.copyFrom(dominant.normalize().scale(-1));
        if (dir.y > -MIN_DOWN_COMPONENT) {
          dir.y = -MIN_DOWN_COMPONENT;
        }
        source = "IBL polynomial";
      }
    }
    dir.normalize();
    this.light.direction.copyFrom(dir);
    console.info(
      `[shadow] light dir (${dir.x.toFixed(2)}, ${dir.y.toFixed(2)}, ${dir.z.toFixed(2)}) -- ${source}`,
    );
  }

  /** Toggle the catcher (V key): A/B instrument for the G8 groundedness call. */
  toggle(): boolean {
    if (!this.catcher) return false;
    this.enabled = !this.enabled;
    this.catcher.setEnabled(this.enabled);
    console.info(`[shadow] contact shadow ${this.enabled ? "ON" : "OFF"}`);
    return this.enabled;
  }
}

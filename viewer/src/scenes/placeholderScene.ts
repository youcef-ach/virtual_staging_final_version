import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import { Scene } from "@babylonjs/core/scene";
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { DirectionalLight } from "@babylonjs/core/Lights/directionalLight";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { CreateSphere } from "@babylonjs/core/Meshes/Builders/sphereBuilder";
import { CreateGround } from "@babylonjs/core/Meshes/Builders/groundBuilder";
import { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial";

/**
 * Temporary scene proving the render pipeline end-to-end (engine init, PBR,
 * lights, camera controls). Replaced by the cubemap skybox viewer in F4.
 */
export function createPlaceholderScene(engine: AbstractEngine): Scene {
  const scene = new Scene(engine);
  scene.clearColor = new Color4(0.043, 0.051, 0.063, 1);

  const camera = new ArcRotateCamera(
    "camera",
    Math.PI / 3,
    Math.PI / 2.6,
    6,
    new Vector3(0, 0.5, 0),
    scene,
  );
  camera.attachControl(true);
  camera.wheelDeltaPercentage = 0.01;
  camera.lowerRadiusLimit = 2;
  camera.upperRadiusLimit = 20;

  const ambient = new HemisphericLight("ambient", new Vector3(0, 1, 0), scene);
  ambient.intensity = 0.4;

  const sun = new DirectionalLight("sun", new Vector3(-0.5, -1, -0.4), scene);
  sun.intensity = 1.2;

  const sphereMat = new PBRMaterial("sphereMat", scene);
  sphereMat.albedoColor = new Color3(0.8, 0.82, 0.85);
  sphereMat.metallic = 0.1;
  sphereMat.roughness = 0.3;

  const sphere = CreateSphere("sphere", { diameter: 1, segments: 48 }, scene);
  sphere.position.y = 0.5;
  sphere.material = sphereMat;

  const groundMat = new PBRMaterial("groundMat", scene);
  groundMat.albedoColor = new Color3(0.12, 0.13, 0.15);
  groundMat.metallic = 0;
  groundMat.roughness = 0.9;

  const ground = CreateGround("ground", { width: 12, height: 12 }, scene);
  ground.material = groundMat;

  return scene;
}

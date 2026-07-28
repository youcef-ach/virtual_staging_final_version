import { Engine } from "@babylonjs/core/Engines/engine";
import { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";
import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import type { EngineBackend } from "../state/store";

export interface CreatedEngine {
  engine: AbstractEngine;
  backend: EngineBackend;
}

/**
 * Creates the render engine per Locked Decision #1:
 * WebGPU-native when available, automatic WebGL2 fallback otherwise.
 */
export async function createEngine(canvas: HTMLCanvasElement): Promise<CreatedEngine> {
  try {
    if (await WebGPUEngine.IsSupportedAsync) {
      const engine = new WebGPUEngine(canvas, {
        antialias: true,
        powerPreference: "high-performance",
        adaptToDeviceRatio: true,
      });
      await engine.initAsync();
      return { engine, backend: "webgpu" };
    }
  } catch (err) {
    console.warn("[engine] WebGPU initialization failed, falling back to WebGL2:", err);
  }

  const engine = new Engine(
    canvas,
    true,
    { powerPreference: "high-performance", adaptToDeviceRatio: true },
    true,
  );
  return { engine, backend: "webgl2" };
}

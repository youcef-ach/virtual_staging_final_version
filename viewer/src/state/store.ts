import { create } from "zustand";

export type EngineBackend = "webgpu" | "webgl2";

interface ViewerState {
  /** Active render backend, null while the engine is initializing. */
  backend: EngineBackend | null;
  /** Rendered frames per second, sampled once per second. */
  fps: number;
  setBackend: (backend: EngineBackend) => void;
  setFps: (fps: number) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  backend: null,
  fps: 0,
  setBackend: (backend) => set({ backend }),
  setFps: (fps) => set({ fps }),
}));

import { create } from "zustand";

export type EngineBackend = "webgpu" | "webgl2";
export type ViewerStatus = "boot" | "loading" | "ready" | "error";

export interface SweepSummary {
  uuid: string;
  index: number;
}

interface ViewerState {
  /** Active render backend, null while the engine is initializing. */
  backend: EngineBackend | null;
  /** Rendered frames per second, sampled once per second. */
  fps: number;
  status: ViewerStatus;
  errorMessage: string | null;
  sweeps: SweepSummary[];
  activeSweepUuid: string | null;
  /** F4.4 skybox orientation calibration state (shown in the HUD). */
  calibration: string | null;
  setBackend: (backend: EngineBackend) => void;
  setFps: (fps: number) => void;
  setStatus: (status: ViewerStatus) => void;
  setError: (message: string) => void;
  setSweeps: (sweeps: SweepSummary[]) => void;
  setActiveSweep: (uuid: string) => void;
  setCalibration: (label: string) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  backend: null,
  fps: 0,
  status: "boot",
  errorMessage: null,
  sweeps: [],
  activeSweepUuid: null,
  calibration: null,
  setBackend: (backend) => set({ backend }),
  setFps: (fps) => set({ fps }),
  setStatus: (status) => set({ status, errorMessage: status === "error" ? undefined : null }),
  setError: (errorMessage) => set({ status: "error", errorMessage }),
  setSweeps: (sweeps) => set({ sweeps }),
  setActiveSweep: (activeSweepUuid) => set({ activeSweepUuid }),
  setCalibration: (calibration) => set({ calibration }),
}));

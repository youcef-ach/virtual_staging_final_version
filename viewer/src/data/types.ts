/** Pipeline output schemas (poses.json from the pose stage, manifest.json from the panorama stage). */

export interface PoseSweep {
  sweep_uuid: string;
  index: number;
  /** 4x4 row-major pose matrix, runtime Y-up frame (Coordinate Lock Rule). */
  pose_rt: number[][];
}

export interface PosesFile {
  schema_version: number;
  coordinate_frame: string;
  conversion: string;
  sweep_count: number;
  sweeps: PoseSweep[];
}

export interface ManifestSweep {
  sweep_uuid: string;
  index: number;
  /** face name (px/nx/py/ny/pz/nz) -> path relative to the data base URL */
  faces: Record<string, string>;
}

export interface PanoramaManifest {
  schema_version: number;
  sweep_count: number;
  incomplete_sweeps: string[];
  sweeps: ManifestSweep[];
}

/** Joined, viewer-ready tour model. */
export interface SweepData {
  uuid: string;
  index: number;
  /** Sweep position in the runtime Y-up frame (meters). */
  position: readonly [number, number, number];
  /**
   * 3x3 rotation block of pose_rt (row-major, column convention:
   * d_world = R * d_local). The panos are stored in the sweep-LOCAL frame,
   * so the viewer must rotate sampling directions world->local per sweep.
   */
  rotation: readonly (readonly number[])[];
  faces: Record<string, string>;
}

export interface TourData {
  baseUrl: string;
  sweeps: SweepData[];
}

/** pose_qa.json schema (vs_pipeline.stages.pose_qa, schema v4). */

export interface CalibrationEntry {
  render_face: string;
  op: string;
  mean: number;
}

export interface SweepQA {
  sweep_uuid: string;
  index: number;
  /** Chance-corrected edge-overlap, mean over informative faces. */
  score: number | null;
  face_scores: Record<string, number>;
  /** Raw (chance-uncorrected) overlap per face - reference only. */
  raw_face_scores?: Record<string, number>;
  /** Mean orientation margin: locked orientation minus best wrong orientation. */
  margin?: number | null;
  face_margins?: Record<string, number>;
  /** "low_score" | "low_margin" | "too_few_informative_faces". */
  flag_reasons?: string[];
  uninformative_faces: string[];
  /** Per-face fraction of pixels excluded as glass/mirror (schema v5, seg-dir runs). */
  face_excluded?: Record<string, number>;
  flagged: boolean;
  spot_check: boolean;
}

export interface PoseQAReport {
  schema_version: number;
  /** Gross-failure floor for the normalized sweep score. */
  threshold: number;
  /** Subtle-drift floor for the mean orientation margin. */
  margin_threshold?: number;
  spot_check_rate: number;
  calibration: Record<string, CalibrationEntry>;
  /** Segmentation dir used for glass/mirror exclusion (schema v5), or null. */
  seg_dir?: string | null;
  /** Sweeps that had at least one face with excluded pixels (schema v5). */
  seg_excluded_sweep_count?: number;
  sweep_count: number;
  flagged_count: number;
  spot_check_count: number;
  sweeps: SweepQA[];
}

export interface CatalogItem {
  id: string;
  sku: string;
  manufacturer: string;
  product_name: string;
  category: string;
  price: number | null;
  currency: string | null;
  purchase_url: string | null;
  materials?: Record<string, unknown> | null;
  dimensions_mm?: { x: number; y: number; z: number } | null;
  weight_kg?: number | null;
  glb_url: string | null;
  thumbnail_url: string | null;
  bounding_box?: { min: number[]; max: number[] } | null;
  polygon_count?: number | null;
  lod_urls?: Record<string, string> | null;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Property & Scan Version types
// ---------------------------------------------------------------------------

export interface Property {
  id: string;
  organization_id: string;
  name: string;
  address: string | null;
  created_at: string;
}

export interface Completeness {
  has_e57: boolean;
  has_matterpak: boolean;
  has_cubemaps: boolean;
  complete: boolean;
}

export interface ScanVersion {
  id: string;
  property_id: string;
  version: number;
  status: string;
  is_authoritative: boolean;
  raw_assets: Record<string, unknown>;
  completeness: Completeness;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Pipeline orchestration types
// ---------------------------------------------------------------------------

export type StageStatusType =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "skipped"
  | "cancelled";

export type PipelineStatusType =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface StageExecution {
  id: string;
  run_id: string;
  stage_number: number;
  stage_name: string;
  status: StageStatusType;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  log_path: string | null;
}

export interface PipelineRun {
  id: string;
  scan_version_id: string;
  status: PipelineStatusType;
  created_at: string;
  updated_at: string;
}

export interface PipelineRunDetail extends PipelineRun {
  stages: StageExecution[];
}

/** DAG edge for rendering the stage dependency graph. */
export interface DagEdge {
  from: number;
  to: number;
}

/** Pre-defined DAG edges matching the backend dag.py. */
export const DAG_EDGES: DagEdge[] = [
  { from: 1, to: 2 },
  { from: 1, to: 3 },
  { from: 1, to: 8 },
  { from: 3, to: 4 },
  { from: 3, to: 5 },
  { from: 5, to: 6 },
  { from: 4, to: 6 },
  { from: 6, to: 7 },
  { from: 5, to: 9 },
  { from: 6, to: 9 },
  { from: 2, to: 10 },
  { from: 4, to: 10 },
  { from: 7, to: 10 },
  { from: 8, to: 10 },
  { from: 9, to: 10 },
];

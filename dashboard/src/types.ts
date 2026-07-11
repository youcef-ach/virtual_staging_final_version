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
  sweep_count: number;
  flagged_count: number;
  spot_check_count: number;
  sweeps: SweepQA[];
}

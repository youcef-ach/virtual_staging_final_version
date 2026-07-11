/** pose_qa.json schema (vs_pipeline.stages.pose_qa, schema v2). */

export interface CalibrationEntry {
  render_face: string;
  op: string;
  mean: number;
}

export interface SweepQA {
  sweep_uuid: string;
  index: number;
  score: number | null;
  face_scores: Record<string, number>;
  uninformative_faces: string[];
  flagged: boolean;
  spot_check: boolean;
}

export interface PoseQAReport {
  schema_version: number;
  threshold: number;
  spot_check_rate: number;
  calibration: Record<string, CalibrationEntry>;
  sweep_count: number;
  flagged_count: number;
  spot_check_count: number;
  sweeps: SweepQA[];
}

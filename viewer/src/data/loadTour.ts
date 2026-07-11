import type { PanoramaManifest, PosesFile, SweepData, TourData } from "./types";

/**
 * Loads and joins the pipeline outputs (manifest.json + poses.json) into a
 * viewer-ready tour model, keyed by sweep UUID.
 */
export async function loadTour(baseUrl: string): Promise<TourData> {
  const [manifestRes, posesRes] = await Promise.all([
    fetch(`${baseUrl}/manifest.json`),
    fetch(`${baseUrl}/poses.json`),
  ]);
  if (!manifestRes.ok) {
    throw new Error(
      `manifest.json not found under ${baseUrl}/ (HTTP ${manifestRes.status}). ` +
        "Copy the panorama-stage output into viewer/public/data/ (see docs/DEVELOPMENT.md).",
    );
  }
  if (!posesRes.ok) {
    throw new Error(
      `poses.json not found under ${baseUrl}/ (HTTP ${posesRes.status}). ` +
        "Copy the pose-stage output into viewer/public/data/ (see docs/DEVELOPMENT.md).",
    );
  }

  const manifest = (await manifestRes.json()) as PanoramaManifest;
  const poses = (await posesRes.json()) as PosesFile;

  if (poses.coordinate_frame !== "runtime_yup") {
    throw new Error(`poses.json is not in the runtime Y-up frame: ${poses.coordinate_frame}`);
  }

  const poseByUuid = new Map(poses.sweeps.map((s) => [s.sweep_uuid, s]));
  const sweeps: SweepData[] = [];
  for (const entry of manifest.sweeps) {
    const pose = poseByUuid.get(entry.sweep_uuid);
    if (!pose || Object.keys(entry.faces).length !== 6) continue;
    sweeps.push({
      uuid: entry.sweep_uuid,
      index: entry.index,
      position: [pose.pose_rt[0][3], pose.pose_rt[1][3], pose.pose_rt[2][3]],
      faces: entry.faces,
    });
  }
  sweeps.sort((a, b) => a.index - b.index);

  if (sweeps.length === 0) {
    throw new Error("no complete sweeps with matching poses (manifest/poses UUID mismatch?)");
  }
  return { baseUrl, sweeps };
}

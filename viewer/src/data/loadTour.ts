import type { NavGraph, PanoramaManifest, PosesFile, SweepData, TourData } from "./types";

/**
 * Loads and joins the pipeline outputs (manifest.json + poses.json) into a
 * viewer-ready tour model, keyed by sweep UUID. Also loads the T1 navigation
 * graph (navigation_graph.json) when available.
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
      rotation: [
        [pose.pose_rt[0][0], pose.pose_rt[0][1], pose.pose_rt[0][2]],
        [pose.pose_rt[1][0], pose.pose_rt[1][1], pose.pose_rt[1][2]],
        [pose.pose_rt[2][0], pose.pose_rt[2][1], pose.pose_rt[2][2]],
      ],
      faces: entry.faces,
    });
  }
  sweeps.sort((a, b) => a.index - b.index);

  if (sweeps.length === 0) {
    throw new Error("no complete sweeps with matching poses (manifest/poses UUID mismatch?)");
  }

  // T1: navigation graph (optional — tour works without it).
  let nav: NavGraph | null = null;
  try {
    const navRes = await fetch(`${baseUrl}/navigation_graph.json`);
    if (navRes.ok) {
      nav = (await navRes.json()) as NavGraph;
      console.info(
        `[nav] graph loaded: ${nav.node_count} nodes, ${nav.edge_count} edges ` +
          `(${nav.portal_count} portal), ${nav.connected_components} component(s)`,
      );
    }
  } catch {
    // navigation_graph.json missing — the tour keeps working.
  }

  return { baseUrl, sweeps, nav };
}

/**
 * T1: build a neighbor lookup from the navigation graph.
 * Returns a Map: sweepUuid -> sorted list of neighbor UUIDs (nearest first).
 */
export function buildNeighborMap(nav: NavGraph): Map<string, string[]> {
  const adj = new Map<string, { uuid: string; dist: number }[]>();
  for (const edge of nav.edges) {
    if (!adj.has(edge.from)) adj.set(edge.from, []);
    if (!adj.has(edge.to)) adj.set(edge.to, []);
    adj.get(edge.from)!.push({ uuid: edge.to, dist: edge.distance_m });
    adj.get(edge.to)!.push({ uuid: edge.from, dist: edge.distance_m });
  }
  const result = new Map<string, string[]>();
  for (const [uuid, neighbors] of adj) {
    neighbors.sort((a, b) => a.dist - b.dist);
    result.set(
      uuid,
      neighbors.map((n) => n.uuid),
    );
  }
  return result;
}


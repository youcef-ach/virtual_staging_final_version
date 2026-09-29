/**
 * Staging Project Serialization & State Persistence
 *
 * Provides localStorage persistence, project export, and share-link
 * URL parsing for virtual staging state.
 */

export interface StagingAssetDTO {
  /** [x, y, z] in meters on proxy floor (Y-up frame) */
  position: [number, number, number];
  /** Y-axis rotation in radians */
  rotationY: number;
}

export interface StagingProjectDTO {
  /** Schema version identifier for project serialization. */
  version?: number;
  activeSweepUuid?: string | null;
  asset?: StagingAssetDTO | null;
}

/** Local storage key for auto-saving project state. */
const STORAGE_KEY = "vs_staging_project";

/** Default schema version for newly serialized projects. */
const CURRENT_SCHEMA_VERSION = 1;

/**
 * Persists the current staging project state to localStorage.
 * Auto-saves on sweep change, double-click placement, and drag-end.
 *
 * Defensively catches and ignores QuotaExceededError or security restrictions.
 */
export function saveProject(
  activeSweepUuid: string | null,
  position: { x: number; y: number; z: number } | null,
  rotationY: number | null,
): void {
  if (typeof window === "undefined" || !window.localStorage) {
    return;
  }

  try {
    let asset: StagingAssetDTO | null = null;
    if (
      position !== null &&
      Number.isFinite(position.x) &&
      Number.isFinite(position.y) &&
      Number.isFinite(position.z) &&
      rotationY !== null &&
      Number.isFinite(rotationY)
    ) {
      asset = {
        position: [position.x, position.y, position.z],
        rotationY,
      };
    }

    const dto: StagingProjectDTO = {
      version: CURRENT_SCHEMA_VERSION,
      activeSweepUuid: activeSweepUuid ?? null,
      asset,
    };

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(dto));
  } catch {
    // QuotaExceededError, SecurityError (sandboxed iframe), or storage disabled
  }
}

/**
 * Loads and validates the staging project state from localStorage.
 * Returns null if no data is found or if the data fails schema validation.
 */
export function loadProject(): StagingProjectDTO | null {
  if (typeof window === "undefined" || !window.localStorage) {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;

    const parsed: unknown = JSON.parse(raw);
    return validateProjectDTO(parsed);
  } catch {
    return null;
  }
}

/**
 * Triggers a browser download of the staging project as a JSON file.
 * Key shortcut 's' / 'S'.
 */
export function exportProject(
  activeSweepUuid: string | null,
  position: { x: number; y: number; z: number } | null,
  rotationY: number | null,
): void {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return;
  }

  let asset: StagingAssetDTO | null = null;
  if (
    position !== null &&
    Number.isFinite(position.x) &&
    Number.isFinite(position.y) &&
    Number.isFinite(position.z) &&
    rotationY !== null &&
    Number.isFinite(rotationY)
  ) {
    asset = {
      position: [position.x, position.y, position.z],
      rotationY,
    };
  }

  const dto: StagingProjectDTO = {
    version: CURRENT_SCHEMA_VERSION,
    activeSweepUuid: activeSweepUuid ?? null,
    asset,
  };

  try {
    const jsonStr = JSON.stringify(dto, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const filename = activeSweepUuid
      ? `staging_${activeSweepUuid.slice(0, 8)}.json`
      : "staging_project.json";
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch {
    // Non-fatal if export fails (e.g. headless or disabled download)
  }
}

/**
 * Parses staging state from URL hash parameters.
 *
 * Supported formats:
 *   #sweep=<uuid>&x=<num>&y=<num>&z=<num>&rot=<num>
 *   #?sweep=<uuid>&x=<num>&y=<num>&z=<num>&rot=<num>
 *
 * Returns null if no valid hash params are found or if coordinate parsing fails.
 */
export function parseShareLink(hash?: string): StagingProjectDTO | null {
  const rawHash =
    hash !== undefined
      ? hash
      : typeof window !== "undefined"
        ? window.location.hash
        : "";

  if (!rawHash) return null;

  // Strip leading '#' and optional '?'
  let cleanHash = rawHash.startsWith("#") ? rawHash.slice(1) : rawHash;
  if (cleanHash.startsWith("?")) {
    cleanHash = cleanHash.slice(1);
  }
  if (!cleanHash) return null;

  try {
    const params = new URLSearchParams(cleanHash);
    const sweepUuid = params.get("sweep");

    let asset: StagingAssetDTO | null = null;
    const xStr = params.get("x");
    const yStr = params.get("y");
    const zStr = params.get("z");
    const rotStr = params.get("rot");

    if (
      xStr !== null &&
      yStr !== null &&
      zStr !== null &&
      xStr.trim() !== "" &&
      yStr.trim() !== "" &&
      zStr.trim() !== "" &&
      (rotStr === null || rotStr.trim() !== "")
    ) {
      const x = Number(xStr);
      const y = Number(yStr);
      const z = Number(zStr);
      const rotY = rotStr !== null ? Number(rotStr) : 0;

      if (
        Number.isFinite(x) &&
        Number.isFinite(y) &&
        Number.isFinite(z) &&
        Number.isFinite(rotY)
      ) {
        asset = {
          position: [x, y, z],
          rotationY: rotY,
        };
      }
    }

    if (!sweepUuid && !asset) {
      return null;
    }

    return {
      version: CURRENT_SCHEMA_VERSION,
      activeSweepUuid: sweepUuid || null,
      asset,
    };
  } catch {
    return null;
  }
}

/**
 * Validates untrusted parsed JSON as a valid StagingProjectDTO.
 */
function validateProjectDTO(obj: unknown): StagingProjectDTO | null {
  if (!obj || typeof obj !== "object") {
    return null;
  }

  const record = obj as Record<string, unknown>;
  const activeSweepUuid =
    typeof record.activeSweepUuid === "string" ? record.activeSweepUuid : null;

  let asset: StagingAssetDTO | null = null;
  if (record.asset && typeof record.asset === "object") {
    const assetRecord = record.asset as Record<string, unknown>;
    const pos = assetRecord.position;
    const rot = assetRecord.rotationY;

    if (
      Array.isArray(pos) &&
      pos.length === 3 &&
      typeof pos[0] === "number" &&
      Number.isFinite(pos[0]) &&
      typeof pos[1] === "number" &&
      Number.isFinite(pos[1]) &&
      typeof pos[2] === "number" &&
      Number.isFinite(pos[2]) &&
      typeof rot === "number" &&
      Number.isFinite(rot)
    ) {
      asset = {
        position: [pos[0], pos[1], pos[2]],
        rotationY: rot,
      };
    }
  }

  // If both are missing/invalid, this isn't a valid project state
  if (!activeSweepUuid && !asset) {
    return null;
  }

  return {
    version: typeof record.version === "number" ? record.version : CURRENT_SCHEMA_VERSION,
    activeSweepUuid,
    asset,
  };
}

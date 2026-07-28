/**
 * T2 -- LRU cubemap texture cache for neighbor prefetch.
 *
 * VRAM discipline: at most MAX_ENTRIES cubemaps are kept resident. When the
 * budget is exceeded the least-recently-used entry is disposed. The active
 * sweep's texture is never evicted (it is managed by TourViewer, not the cache).
 *
 * Prefetch is fire-and-forget: it starts loading neighbor cubemaps in the
 * background without blocking the current frame. A stale prefetch (the user
 * jumped again before it completed) is still cached -- it's likely a future
 * neighbor too.
 */
import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture";
import type { Scene } from "@babylonjs/core/scene";

/** Babylon CubeTexture file order: +x, +y, +z, -x, -y, -z. */
const FACE_ORDER = ["px", "py", "pz", "nx", "ny", "nz"] as const;

const MAX_ENTRIES = 4;

interface CacheEntry {
  texture: CubeTexture;
  lastUsed: number;
}

export class TextureCache {
  private readonly scene: Scene;
  private readonly cache = new Map<string, CacheEntry>();
  private readonly loading = new Set<string>();

  constructor(scene: Scene) {
    this.scene = scene;
  }

  /**
   * Get a cubemap for the given sweep. Returns a cached texture immediately if
   * available, otherwise loads it (with LRU eviction if over budget).
   */
  async get(
    sweepUuid: string,
    faceFiles: string[],
  ): Promise<CubeTexture> {
    const existing = this.cache.get(sweepUuid);
    if (existing) {
      existing.lastUsed = performance.now();
      return existing.texture;
    }
    return this.load(sweepUuid, faceFiles);
  }

  /** Check whether a sweep's cubemap is already cached. */
  has(sweepUuid: string): boolean {
    return this.cache.has(sweepUuid);
  }

  /**
   * Prefetch cubemaps for the given sweeps (non-blocking, fire-and-forget).
   * Each sweep needs a baseUrl and faces record to build file URLs.
   */
  prefetch(
    sweeps: { uuid: string; baseUrl: string; faces: Record<string, string> }[],
  ): void {
    for (const sweep of sweeps) {
      if (this.cache.has(sweep.uuid) || this.loading.has(sweep.uuid)) continue;
      const files = FACE_ORDER.map((f) => `${sweep.baseUrl}/${sweep.faces[f]}`);
      // Fire and forget -- errors are swallowed (prefetch is best-effort).
      void this.load(sweep.uuid, files).catch(() => {
        /* prefetch miss -- not fatal */
      });
    }
  }

  /**
   * Remove a sweep from the cache (e.g. when its texture is handed off to the
   * skybox material and managed externally). Does NOT dispose the texture.
   */
  detach(sweepUuid: string): void {
    this.cache.delete(sweepUuid);
  }

  /** Dispose all cached textures. */
  dispose(): void {
    for (const entry of this.cache.values()) {
      entry.texture.dispose();
    }
    this.cache.clear();
    this.loading.clear();
  }

  private async load(
    sweepUuid: string,
    files: string[],
  ): Promise<CubeTexture> {
    this.loading.add(sweepUuid);
    try {
      const texture = await this.loadCubeTexture(`cache:${sweepUuid}`, files);
      this.evictIfNeeded();
      this.cache.set(sweepUuid, { texture, lastUsed: performance.now() });
      return texture;
    } finally {
      this.loading.delete(sweepUuid);
    }
  }

  private evictIfNeeded(): void {
    while (this.cache.size >= MAX_ENTRIES) {
      let oldestKey: string | null = null;
      let oldestTime = Infinity;
      for (const [key, entry] of this.cache) {
        if (entry.lastUsed < oldestTime) {
          oldestTime = entry.lastUsed;
          oldestKey = key;
        }
      }
      if (oldestKey) {
        const entry = this.cache.get(oldestKey);
        entry?.texture.dispose();
        this.cache.delete(oldestKey);
      } else {
        break;
      }
    }
  }

  private loadCubeTexture(
    cacheKey: string,
    files: string[],
  ): Promise<CubeTexture> {
    return new Promise((resolve, reject) => {
      const texture = new CubeTexture(
        cacheKey,
        this.scene,
        null,
        false,
        files,
        () => resolve(texture),
        (message) => reject(new Error(message ?? "cubemap load failed")),
      );
    });
  }
}

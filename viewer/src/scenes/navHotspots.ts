/**
 * T1 -- navigation hotspot markers: translucent floor discs at neighbor sweep
 * positions that the operator can click to jump.  Updated every sweep change
 * from the navigation graph's neighbor list.
 *
 * Design:
 * - Thin cylinder (disc) at each neighbor's floor position, y = sweep camera Y
 *   minus TRIPOD_HEIGHT_M (same invariant as staging.ts placement).
 * - GROUP_ASSET so they are depth-tested against the proxy (a marker behind a
 *   real wall is occluded like an asset).
 * - Two colors: same-room neighbors (white-ish) vs portal neighbors (blue).
 * - Click detection via scene.pick on the marker meshes.
 */
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import type { Scene } from "@babylonjs/core/scene";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import type { NavGraph } from "../data/types";
import { GROUP_ASSET } from "./staging";

const MARKER_RADIUS = 0.18;
const MARKER_HEIGHT = 0.02;
/** Visual y-offset above the floor to avoid z-fighting with the catcher. */
const MARKER_LIFT = 0.01;
/** Tripod height from staging.ts (camera Y - floor Y ≈ 1.4 m). */
const TRIPOD_HEIGHT_M = 1.4;
/** Maximum neighbor markers to show (avoid visual clutter). */
const MAX_MARKERS = 8;

export class NavHotspots {
  private readonly scene: Scene;
  private readonly matIntraRoom: StandardMaterial;
  private readonly matPortal: StandardMaterial;
  private readonly markerPool: Mesh[] = [];
  /** UUID of the sweep each active marker points to (index-matched). */
  private activeTargets: string[] = [];
  private neighbors: Map<string, string[]> = new Map();
  private portalPairs: Set<string> = new Set();
  private nodePositions: Map<string, [number, number, number]> = new Map();
  private nodeRooms: Map<string, string> = new Map();

  constructor(scene: Scene) {
    this.scene = scene;

    // Intra-room marker: translucent white disc
    this.matIntraRoom = new StandardMaterial("nav-marker-intra", scene);
    this.matIntraRoom.diffuseColor = new Color3(0.9, 0.95, 1.0);
    this.matIntraRoom.emissiveColor = new Color3(0.4, 0.45, 0.5);
    this.matIntraRoom.alpha = 0.55;
    this.matIntraRoom.disableLighting = true;
    this.matIntraRoom.backFaceCulling = false;

    // Portal marker: translucent blue disc
    this.matPortal = new StandardMaterial("nav-marker-portal", scene);
    this.matPortal.diffuseColor = new Color3(0.3, 0.5, 1.0);
    this.matPortal.emissiveColor = new Color3(0.15, 0.3, 0.7);
    this.matPortal.alpha = 0.65;
    this.matPortal.disableLighting = true;
    this.matPortal.backFaceCulling = false;
  }

  /** Parse the navigation graph and build lookup tables. */
  loadGraph(nav: NavGraph): void {
    // Build adjacency
    const adj = new Map<string, { uuid: string; dist: number }[]>();
    for (const edge of nav.edges) {
      if (!adj.has(edge.from)) adj.set(edge.from, []);
      if (!adj.has(edge.to)) adj.set(edge.to, []);
      adj.get(edge.from)!.push({ uuid: edge.to, dist: edge.distance_m });
      adj.get(edge.to)!.push({ uuid: edge.from, dist: edge.distance_m });
    }
    this.neighbors = new Map<string, string[]>();
    for (const [uuid, nbrs] of adj) {
      nbrs.sort((a, b) => a.dist - b.dist);
      this.neighbors.set(uuid, nbrs.map((n) => n.uuid));
    }

    // Portal pairs (for coloring)
    for (const edge of nav.edges) {
      if (edge.portal) {
        this.portalPairs.add(`${edge.from}:${edge.to}`);
        this.portalPairs.add(`${edge.to}:${edge.from}`);
      }
    }

    // Node positions and rooms
    for (const node of nav.nodes) {
      this.nodePositions.set(node.uuid, node.position);
      if (node.room) this.nodeRooms.set(node.uuid, node.room);
    }
  }

  /**
   * Show markers for the current sweep's neighbors.
   * cameraY is used to compute the floor level for marker placement.
   */
  showForSweep(sweepUuid: string, cameraY: number): void {
    this.hideAll();
    const nbrs = this.neighbors.get(sweepUuid);
    if (!nbrs) return;

    const floorY = cameraY - TRIPOD_HEIGHT_M;
    const toShow = nbrs.slice(0, MAX_MARKERS);
    this.activeTargets = [];

    for (let i = 0; i < toShow.length; i++) {
      const targetUuid = toShow[i];
      const pos = this.nodePositions.get(targetUuid);
      if (!pos) continue;

      const mesh = this.getOrCreateMarker(i);
      mesh.position.set(pos[0], floorY + MARKER_LIFT, pos[2]);
      const isPortal = this.portalPairs.has(`${sweepUuid}:${targetUuid}`);
      mesh.material = isPortal ? this.matPortal : this.matIntraRoom;
      mesh.setEnabled(true);
      this.activeTargets.push(targetUuid);
    }
  }

  /** Pick a marker at screen coords. Returns the target sweep UUID, or null. */
  pick(x: number, y: number): string | null {
    const hit = this.scene.pick(x, y, (m) => this.markerPool.includes(m as Mesh));
    if (!hit?.hit || !hit.pickedMesh) return null;
    const idx = this.markerPool.indexOf(hit.pickedMesh as Mesh);
    if (idx < 0 || idx >= this.activeTargets.length) return null;
    return this.activeTargets[idx];
  }

  /**
   * Get the neighbor list for the given sweep (sorted by distance, nearest first).
   * Returns null if no graph or no neighbors for this sweep.
   */
  getNeighbors(sweepUuid: string): string[] | null {
    return this.neighbors.get(sweepUuid) ?? null;
  }

  private hideAll(): void {
    for (const mesh of this.markerPool) {
      mesh.setEnabled(false);
    }
    this.activeTargets = [];
  }

  private getOrCreateMarker(index: number): Mesh {
    if (index < this.markerPool.length) {
      return this.markerPool[index];
    }
    const mesh = MeshBuilder.CreateCylinder(
      `nav-marker-${index}`,
      { diameter: MARKER_RADIUS * 2, height: MARKER_HEIGHT, tessellation: 24 },
      this.scene,
    );
    mesh.renderingGroupId = GROUP_ASSET;
    mesh.isPickable = true;
    mesh.setEnabled(false);
    this.markerPool.push(mesh);
    return mesh;
  }
}

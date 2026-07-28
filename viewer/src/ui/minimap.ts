/**
 * T4 -- Minimap: floor plan overlay with position indicator and hotspots.
 *
 * Renders room polygons, sweep dots, and nav edges as an SVG overlay in the
 * bottom-left corner. The active sweep is highlighted and a storey selector
 * lets the user switch floors.
 *
 * The minimap loads ``floorplan.json`` (exported by ``vs_pipeline.stages.floorplan``).
 * Clicking a sweep dot jumps to that sweep.
 *
 * Toggle visibility with the M key.
 */

export interface FloorplanStorey {
  storey: number;
  rooms: { room_id: string; polygon: [number, number][] }[];
  sweeps: { uuid: string; index: number; x: number; z: number }[];
  edges: { from: string; to: string; portal: boolean }[];
  bounds: { min_x: number; max_x: number; min_z: number; max_z: number };
}

export interface FloorplanData {
  schema_version: number;
  storey_count: number;
  storeys: FloorplanStorey[];
}

/** Load floorplan.json. Returns null if not found. */
export async function loadFloorplan(baseUrl: string): Promise<FloorplanData | null> {
  try {
    const res = await fetch(`${baseUrl}/floorplan.json`);
    if (!res.ok) return null;
    const data = (await res.json()) as FloorplanData;
    console.info(`[minimap] floorplan loaded: ${data.storey_count} storeys`);
    return data;
  } catch {
    return null;
  }
}

const MAP_WIDTH = 220;
const MAP_HEIGHT = 200;
const PADDING = 12;

/**
 * Create the minimap DOM element and return a controller object.
 * The minimap is appended to the given container.
 */
export function createMinimap(
  container: HTMLElement,
  floorplan: FloorplanData,
  onSweepClick: (uuid: string) => void,
): MinimapController {
  // Root container.
  const root = document.createElement("div");
  root.id = "minimap";
  root.style.cssText = `
    position: absolute; bottom: 16px; left: 16px;
    width: ${MAP_WIDTH}px; height: ${MAP_HEIGHT + 30}px;
    background: rgba(0,0,0,0.75); border-radius: 10px;
    pointer-events: auto; z-index: 100; padding: 8px;
    backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.1);
    font-family: 'Inter', sans-serif; box-sizing: border-box;
    transition: opacity 0.2s;
  `;

  // Storey tabs.
  const tabs = document.createElement("div");
  tabs.style.cssText = `
    display: flex; gap: 4px; margin-bottom: 4px;
  `;
  root.appendChild(tabs);

  // SVG canvas.
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", String(MAP_WIDTH - 16));
  svg.setAttribute("height", String(MAP_HEIGHT - 8));
  svg.style.cssText = "display: block;";
  root.appendChild(svg);

  container.appendChild(root);

  let activeStorey = floorplan.storeys.length > 1 ? 1 : 0; // default to storey 1
  let activeUuid: string | null = null;

  // Track sweep positions for the active storey (for click detection).
  const sweepElements = new Map<string, SVGCircleElement>();

  function getTransform(storey: FloorplanStorey) {
    const b = storey.bounds;
    const spanX = b.max_x - b.min_x || 1;
    const spanZ = b.max_z - b.min_z || 1;
    const svgW = MAP_WIDTH - 16 - PADDING * 2;
    const svgH = MAP_HEIGHT - 8 - PADDING * 2;
    const scale = Math.min(svgW / spanX, svgH / spanZ);
    const offsetX = PADDING + (svgW - spanX * scale) / 2;
    const offsetZ = PADDING + (svgH - spanZ * scale) / 2;
    return {
      toSvg: (wx: number, wz: number): [number, number] => [
        offsetX + (wx - b.min_x) * scale,
        offsetZ + (wz - b.min_z) * scale,
      ],
    };
  }

  function renderStorey(storeyIdx: number) {
    activeStorey = storeyIdx;
    sweepElements.clear();
    // Clear SVG.
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const storey = floorplan.storeys[storeyIdx];
    if (!storey) return;
    const { toSvg } = getTransform(storey);

    // Room polygons.
    for (const room of storey.rooms) {
      const pts = room.polygon.map(([x, z]) => toSvg(x, z).join(",")).join(" ");
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("points", pts);
      poly.setAttribute("fill", "rgba(60,130,200,0.15)");
      poly.setAttribute("stroke", "rgba(100,170,255,0.5)");
      poly.setAttribute("stroke-width", "1");
      svg.appendChild(poly);
    }

    // Nav edges.
    const sweepPos = new Map<string, [number, number]>();
    for (const sw of storey.sweeps) {
      sweepPos.set(sw.uuid, toSvg(sw.x, sw.z));
    }
    for (const edge of storey.edges) {
      const p1 = sweepPos.get(edge.from);
      const p2 = sweepPos.get(edge.to);
      if (!p1 || !p2) continue;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(p1[0]));
      line.setAttribute("y1", String(p1[1]));
      line.setAttribute("x2", String(p2[0]));
      line.setAttribute("y2", String(p2[1]));
      line.setAttribute("stroke", edge.portal ? "rgba(100,180,255,0.4)" : "rgba(255,255,255,0.1)");
      line.setAttribute("stroke-width", edge.portal ? "1.5" : "0.5");
      svg.appendChild(line);
    }

    // Sweep dots.
    for (const sw of storey.sweeps) {
      const [cx, cy] = toSvg(sw.x, sw.z);
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", String(cx));
      circle.setAttribute("cy", String(cy));
      circle.setAttribute("r", sw.uuid === activeUuid ? "5" : "3");
      circle.setAttribute(
        "fill",
        sw.uuid === activeUuid ? "#ff6622" : "rgba(200,220,255,0.6)",
      );
      circle.setAttribute("stroke", sw.uuid === activeUuid ? "#fff" : "none");
      circle.setAttribute("stroke-width", "1.5");
      circle.style.cursor = "pointer";
      circle.addEventListener("click", (e) => {
        e.stopPropagation();
        onSweepClick(sw.uuid);
      });
      svg.appendChild(circle);
      sweepElements.set(sw.uuid, circle);
    }
  }

  function renderTabs() {
    tabs.innerHTML = "";
    for (let i = 0; i < floorplan.storeys.length; i++) {
      const s = floorplan.storeys[i];
      const btn = document.createElement("button");
      btn.textContent = `F${s.storey}`;
      btn.style.cssText = `
        padding: 2px 8px; font-size: 11px; border-radius: 4px; cursor: pointer;
        border: 1px solid rgba(255,255,255,0.2);
        background: ${i === activeStorey ? "rgba(100,160,255,0.5)" : "rgba(255,255,255,0.08)"};
        color: ${i === activeStorey ? "#fff" : "rgba(255,255,255,0.6)"};
      `;
      btn.addEventListener("click", () => {
        renderStorey(i);
        renderTabs();
      });
      tabs.appendChild(btn);
    }
  }

  renderTabs();
  renderStorey(activeStorey);

  return {
    setActiveSweep(uuid: string) {
      activeUuid = uuid;
      // Find which storey this sweep is on and switch if needed.
      for (let i = 0; i < floorplan.storeys.length; i++) {
        if (floorplan.storeys[i].sweeps.some((s) => s.uuid === uuid)) {
          if (i !== activeStorey) {
            renderStorey(i);
            renderTabs();
          } else {
            // Just update the highlight.
            renderStorey(i);
          }
          return;
        }
      }
    },
    setVisible(visible: boolean) {
      root.style.display = visible ? "block" : "none";
    },
    dispose() {
      root.remove();
    },
  };
}

export interface MinimapController {
  setActiveSweep(uuid: string): void;
  setVisible(visible: boolean): void;
  dispose(): void;
}

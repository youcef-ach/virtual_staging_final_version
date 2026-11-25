import { useEffect, useRef, useState } from "react";
import { createEngine } from "./engine/createEngine";
import { loadTour } from "./data/loadTour";
import { TourViewer } from "./scenes/tourViewer";
import { StagingLayer } from "./scenes/staging";
import { IblLighting } from "./scenes/lighting";
import { ContactShadow } from "./scenes/shadow";
import { NavHotspots } from "./scenes/navHotspots";
import { SweepTransition } from "./scenes/transition";
import { TextureCache } from "./scenes/textureCache";
import { DollhouseView } from "./scenes/dollhouse";
import { loadFloorplan, createMinimap } from "./ui/minimap";
import type { MinimapController } from "./ui/minimap";
import { MeasureTool } from "./scenes/measure";
import {
  saveProject,
  loadProject,
  exportProject,
  parseShareLink,
} from "./data/stagingProject";
import { useViewerStore } from "./state/store";
import { ModelOutputs } from "./ui/ModelOutputs";

const SHOWCASE_BUILD = import.meta.env.MODE === "showcase";
const DEMO = SHOWCASE_BUILD || new URLSearchParams(window.location.search).get("demo") === "1";
const DATA_BASE_URL = `${import.meta.env.BASE_URL}${DEMO ? "demo" : "data"}`;

/** Placement nudge steps (meters). */
const NUDGE_M = 0.05;
const NUDGE_BIG_M = 0.25;
const DEMO_CHAIR_POSITION: [number, number, number] = [1.22, 0, 1.65];
const DEMO_CHAIR_ROTATION = Math.PI / 10;

interface SceneControls {
  toggleDollhouse(): void;
  setFurnitureVisible(visible: boolean): void;
  resetFurniture(): void;
  nudge(x: number, z: number): void;
  rotate(): void;
  openSweep(uuid: string): void;
}

const sweepLabel = (uuid: string, index: number) =>
  DEMO ? (uuid === "demo-entry" ? "Room entry" : "By the window") : `Viewpoint ${String(index).padStart(3, "0")}`;

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<TourViewer | null>(null);
  const navRef = useRef<NavHotspots | null>(null);
  const minimapRef = useRef<MinimapController | null>(null);
  const controls = useRef<SceneControls | null>(null);
  const [panel, setPanel] = useState<"scene" | "outputs">("scene");
  const [modelAvailable, setModelAvailable] = useState(false);
  const [furnitureAvailable, setFurnitureAvailable] = useState(false);
  const [furnitureVisible, setFurnitureVisible] = useState(true);
  const [dollhouseActive, setDollhouseActive] = useState(false);
  const [navigationError, setNavigationError] = useState<string | null>(null);
  const [pendingSweep, setPendingSweep] = useState<string | null>(null);
  const backend = useViewerStore((s) => s.backend);
  const fps = useViewerStore((s) => s.fps);
  const status = useViewerStore((s) => s.status);
  const errorMessage = useViewerStore((s) => s.errorMessage);
  const sweeps = useViewerStore((s) => s.sweeps);
  const activeSweepUuid = useViewerStore((s) => s.activeSweepUuid);
  const proxyState = useViewerStore((s) => s.proxyState);
  const assetState = useViewerStore((s) => s.assetState);
  const xray = useViewerStore((s) => s.xray);
  const envState = useViewerStore((s) => s.envState);
  const pickInfo = useViewerStore((s) => s.pickInfo);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;
    let cleanup: (() => void) | undefined;
    let changingSweep = false;

    void (async () => {
      const store = useViewerStore.getState();
      try {
        store.setStatus("loading");
        const tour = await loadTour(DATA_BASE_URL);
        if (disposed) return;
        const { engine, backend } = await createEngine(canvas);
        if (disposed) {
          engine.dispose();
          return;
        }
        store.setBackend(backend);
        store.setSweeps(tour.sweeps.map((s) => ({ uuid: s.uuid, index: s.index })));

        const viewer = new TourViewer(engine, tour);
        viewerRef.current = viewer;
        const savedProject = DEMO ? null : parseShareLink() ?? loadProject();

        // G7: Layer 2 (proxy occluders) + Layer 3 (staged asset). Missing
        // files degrade gracefully -- the tour keeps working either way.
        const staging = new StagingLayer(viewer.scene);
        // G7.2: per-sweep IBL (env swap + chrome-sphere probe). Sweeps without
        // an exported .env fall back to the staging stand-in lights.
        const lighting = new IblLighting(viewer.scene);
        // G7.3: contact shadow -- catcher follows the asset; light direction
        // derives from the active sweep's IBL polynomial.
        const shadow = new ContactShadow(viewer.scene);
        // T5: measurement tools (point-to-point distance + area).
        const measure = new MeasureTool(viewer.scene);
        // T1: navigation hotspot markers (click-to-jump floor discs).
        const hotspots = new NavHotspots(viewer.scene);
        if (tour.nav) {
          hotspots.loadGraph(tour.nav);
        }
        navRef.current = hotspots;
        // T3: dollhouse view (orbit camera + sweep markers on reference mesh).
        const dollhouse = new DollhouseView(viewer.scene);
        dollhouse.initMarkers(tour.sweeps);
        const save = () => {
          if (!DEMO) saveProject(viewer.activeSweepUuid, staging.getAssetPosition(), staging.getAssetRotationY());
        };
        const placeDefault = () => {
          if (DEMO) staging.restoreAssetState(DEMO_CHAIR_POSITION, DEMO_CHAIR_ROTATION);
          else staging.placeInFrontOf(viewer.getCamera());
        };
        const restorePlacement = () => {
          if (savedProject?.asset) staging.restoreAssetState(savedProject.asset.position, savedProject.asset.rotationY);
          else if (viewer.activeSweepUuid) placeDefault();
        };
        const nudgeRelative = (x: number, z: number) => {
          const { x: dx, z: dz } = viewer.getCamera().getForwardRay(1).direction;
          const length = Math.hypot(dx, dz);
          if (length < 0.0001) return;
          staging.nudge((dx * z - dz * x) / length, (dz * z + dx * x) / length);
          save();
        };
        const nudgeFromShortcut = DEMO ? nudgeRelative : (x: number, z: number) => staging.nudge(x, z);
        const toggleDollhouse = () => {
          if (changingSweep) return;
          if (dollhouse.isActive) dollhouse.exit(viewer.getCamera());
          else dollhouse.enter(viewer.getCamera());
          setDollhouseActive(dollhouse.isActive);
        };
        const openSweep = async (uuid: string) => {
          if (changingSweep || viewer.isTransitioning) return;
          if (dollhouse.isActive) {
            dollhouse.exit(viewer.getCamera());
            setDollhouseActive(false);
          }
          if (uuid === viewer.activeSweepUuid) return;
          changingSweep = true;
          setPendingSweep(uuid);
          setNavigationError(null);
          try {
            await viewer.showSweep(uuid);
          } catch {
            if (!disposed) setNavigationError("The viewpoint could not be loaded. Select it to try again.");
          } finally {
            changingSweep = false;
            if (!disposed) setPendingSweep(null);
          }
        };
        controls.current = {
          toggleDollhouse,
          setFurnitureVisible: (visible) => {
            staging.setAssetVisible(visible);
            setFurnitureVisible(visible);
          },
          resetFurniture: () => {
            if (DEMO) placeDefault();
            else staging.resetPlacement(viewer.getCamera());
            save();
          },
          nudge: nudgeRelative,
          rotate: () => { staging.rotate(Math.PI / 12); save(); },
          openSweep: (uuid) => { void openSweep(uuid); },
        };
        viewer.onSweepShown = (uuid) => {
          if (disposed) return;
          useViewerStore.getState().setActiveSweep(uuid);
          // LOCAL X-ray scope follows the camera across sweep jumps.
          staging.refreshXray(viewer.getCamera().position);
          // T1: update hotspot markers for the new sweep's neighbors.
          hotspots.showForSweep(uuid, viewer.getCamera().position.y);
          // T3: highlight the active sweep in dollhouse mode.
          dollhouse.setActiveSweep(uuid);
          // T4: update minimap position indicator.
          minimapRef.current?.setActiveSweep(uuid);
          if (DEMO) useViewerStore.getState().setEnvState("Studio lighting");
          else void lighting
            .setForSweep(uuid, `${DATA_BASE_URL}/env/${uuid}.env`, viewer.getPanoOrientation())
            .then((state) => {
              if (disposed) return;
              useViewerStore.getState().setEnvState(state);
              shadow.updateFromEnvironment(viewer.scene.environmentTexture);
            });
          // T7: auto-save project state on every sweep change.
          save();
        };
        void staging
          .loadProxy([
            `${DATA_BASE_URL}/proxy_shell.glb`,
            `${DATA_BASE_URL}/proxy_details.glb`,
          ])
          .then(({ count, floors, failed }) => {
            if (disposed) return;
            if (!DEMO && viewer.activeSweepUuid) staging.placeInFrontOf(viewer.getCamera());
            useViewerStore
              .getState()
              .setProxyState(
                count > 0
                  ? `proxy \u2713 ${count} meshes \u00b7 ${floors} floors${failed.length ? ` (${failed.length} file missing)` : ""}`
                  : "proxy \u2717 copy proxy_shell.glb (+ proxy_details.glb) into viewer/public/data/",
              );
          });
        // G7.1 ground-truth reference (optional file): textured MatterPak mesh.
        // T2: after the reference loads, create the projective transition controller.
        void staging.loadReference(`${DATA_BASE_URL}/reference.glb`, DEMO ? "Y" : "Z").then((loaded) => {
          if (disposed) return;
          setModelAvailable(loaded);
          if (loaded) {
            const refMeshes = staging.getReferenceMeshes();
            if (refMeshes.length > 0 && !DEMO) {
              const transition = new SweepTransition(viewer.scene, [...refMeshes]);
              viewer.setTransition(transition);
              console.info(`[transition] projective morph ready (${refMeshes.length} ref meshes)`);
            }
          }
        });
        // T2: texture cache for neighbor prefetch (VRAM-bounded LRU).
        const texCache = new TextureCache(viewer.scene);
        viewer.setTextureCache(texCache);
        void staging
          .loadAsset(`${DATA_BASE_URL}/asset.glb`)
          .then(() => {
            if (disposed) return;
            restorePlacement();
            shadow.attachToAsset();
            setFurnitureAvailable(true);
            useViewerStore.getState().setAssetState("asset \u2713 dblclick floor to move");
          })
          .catch(() => {
            if (disposed) return;
            useViewerStore
              .getState()
              .setAssetState("asset \u2717 copy asset.glb into viewer/public/data/");
          });

        const onDblClick = (event: MouseEvent) => {
          if (dollhouse.isActive) return;
          staging.placeFromPick(event.offsetX, event.offsetY);
          // T7: auto-save after placement.
          save();
        };
        canvas.addEventListener("dblclick", onDblClick);

        // T6: shift+click drag to slide asset on floor; auto wall-snap on release.
        const onDragStart = (event: PointerEvent) => {
          if (dollhouse.isActive || !event.shiftKey || event.button !== 0) return;
          staging.dragStart();
          canvas.setPointerCapture(event.pointerId);
          event.preventDefault();
        };
        const onDragMove = (event: PointerEvent) => {
          if (!staging.isDragging) return;
          staging.dragMove(event.offsetX, event.offsetY);
        };
        const onDragEnd = (event: PointerEvent) => {
          if (!staging.isDragging) return;
          staging.dragEnd(true); // auto wall-snap
          canvas.releasePointerCapture(event.pointerId);
          // T7: auto-save after drag.
          save();
        };
        canvas.addEventListener("pointerdown", onDragStart);
        canvas.addEventListener("pointermove", onDragMove);
        canvas.addEventListener("pointerup", onDragEnd);

        // G7.1 defect triage: Ctrl+click identifies the proxy mesh under the
        // cursor (exact sub-mesh name -> console + HUD, magenta highlight).
        const onIdentify = (event: PointerEvent) => {
          if (!event.ctrlKey) return;
          useViewerStore.getState().setPickInfo(staging.identify(event.offsetX, event.offsetY));
        };
        canvas.addEventListener("pointerdown", onIdentify);

        // T1/T3: single-click on a nav hotspot or dollhouse marker -> jump.
        const onHotspotClick = (event: PointerEvent) => {
          if (event.ctrlKey) return;
          if (viewer.isTransitioning) return;
          // T5: in measurement mode, add points instead of jumping.
          if (measure.mode !== "off") {
            const status = measure.addPoint(event.offsetX, event.offsetY);
            if (status) useViewerStore.getState().setPickInfo(status);
            return;
          }
          // T3: in dollhouse mode, pick sweep markers instead of nav hotspots.
          if (dollhouse.isActive) {
            const target = dollhouse.pick(event.offsetX, event.offsetY);
            if (target) {
              void openSweep(target);
            }
            return;
          }
          const target = hotspots.pick(event.offsetX, event.offsetY);
          if (target) {
            void openSweep(target);
          }
        };
        canvas.addEventListener("click", onHotspotClick);

        const onKeyDown = (event: KeyboardEvent) => {
          const target = event.target;
          if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey ||
            (target instanceof HTMLElement && target !== canvas && target.closest("input, select, textarea, button, a, [contenteditable]"))) return;
          if (DEMO && target !== canvas) return;
          const step = event.shiftKey ? NUDGE_BIG_M : NUDGE_M;
          switch (event.key) {
            case "ArrowLeft":
              nudgeFromShortcut(-step, 0);
              break;
            case "ArrowRight":
              nudgeFromShortcut(step, 0);
              break;
            case "ArrowUp":
              nudgeFromShortcut(0, step);
              break;
            case "ArrowDown":
              nudgeFromShortcut(0, -step);
              break;
            case "PageUp":
              staging.lift(step);
              break;
            case "PageDown":
              staging.lift(-step);
              break;
            case "r":
            case "R":
              staging.rotate(Math.PI / 12);
              break;
            case "x":
            case "X": {
              const mode = staging.cycleXray(viewer.getCamera().position);
              useViewerStore.getState().setXray(mode === "off" ? null : mode);
              break;
            }
            case "p":
            case "P":
              viewer.toggleSkybox();
              break;
            case "c":
            case "C":
              lighting.toggleChromeSphere(viewer.getCamera());
              break;
            case "v":
            case "V":
              shadow.toggle();
              break;
            case "d":
            case "D":
              staging.toggleDetails();
              break;
            case "b":
            case "B":
              staging.toggleReference();
              break;
            case "n":
            case "N":
              staging.toggleReferenceFrame();
              break;
            case "g":
            case "G":
              staging.toggleReferenceOccluder();
              break;
            case "h":
            case "H":
              toggleDollhouse();
              break;
            case "m":
            case "M":
              // T4: toggle minimap.
              if (minimapRef.current) {
                const el = document.getElementById("minimap");
                if (el) {
                  const visible = el.style.display !== "none";
                  minimapRef.current.setVisible(!visible);
                }
              }
              break;
            case "l":
            case "L": {
              // T5: cycle measurement mode.
              const mmode = measure.cycleMode();
              useViewerStore.getState().setPickInfo(
                mmode === "off" ? null : `📏 ${mmode} mode — click to place points`,
              );
              break;
            }
            case "Escape":
              // T5: finish area measurement or disable measure mode.
              if (measure.mode === "area") {
                const result = measure.finishArea();
                if (result) useViewerStore.getState().setPickInfo(result);
              } else if (measure.mode !== "off") {
                measure.disable();
                useViewerStore.getState().setPickInfo(null);
              }
              break;
            case "s":
            case "S":
              // T7: export staging project as JSON file.
              if (DEMO) return;
              exportProject(
                viewer.activeSweepUuid,
                staging.getAssetPosition(),
                staging.getAssetRotationY(),
              );
              break;
            default:
              return;
          }
          event.preventDefault();
        };
        window.addEventListener("keydown", onKeyDown);

        engine.runRenderLoop(() => viewer.scene.render());
        const onResize = () => engine.resize();
        window.addEventListener("resize", onResize);
        const observer = new ResizeObserver(onResize);
        observer.observe(canvas);
        const fpsTimer = DEMO ? undefined : window.setInterval(() => {
          useViewerStore.getState().setFps(Math.round(engine.getFps()));
        }, 1000);

        cleanup = () => {
          window.clearInterval(fpsTimer);
          observer.disconnect();
          window.removeEventListener("resize", onResize);
          window.removeEventListener("keydown", onKeyDown);
          canvas.removeEventListener("dblclick", onDblClick);
          canvas.removeEventListener("pointerdown", onDragStart);
          canvas.removeEventListener("pointermove", onDragMove);
          canvas.removeEventListener("pointerup", onDragEnd);
          canvas.removeEventListener("pointerdown", onIdentify);
          canvas.removeEventListener("click", onHotspotClick);
          navRef.current = null;
          minimapRef.current?.dispose();
          minimapRef.current = null;
          viewer.dispose();
          engine.dispose();
          viewerRef.current = null;
          controls.current = null;
        };

        // T7: restore project from URL hash or localStorage.
        const initialSweepUuid = savedProject?.activeSweepUuid ?? tour.sweeps[0].uuid;
        // Validate the sweep UUID exists in the tour.
        const validSweep = tour.sweeps.some((s) => s.uuid === initialSweepUuid);
        await viewer.showSweep(validSweep ? initialSweepUuid : tour.sweeps[0].uuid);
        if (disposed) return;
        // If the asset finished loading before the first sweep positioned the
        // camera, re-run the initial placement from the real viewpoint.
        restorePlacement();
        useViewerStore.getState().setStatus("ready");

        // T4: load floorplan and create minimap overlay.
        const floorplan = DEMO ? null : await loadFloorplan(DATA_BASE_URL);
        if (!disposed && floorplan && canvas.parentElement) {
          const mm = createMinimap(canvas.parentElement, floorplan, (uuid) => {
            void openSweep(uuid);
          });
          mm.setActiveSweep(tour.sweeps[0].uuid);
          minimapRef.current = mm;
        }
      } catch (error) {
        if (!disposed) useViewerStore.getState().setError(error instanceof Error ? error.message : String(error));
      }
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, []);

  const goTo = (uuid: string) => {
    controls.current?.openSweep(uuid);
  };
  const goNeighbor = (offset: number) => {
    const viewer = viewerRef.current;
    const hotspots = navRef.current;
    if (!viewer || viewer.isTransitioning) return; // T2: block during animation
    // T1: use nav graph neighbors when available, fallback to sequential.
    const uuid = useViewerStore.getState().activeSweepUuid;
    if (!DEMO && uuid && hotspots) {
      const nbrs = hotspots.getNeighbors(uuid);
      if (nbrs && nbrs.length > 0) {
        // offset > 0 = next neighbor (nearest), < 0 = previous (farthest nearby)
        const idx = offset > 0 ? 0 : Math.min(nbrs.length - 1, 2);
        goTo(nbrs[idx]);
        return;
      }
    }
    const index = sweeps.findIndex((sweep) => sweep.uuid === uuid);
    const next = index >= 0 ? sweeps[index + offset] : undefined;
    if (next) goTo(next.uuid);
  };

  const sweepIndex = sweeps.findIndex((s) => s.uuid === activeSweepUuid);
  const activeSweep = sweeps[sweepIndex];
  const ready = status === "ready";
  const busy = !ready || pendingSweep !== null;
  const furnitureDisabled = busy || dollhouseActive || !furnitureAvailable;
  const placementDisabled = furnitureDisabled || !furnitureVisible;
  const sceneLabel = activeSweep ? sweepLabel(activeSweep.uuid, activeSweep.index) : "Preparing scene";

  return (
    <main className="studio">
      <header className="studio-header">
        <div>
          <p className="eyebrow">Achouri Youcef Islem / Spatial computing</p>
          <h1>Virtual staging</h1>
          <p className="studio-intro">Place furniture in a 360° scene and inspect the geometry behind it.</p>
        </div>
        <div className="workspace-meta">
          <span className="dataset-badge">{DEMO ? "Synthetic room" : "Local scan"}</span>
          {!SHOWCASE_BUILD && <a href={DEMO ? "./" : "?demo=1"}>{DEMO ? "Open local scan" : "Open sample room"}</a>}
        </div>
      </header>

      {DEMO && (
        <div className="panel-switch" role="group" aria-label="Explore the demo">
          <button type="button" aria-pressed={panel === "scene"} onClick={() => setPanel("scene")}>1. Explore the room</button>
          <button type="button" aria-pressed={panel === "outputs"} onClick={() => setPanel("outputs")}>2. Inspect model outputs</button>
        </div>
      )}

      <div className="workspace" hidden={panel !== "scene"}>
        <section className="scene-card" aria-label="Interactive staging viewer">
          <div className="scene-heading">
            <div><span className="eyebrow">{dollhouseActive ? "3D cutaway" : "Panorama"}</span><h2>{DEMO ? "Living room" : "Scan workspace"}</h2></div>
            <span className="scene-status" role="status">{pendingSweep ? "Changing viewpoint…" : ready ? sceneLabel : status === "error" ? "Viewer unavailable" : "Loading scene…"}</span>
          </div>
          <div className="canvas-wrap">
            <canvas ref={canvasRef} className="render-canvas" tabIndex={0} aria-label={dollhouseActive ? "3D room cutaway" : "360-degree room panorama"} />
            {(status === "boot" || status === "loading") && <div className="overlay" role="status">Loading geometry and panorama…</div>}
            {status === "error" && (
              <div className="overlay" role="alert">
                <strong>Scene unavailable</strong>
                <p>{DEMO ? "The interactive scene could not start. The recorded model outputs are available in the next view." : errorMessage}</p>
                <button type="button" onClick={() => window.location.reload()}>Reload scene</button>
              </div>
            )}
          </div>
          {navigationError && <p className="notice" role="alert">{navigationError}</p>}
          <p className="scene-hint">{dollhouseActive ? "Drag to orbit the room. Scroll to zoom." : "Drag to look around. Use the placement controls or double-click the floor to move the furniture."}</p>
          {pickInfo && <p className="scene-hint" role="status">{pickInfo}</p>}
        </section>

        <aside className="inspector" aria-label="Scene controls">
          <section>
            <h2>Viewpoints</h2>
            <label className="field-label" htmlFor="viewpoint">Choose a panorama</label>
            <select id="viewpoint" value={pendingSweep ?? activeSweepUuid ?? ""} disabled={busy} onChange={event => goTo(event.target.value)}>
              <option value="" disabled>Loading viewpoints…</option>
              {sweeps.map(({ uuid, index }) => <option key={uuid} value={uuid}>{sweepLabel(uuid, index)}</option>)}
            </select>
            <div className="button-row">
              <button type="button" disabled={busy || (DEMO && sweepIndex <= 0)} onClick={() => goNeighbor(-1)}>Previous</button>
              <button type="button" disabled={busy || (DEMO && sweepIndex >= sweeps.length - 1)} onClick={() => goNeighbor(1)}>Next</button>
            </div>
          </section>
          <section>
            <h2>Scene layers</h2>
            <button type="button" className="layer-control" aria-pressed={dollhouseActive} disabled={busy || !modelAvailable} onClick={() => controls.current?.toggleDollhouse()}>
              <span>3D cutaway</span><span>{dollhouseActive ? "On" : "Off"}</span>
            </button>
            <button type="button" className="layer-control" aria-pressed={furnitureVisible && furnitureAvailable} disabled={furnitureDisabled} onClick={() => controls.current?.setFurnitureVisible(!furnitureVisible)}>
              <span>Staged furniture</span><span>{furnitureAvailable ? furnitureVisible ? "On" : "Off" : "Unavailable"}</span>
            </button>
            <p className="inspector-note">Compare the original panorama with the added object. The room geometry supplies depth for occlusion.</p>
          </section>
          <section>
            <h2>Furniture placement</h2>
            <div className="button-row">
              <button type="button" disabled={placementDisabled} onClick={() => controls.current?.rotate()}>Rotate 15°</button>
              <button type="button" disabled={placementDisabled} onClick={() => controls.current?.resetFurniture()}>Reset</button>
            </div>
            <div className="button-row nudge-controls" role="group" aria-label="Nudge furniture by five centimetres">
              <button type="button" aria-label="Nudge left" disabled={placementDisabled} onClick={() => controls.current?.nudge(-NUDGE_M, 0)}>←</button>
              <button type="button" aria-label="Nudge forward" disabled={placementDisabled} onClick={() => controls.current?.nudge(0, NUDGE_M)}>↑</button>
              <button type="button" aria-label="Nudge back" disabled={placementDisabled} onClick={() => controls.current?.nudge(0, -NUDGE_M)}>↓</button>
              <button type="button" aria-label="Nudge right" disabled={placementDisabled} onClick={() => controls.current?.nudge(NUDGE_M, 0)}>→</button>
            </div>
          </section>
          {!DEMO && (
            <details className="diagnostics">
              <summary>Technical details</summary>
              <ul>
                {backend && <li>{backend.toUpperCase()} · {fps} FPS</li>}
                {[proxyState, assetState, envState, xray && `X-ray ${xray}`].filter(Boolean).map((item, i) => <li key={i}>{item}</li>)}
              </ul>
              <p>Shortcuts: H cutaway, X geometry, L measurement, M map, S export. Shift-drag slides the object.</p>
            </details>
          )}
        </aside>
      </div>

      {DEMO && panel === "outputs" && <ModelOutputs baseUrl={DATA_BASE_URL} />}
      <footer className="studio-footer"><span>Virtual staging prototype</span><span>{DEMO ? "Original synthetic data · Babylon.js" : backend ? backend === "webgpu" ? "WebGPU renderer" : "WebGL 2 renderer" : "Babylon.js viewer"}</span></footer>
    </main>
  );
}

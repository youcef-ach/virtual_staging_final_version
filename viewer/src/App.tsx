import { useEffect, useRef } from "react";
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

const DATA_BASE_URL = "/data";

/** Placement nudge steps (meters). */
const NUDGE_M = 0.05;
const NUDGE_BIG_M = 0.25;

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<TourViewer | null>(null);
  const navRef = useRef<NavHotspots | null>(null);
  const minimapRef = useRef<MinimapController | null>(null);
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

    void (async () => {
      const store = useViewerStore.getState();
      try {
        store.setStatus("loading");
        const tour = await loadTour(DATA_BASE_URL);
        const { engine, backend } = await createEngine(canvas);
        if (disposed) {
          engine.dispose();
          return;
        }
        store.setBackend(backend);
        store.setSweeps(tour.sweeps.map((s) => ({ uuid: s.uuid, index: s.index })));

        const viewer = new TourViewer(engine, tour);
        viewerRef.current = viewer;

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
        viewer.onSweepShown = (uuid) => {
          useViewerStore.getState().setActiveSweep(uuid);
          // LOCAL X-ray scope follows the camera across sweep jumps.
          staging.refreshXray(viewer.getCamera().position);
          // T1: update hotspot markers for the new sweep's neighbors.
          hotspots.showForSweep(uuid, viewer.getCamera().position.y);
          // T3: highlight the active sweep in dollhouse mode.
          dollhouse.setActiveSweep(uuid);
          // T4: update minimap position indicator.
          minimapRef.current?.setActiveSweep(uuid);
          void lighting
            .setForSweep(uuid, `${DATA_BASE_URL}/env/${uuid}.env`, viewer.getPanoOrientation())
            .then((state) => {
              useViewerStore.getState().setEnvState(state);
              shadow.updateFromEnvironment(viewer.scene.environmentTexture);
            });
          // T7: auto-save project state on every sweep change.
          saveProject(
            uuid,
            staging.getAssetPosition(),
            staging.getAssetRotationY(),
          );
        };
        void staging
          .loadProxy([
            `${DATA_BASE_URL}/proxy_shell.glb`,
            `${DATA_BASE_URL}/proxy_details.glb`,
          ])
          .then(({ count, floors, failed }) => {
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
        void staging.loadReference(`${DATA_BASE_URL}/reference.glb`).then((loaded) => {
          if (loaded) {
            const refMeshes = staging.getReferenceMeshes();
            if (refMeshes.length > 0) {
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
            staging.placeInFrontOf(viewer.getCamera());
            shadow.attachToAsset();
            useViewerStore.getState().setAssetState("asset \u2713 dblclick floor to move");
          })
          .catch(() => {
            useViewerStore
              .getState()
              .setAssetState("asset \u2717 copy asset.glb into viewer/public/data/");
          });

        const onDblClick = (event: MouseEvent) => {
          staging.placeFromPick(event.offsetX, event.offsetY);
          // T7: auto-save after placement.
          saveProject(
            viewer.activeSweepUuid,
            staging.getAssetPosition(),
            staging.getAssetRotationY(),
          );
        };
        canvas.addEventListener("dblclick", onDblClick);

        // T6: shift+click drag to slide asset on floor; auto wall-snap on release.
        const onDragStart = (event: PointerEvent) => {
          if (!event.shiftKey || event.button !== 0) return;
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
          saveProject(
            viewer.activeSweepUuid,
            staging.getAssetPosition(),
            staging.getAssetRotationY(),
          );
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
              dollhouse.exit(viewer.getCamera());
              void viewer.showSweep(target);
            }
            return;
          }
          const target = hotspots.pick(event.offsetX, event.offsetY);
          if (target) {
            void viewer.showSweep(target);
          }
        };
        canvas.addEventListener("click", onHotspotClick);

        const onKeyDown = (event: KeyboardEvent) => {
          const step = event.shiftKey ? NUDGE_BIG_M : NUDGE_M;
          switch (event.key) {
            case "ArrowLeft":
              staging.nudge(-step, 0);
              break;
            case "ArrowRight":
              staging.nudge(step, 0);
              break;
            case "ArrowUp":
              staging.nudge(0, step);
              break;
            case "ArrowDown":
              staging.nudge(0, -step);
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
              // T3: toggle dollhouse view.
              if (dollhouse.isActive) {
                dollhouse.exit(viewer.getCamera());
              } else {
                dollhouse.enter(viewer.getCamera());
              }
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
        const fpsTimer = window.setInterval(() => {
          useViewerStore.getState().setFps(Math.round(engine.getFps()));
        }, 1000);

        cleanup = () => {
          window.clearInterval(fpsTimer);
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
        };

        // T7: restore project from URL hash or localStorage.
        const shareProject = parseShareLink();
        const savedProject = shareProject ?? loadProject();
        const initialSweepUuid = savedProject?.activeSweepUuid ?? tour.sweeps[0].uuid;
        // Validate the sweep UUID exists in the tour.
        const validSweep = tour.sweeps.some((s) => s.uuid === initialSweepUuid);
        await viewer.showSweep(validSweep ? initialSweepUuid : tour.sweeps[0].uuid);
        // If the asset finished loading before the first sweep positioned the
        // camera, re-run the initial placement from the real viewpoint.
        if (savedProject?.asset) {
          staging.restoreAssetState(savedProject.asset.position, savedProject.asset.rotationY);
        } else {
          staging.placeInFrontOf(viewer.getCamera());
        }
        useViewerStore.getState().setStatus("ready");

        // T4: load floorplan and create minimap overlay.
        const floorplan = await loadFloorplan(DATA_BASE_URL);
        if (floorplan && canvas.parentElement) {
          const mm = createMinimap(canvas.parentElement, floorplan, (uuid) => {
            void viewer.showSweep(uuid);
          });
          mm.setActiveSweep(tour.sweeps[0].uuid);
          minimapRef.current = mm;
        }
      } catch (error) {
        useViewerStore.getState().setError(error instanceof Error ? error.message : String(error));
      }
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, []);

  const goTo = (uuid: string) => {
    void viewerRef.current?.showSweep(uuid);
  };
  const goNeighbor = (offset: number) => {
    const viewer = viewerRef.current;
    const hotspots = navRef.current;
    if (!viewer || viewer.isTransitioning) return; // T2: block during animation
    // T1: use nav graph neighbors when available, fallback to sequential.
    const uuid = useViewerStore.getState().activeSweepUuid;
    if (uuid && hotspots) {
      const nbrs = hotspots.getNeighbors(uuid);
      if (nbrs && nbrs.length > 0) {
        // offset > 0 = next neighbor (nearest), < 0 = previous (farthest nearby)
        const idx = offset > 0 ? 0 : Math.min(nbrs.length - 1, 2);
        void viewer.showSweep(nbrs[idx]);
        return;
      }
    }
    void viewer.showNeighbor(offset);
  };

  const activeIndex = sweeps.find((s) => s.uuid === activeSweepUuid)?.index;

  return (
    <div className="viewer-root">
      <canvas ref={canvasRef} className="render-canvas" />

      <div className="hud">
        <span>{backend ? backend.toUpperCase() : "initializing\u2026"}</span>
        {backend !== null && <span>{fps} FPS</span>}
        {activeIndex !== undefined && <span>sweep #{String(activeIndex).padStart(3, "0")}</span>}
        {proxyState && <span>{proxyState}</span>}
        {assetState && <span>{assetState}</span>}
        {envState && <span>{envState} (C)</span>}
        {xray && <span>X-RAY {xray} (X)</span>}
        {pickInfo && <span>{pickInfo}</span>}
      </div>

      {status === "loading" && <div className="overlay">Loading tour\u2026</div>}
      {status === "error" && (
        <div className="overlay overlay-error">
          <strong>Failed to start the viewer</strong>
          <p>{errorMessage}</p>
        </div>
      )}

      {sweeps.length > 0 && (
        <aside className="sweep-panel">
          <header>
            <span>Sweeps ({sweeps.length})</span>
            <span className="nav-buttons">
              <button onClick={() => goNeighbor(-1)} aria-label="previous sweep">
                {"\u2190"}
              </button>
              <button onClick={() => goNeighbor(1)} aria-label="next sweep">
                {"\u2192"}
              </button>
            </span>
          </header>
          <ul>
            {sweeps.map((sweep) => (
              <li key={sweep.uuid}>
                <button
                  className={sweep.uuid === activeSweepUuid ? "active" : ""}
                  onClick={() => goTo(sweep.uuid)}
                >
                  #{String(sweep.index).padStart(3, "0")} {"\u00b7"} {sweep.uuid.slice(0, 8)}
                </button>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  );
}

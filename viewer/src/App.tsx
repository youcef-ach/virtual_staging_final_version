import { useEffect, useRef } from "react";
import { createEngine } from "./engine/createEngine";
import { loadTour } from "./data/loadTour";
import { TourViewer } from "./scenes/tourViewer";
import { StagingLayer } from "./scenes/staging";
import { useViewerStore } from "./state/store";

const DATA_BASE_URL = "/data";

/** Placement nudge steps (meters). */
const NUDGE_M = 0.05;
const NUDGE_BIG_M = 0.25;

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<TourViewer | null>(null);
  const backend = useViewerStore((s) => s.backend);
  const fps = useViewerStore((s) => s.fps);
  const status = useViewerStore((s) => s.status);
  const errorMessage = useViewerStore((s) => s.errorMessage);
  const sweeps = useViewerStore((s) => s.sweeps);
  const activeSweepUuid = useViewerStore((s) => s.activeSweepUuid);
  const proxyState = useViewerStore((s) => s.proxyState);
  const assetState = useViewerStore((s) => s.assetState);
  const xray = useViewerStore((s) => s.xray);
  const skyboxYawDeg = useViewerStore((s) => s.skyboxYawDeg);

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
        viewer.onSweepShown = (uuid) => useViewerStore.getState().setActiveSweep(uuid);
        store.setSkyboxYaw(viewer.skyboxYawDeg);

        // G7: Layer 2 (proxy occluders) + Layer 3 (staged asset). Missing
        // files degrade gracefully -- the tour keeps working either way.
        const staging = new StagingLayer(viewer.scene);
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
        void staging
          .loadAsset(`${DATA_BASE_URL}/asset.glb`)
          .then(() => {
            staging.placeInFrontOf(viewer.getCamera());
            useViewerStore.getState().setAssetState("asset \u2713 dblclick floor to move");
          })
          .catch(() => {
            useViewerStore
              .getState()
              .setAssetState("asset \u2717 copy asset.glb into viewer/public/data/");
          });

        const onDblClick = (event: MouseEvent) => {
          staging.placeFromPick(event.offsetX, event.offsetY);
        };
        canvas.addEventListener("dblclick", onDblClick);

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
            case "X":
              useViewerStore.getState().setXray(staging.toggleXray());
              break;
            case "o":
            case "O":
              useViewerStore.getState().setSkyboxYaw(viewer.cycleSkyboxYaw());
              break;
            case "p":
            case "P":
              viewer.toggleSkybox();
              break;
            case "i":
            case "I":
              viewer.toggleRotationInverse();
              break;
            case "d":
            case "D":
              staging.toggleDetails();
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
          viewer.dispose();
          engine.dispose();
          viewerRef.current = null;
        };

        await viewer.showSweep(tour.sweeps[0].uuid);
        // If the asset finished loading before the first sweep positioned the
        // camera, re-run the initial placement from the real viewpoint.
        staging.placeInFrontOf(viewer.getCamera());
        useViewerStore.getState().setStatus("ready");
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
    void viewerRef.current?.showNeighbor(offset);
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
        {skyboxYawDeg !== null && <span>pano yaw {skyboxYawDeg}\u00b0 (O)</span>}
        {xray && <span>X-RAY (X)</span>}
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
                \u2190
              </button>
              <button onClick={() => goNeighbor(1)} aria-label="next sweep">
                \u2192
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
                  #{String(sweep.index).padStart(3, "0")} \u00b7 {sweep.uuid.slice(0, 8)}
                </button>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  );
}

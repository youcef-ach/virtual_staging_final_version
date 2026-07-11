import { useEffect, useRef } from "react";
import { createEngine } from "./engine/createEngine";
import { loadTour } from "./data/loadTour";
import { TourViewer } from "./scenes/tourViewer";
import { useViewerStore } from "./state/store";

const DATA_BASE_URL = "/data";

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<TourViewer | null>(null);
  const backend = useViewerStore((s) => s.backend);
  const fps = useViewerStore((s) => s.fps);
  const status = useViewerStore((s) => s.status);
  const errorMessage = useViewerStore((s) => s.errorMessage);
  const sweeps = useViewerStore((s) => s.sweeps);
  const activeSweepUuid = useViewerStore((s) => s.activeSweepUuid);

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

        engine.runRenderLoop(() => viewer.scene.render());
        const onResize = () => engine.resize();
        window.addEventListener("resize", onResize);
        const fpsTimer = window.setInterval(() => {
          useViewerStore.getState().setFps(Math.round(engine.getFps()));
        }, 1000);

        cleanup = () => {
          window.clearInterval(fpsTimer);
          window.removeEventListener("resize", onResize);
          viewer.dispose();
          engine.dispose();
          viewerRef.current = null;
        };

        await viewer.showSweep(tour.sweeps[0].uuid);
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
        <span>{backend ? backend.toUpperCase() : "initializing…"}</span>
        {backend !== null && <span>{fps} FPS</span>}
        {activeIndex !== undefined && <span>sweep #{String(activeIndex).padStart(3, "0")}</span>}
      </div>

      {status === "loading" && <div className="overlay">Loading tour…</div>}
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
                ←
              </button>
              <button onClick={() => goNeighbor(1)} aria-label="next sweep">
                →
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
                  #{String(sweep.index).padStart(3, "0")} · {sweep.uuid.slice(0, 8)}
                </button>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  );
}

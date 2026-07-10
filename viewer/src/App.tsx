import { useEffect, useRef } from "react";
import { createEngine } from "./engine/createEngine";
import { createPlaceholderScene } from "./scenes/placeholderScene";
import { useViewerStore } from "./state/store";

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const backend = useViewerStore((s) => s.backend);
  const fps = useViewerStore((s) => s.fps);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;
    let cleanup: (() => void) | undefined;

    void createEngine(canvas).then(({ engine, backend }) => {
      if (disposed) {
        engine.dispose();
        return;
      }
      useViewerStore.getState().setBackend(backend);

      const scene = createPlaceholderScene(engine);
      engine.runRenderLoop(() => scene.render());

      const onResize = () => engine.resize();
      window.addEventListener("resize", onResize);

      const fpsTimer = window.setInterval(() => {
        useViewerStore.getState().setFps(Math.round(engine.getFps()));
      }, 1000);

      cleanup = () => {
        window.clearInterval(fpsTimer);
        window.removeEventListener("resize", onResize);
        scene.dispose();
        engine.dispose();
      };
    });

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, []);

  return (
    <div className="viewer-root">
      <canvas ref={canvasRef} className="render-canvas" />
      <div className="hud">
        <span>{backend ? backend.toUpperCase() : "initializing…"}</span>
        {backend !== null && <span>{fps} FPS</span>}
      </div>
    </div>
  );
}

import { useState } from "react";

const classes = [
  ["Floor", "#50c850"], ["Wall", "#466ee6"], ["Ceiling", "#5adcdc"],
  ["Window", "#fadc3c"], ["Door", "#f58c28"], ["Furniture", "#dc50c8"], ["Other", "#808080"],
] as const;

const stages = [
  {
    id: "mask2former",
    label: "Semantic classes",
    model: "Mask2Former · Swin-T · ADE20K",
    image: "mask2former.webp",
    alt: "Color-coded semantic prediction on the living-room image",
    description: "Mask2Former predicts a class for each pixel. The pipeline maps the original labels into a smaller set used by the staging tools.",
  },
  {
    id: "sam",
    label: "Refined masks",
    model: "Segment Anything · ViT-B",
    image: "sam.webp",
    alt: "Semantic class overlay after accepted Segment Anything proposals are applied",
    description: "SAM proposes a contour for a selected Mask2Former region. The pipeline accepts a proposal only when it overlaps the original mask enough to pass its gate.",
  },
  {
    id: "difference",
    label: "Changed pixels",
    model: "Mask comparison",
    image: "difference.webp",
    alt: "Highlighted pixels whose semantic class changed between the two stages",
    description: "Highlighted pixels changed class after the refinement step. A change shows what the pipeline did. Accuracy would require independent annotations.",
  },
] as const;

export function ModelOutputs({ baseUrl }: { baseUrl: string }) {
  const [active, setActive] = useState<(typeof stages)[number]["id"]>("mask2former");
  const stage = stages.find((entry) => entry.id === active) ?? stages[0];
  const evidence = `${baseUrl}/evidence`;

  return (
    <section className="output-workspace" aria-labelledby="outputs-heading">
      <header>
        <p className="eyebrow">Pipeline inspection / One synthetic viewpoint</p>
        <h2 id="outputs-heading">From pixels to regions</h2>
        <p>Compare precomputed results from the two models on the same camera image.</p>
      </header>
      <div className="output-switch" role="group" aria-label="Choose a model output">
        {stages.map((entry) => (
          <button key={entry.id} type="button" aria-pressed={entry.id === active} onClick={() => setActive(entry.id)}>{entry.label}</button>
        ))}
      </div>
      <div className="output-grid" aria-label="Input and model output comparison">
        <figure>
          <img src={`${evidence}/input.webp`} width={640} height={640} alt="A synthetic living room with a broad window, seating, and a doorway" decoding="async" />
          <figcaption><strong>Input image</strong><span>One perspective view of the synthetic room</span></figcaption>
        </figure>
        <figure>
          <img src={`${evidence}/${stage.image}`} width={640} height={640} alt={stage.alt} decoding="async" />
          <figcaption><strong>{stage.label}</strong><span>{stage.model}</span></figcaption>
        </figure>
      </div>
      {active !== "difference" && <ul className="output-legend" aria-label="Semantic class colors">
        {classes.map(([name, color]) => <li key={name}><span style={{ backgroundColor: color }} aria-hidden="true" />{name}</li>)}
      </ul>}
      <p className="output-explanation" role="status">{stage.description}</p>
      <details className="output-provenance"><summary>Run provenance</summary><p>The assets record the model identifiers, input hash, and refinement statistics. <a href={`${evidence}/manifest.json`}>View the run record</a>.</p></details>
    </section>
  );
}

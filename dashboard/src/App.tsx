import { useEffect, useMemo, useState } from "react";
import type { PoseQAReport, SweepQA } from "./types";

const QA_BASE = "/qa";
const FACE_ORDER = ["px", "nx", "py", "ny", "pz", "nz"] as const;

type Filter = "flagged" | "spot" | "all";

function compositeUrl(sweep: SweepQA): string {
  return `${QA_BASE}/composites/${String(sweep.index).padStart(3, "0")}_${sweep.sweep_uuid}.jpg`;
}

export default function App() {
  const [report, setReport] = useState<PoseQAReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("flagged");
  const [selectedUuid, setSelectedUuid] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${QA_BASE}/pose_qa.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(
            `pose_qa.json not found under ${QA_BASE}/ (HTTP ${response.status}). ` +
              "Copy the pipeline qa_output folder into dashboard/public/qa/.",
          );
        }
        return response.json() as Promise<PoseQAReport>;
      })
      .then((data) => {
        setReport(data);
        const first = data.sweeps.find((s) => s.flagged) ?? data.sweeps[0];
        setSelectedUuid(first ? first.sweep_uuid : null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const visibleSweeps = useMemo(() => {
    if (!report) return [];
    let list = [...report.sweeps];
    if (filter === "flagged") list = list.filter((s) => s.flagged);
    if (filter === "spot") list = list.filter((s) => s.spot_check);
    return list.sort((a, b) => (a.score ?? -1) - (b.score ?? -1));
  }, [report, filter]);

  const selected = report?.sweeps.find((s) => s.sweep_uuid === selectedUuid) ?? null;

  if (error) {
    return <div className="overlay error">{error}</div>;
  }
  if (!report) {
    return <div className="overlay">Loading pose QA report…</div>;
  }

  return (
    <div className="dashboard">
      <header>
        <h1>Pose QA · Gate 1</h1>
        <div className="stats">
          <span>{report.sweep_count} sweeps</span>
          <span className="stat-flag">{report.flagged_count} flagged</span>
          <span>{report.spot_check_count} spot-check</span>
          <span>threshold {report.threshold}</span>
        </div>
      </header>

      <div className="layout">
        <aside>
          <div className="filters">
            {(
              [
                ["flagged", "Flagged"],
                ["spot", "Spot-check"],
                ["all", "All"],
              ] as [Filter, string][]
            ).map(([value, label]) => (
              <button
                key={value}
                className={filter === value ? "active" : ""}
                onClick={() => setFilter(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <ul className="sweep-list">
            {visibleSweeps.map((sweep) => (
              <li key={sweep.sweep_uuid}>
                <button
                  className={sweep.sweep_uuid === selectedUuid ? "active" : ""}
                  onClick={() => setSelectedUuid(sweep.sweep_uuid)}
                >
                  <span className="idx">#{String(sweep.index).padStart(3, "0")}</span>
                  <span className={sweep.flagged ? "score bad" : "score"}>
                    {sweep.score === null ? "n/a" : sweep.score.toFixed(3)}
                  </span>
                  {sweep.flagged && <span className="badge flag">FLAG</span>}
                  {sweep.spot_check && <span className="badge">SPOT</span>}
                </button>
              </li>
            ))}
            {visibleSweeps.length === 0 && <li className="empty">No sweeps in this filter.</li>}
          </ul>
        </aside>

        <main>
          {selected ? (
            <>
              <h2>
                #{String(selected.index).padStart(3, "0")} · {selected.sweep_uuid}
              </h2>
              <table className="face-table">
                <thead>
                  <tr>
                    {FACE_ORDER.map((face) => (
                      <th key={face}>{face}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    {FACE_ORDER.map((face) => {
                      const value = selected.face_scores[face];
                      const noSignal = selected.uninformative_faces.includes(face);
                      return (
                        <td
                          key={face}
                          className={
                            noSignal ? "muted" : value !== undefined && value < 0.5 ? "bad" : ""
                          }
                        >
                          {noSignal ? "no-signal" : value !== undefined ? value.toFixed(3) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                </tbody>
              </table>

              {selected.flagged || selected.spot_check ? (
                <figure className="composite">
                  <img
                    src={compositeUrl(selected)}
                    alt={`review composite for sweep ${selected.index}`}
                  />
                  <figcaption>
                    Left: panorama face · Right: rendered geometry edges in red. Rows are the
                    informative faces with their scores.
                  </figcaption>
                </figure>
              ) : (
                <p className="note">
                  Sweep passed — composites are generated for flagged and spot-check sweeps only.
                </p>
              )}
            </>
          ) : (
            <p className="note">Select a sweep from the queue.</p>
          )}
        </main>
      </div>
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import type { PoseQAReport, SweepQA } from "./types";

const QA_BASE = "/qa";
const FACE_ORDER = ["px", "nx", "py", "ny", "pz", "nz"] as const;

type Filter = "flagged" | "low_score" | "low_margin" | "spot" | "all";

function compositeUrl(sweep: SweepQA): string {
  return `${QA_BASE}/composites/${String(sweep.index).padStart(3, "0")}_${sweep.sweep_uuid}.jpg`;
}

function hasReason(sweep: SweepQA, reason: string): boolean {
  return sweep.flag_reasons?.includes(reason) ?? false;
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
    if (filter === "low_score") list = list.filter((s) => hasReason(s, "low_score"));
    if (filter === "low_margin") list = list.filter((s) => hasReason(s, "low_margin"));
    const key = (s: SweepQA) => (filter === "low_margin" ? s.margin ?? -1 : s.score ?? -1);
    return list.sort((a, b) => key(a) - key(b));
  }, [report, filter]);

  const selected = report?.sweeps.find((s) => s.sweep_uuid === selectedUuid) ?? null;

  if (error) {
    return <div className="overlay error">{error}</div>;
  }
  if (!report) {
    return <div className="overlay">Loading pose QA report…</div>;
  }

  const scoreFloor = report.threshold ?? 0.2;
  const marginFloor = report.margin_threshold ?? 0.03;

  return (
    <div className="dashboard">
      <header>
        <h1>Pose QA · Gate 1</h1>
        <div className="stats">
          <span>{report.sweep_count} sweeps</span>
          <span className="stat-flag">{report.flagged_count} flagged</span>
          <span>{report.spot_check_count} spot-check</span>
          <span>score floor {scoreFloor}</span>
          <span>margin floor {marginFloor}</span>
        </div>
      </header>

      <div className="layout">
        <aside>
          <div className="filters">
            {(
              [
                ["flagged", "Flagged"],
                ["low_score", "Score"],
                ["low_margin", "Margin"],
                ["spot", "Spot"],
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
                  <span className={hasReason(sweep, "low_score") ? "score bad" : "score"}>
                    {sweep.score === null || sweep.score === undefined
                      ? "n/a"
                      : sweep.score.toFixed(3)}
                  </span>
                  <span className={hasReason(sweep, "low_margin") ? "margin bad" : "margin"}>
                    {sweep.margin === null || sweep.margin === undefined
                      ? ""
                      : `m ${sweep.margin.toFixed(3)}`}
                  </span>
                  {hasReason(sweep, "low_score") && (
                    <span className="badge flag" title="low_score: gross misregistration floor">
                      S
                    </span>
                  )}
                  {hasReason(sweep, "low_margin") && (
                    <span className="badge flag" title="low_margin: subtle drift / ambiguous">
                      M
                    </span>
                  )}
                  {sweep.flagged && !sweep.flag_reasons && <span className="badge flag">FLAG</span>}
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
              <div className="sweep-summary">
                <span>
                  score{" "}
                  <b className={selected.score !== null && selected.score < scoreFloor ? "bad" : ""}>
                    {selected.score === null ? "n/a" : selected.score.toFixed(3)}
                  </b>
                </span>
                <span>
                  margin{" "}
                  <b
                    className={
                      selected.margin !== null &&
                      selected.margin !== undefined &&
                      selected.margin < marginFloor
                        ? "bad"
                        : ""
                    }
                  >
                    {selected.margin === null || selected.margin === undefined
                      ? "n/a"
                      : selected.margin.toFixed(3)}
                  </b>
                </span>
                {selected.flag_reasons?.map((reason) => (
                  <span key={reason} className="badge flag">
                    {reason}
                  </span>
                ))}
                {selected.spot_check && <span className="badge">spot-check</span>}
              </div>
              <table className="face-table">
                <thead>
                  <tr>
                    <th className="row-label" />
                    {FACE_ORDER.map((face) => (
                      <th key={face}>{face}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th className="row-label">score</th>
                    {FACE_ORDER.map((face) => {
                      const value = selected.face_scores[face];
                      const noSignal = selected.uninformative_faces.includes(face);
                      return (
                        <td
                          key={face}
                          className={
                            noSignal
                              ? "muted"
                              : value !== undefined && value < scoreFloor
                                ? "bad"
                                : ""
                          }
                        >
                          {noSignal ? "no-signal" : value !== undefined ? value.toFixed(3) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                  {selected.face_margins && (
                    <tr>
                      <th className="row-label">margin</th>
                      {FACE_ORDER.map((face) => {
                        const value = selected.face_margins?.[face];
                        return (
                          <td
                            key={face}
                            className={value !== undefined && value < marginFloor ? "bad" : ""}
                          >
                            {value !== undefined ? value.toFixed(3) : "—"}
                          </td>
                        );
                      })}
                    </tr>
                  )}
                  {selected.raw_face_scores && (
                    <tr>
                      <th className="row-label">raw</th>
                      {FACE_ORDER.map((face) => {
                        const value = selected.raw_face_scores?.[face];
                        return (
                          <td key={face} className="muted">
                            {value !== undefined ? value.toFixed(3) : "—"}
                          </td>
                        );
                      })}
                    </tr>
                  )}
                </tbody>
              </table>

              {selected.flagged || selected.spot_check ? (
                <figure className="composite">
                  <img
                    src={compositeUrl(selected)}
                    alt={`review composite for sweep ${selected.index}`}
                  />
                  <figcaption>
                    Columns: panorama · OBJ render (normalized depth, oriented) · panorama with
                    rendered edges in red. Rows are the informative faces with their normalized
                    scores.
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

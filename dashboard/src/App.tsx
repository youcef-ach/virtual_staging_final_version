import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  CatalogItem,
  PoseQAReport,
  SweepQA,
  PipelineRun,
  PipelineRunDetail,
  StageExecution,
  StageStatusType,
  Property,
  ScanVersion,
} from "./types";
import { DAG_EDGES } from "./types";

const QA_BASE = "/qa";
const FACE_ORDER = ["px", "nx", "py", "ny", "pz", "nz"] as const;

type QAFilter = "flagged" | "low_score" | "low_margin" | "spot" | "all";
type Tab = "qa" | "catalog" | "pipeline";

function compositeUrl(sweep: SweepQA): string {
  return `${QA_BASE}/composites/${String(sweep.index).padStart(3, "0")}_${sweep.sweep_uuid}.jpg`;
}

function hasReason(sweep: SweepQA, reason: string): boolean {
  return sweep.flag_reasons?.includes(reason) ?? false;
}

// ── DAG layout positions (SVG viewBox: 0 0 900 220) ──────────────────
// Laid out as a topological flow from left to right:
//   Col 0: Stage 1  (root)
//   Col 1: Stage 2, 3, 8  (depend on 1)
//   Col 2: Stage 4, 5      (depend on 3)
//   Col 3: Stage 6          (depends on 4, 5)
//   Col 4: Stage 7, 9       (depend on 6)
//   Col 5: Stage 10         (depends on all)

const NODE_POSITIONS: Record<number, { x: number; y: number }> = {
  1:  { x: 70,  y: 110 },
  2:  { x: 220, y: 35  },
  3:  { x: 220, y: 110 },
  8:  { x: 220, y: 185 },
  4:  { x: 380, y: 60  },
  5:  { x: 380, y: 155 },
  6:  { x: 530, y: 110 },
  7:  { x: 680, y: 60  },
  9:  { x: 680, y: 160 },
  10: { x: 830, y: 110 },
};

// Short labels for DAG nodes
const NODE_LABELS: Record<number, string> = {
  1: "Ingest",
  2: "Pose QA",
  3: "PointCloud",
  4: "Segment",
  5: "Planes",
  6: "Proxy",
  7: "G-Buffer",
  8: "IBL",
  9: "NavGraph",
  10: "Publish",
};

function statusIcon(status: StageStatusType): string {
  switch (status) {
    case "completed": return "✓";
    case "running": return "⟳";
    case "failed": return "✗";
    case "cancelled": return "—";
    default: return "";
  }
}

function formatDuration(start: string | null, end: string | null): string {
  if (!start) return "—";
  const s = new Date(start).getTime();
  const e = end ? new Date(end).getTime() : Date.now();
  const sec = (e - s) / 1000;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  return `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s`;
}

function formatTime(t: string | null): string {
  if (!t) return "—";
  return new Date(t).toLocaleTimeString();
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("pipeline");

  // ── Pose QA state ──────────────────────────────────────────────────
  const [report, setReport] = useState<PoseQAReport | null>(null);
  const [qaError, setQaError] = useState<string | null>(null);
  const [qaFilter, setQaFilter] = useState<QAFilter>("flagged");
  const [selectedUuid, setSelectedUuid] = useState<string | null>(null);

  // ── Catalog state ──────────────────────────────────────────────────
  const [catalogItems, setCatalogItems] = useState<CatalogItem[]>([]);
  const [catalogCategory, setCatalogCategory] = useState<string>("all");
  const [catalogSearch, setCatalogSearch] = useState<string>("");
  const [catalogLoading, setCatalogLoading] = useState<boolean>(false);
  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);

  // ── Upload form state ──────────────────────────────────────────────
  const [sku, setSku] = useState("");
  const [productName, setProductName] = useState("");
  const [manufacturer, setManufacturer] = useState("Nordic Living");
  const [category, setCategory] = useState("sofas");
  const [price, setPrice] = useState("899");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // ── Pipeline state ─────────────────────────────────────────────────
  const [properties, setProperties] = useState<Property[]>([]);
  const [selectedPropertyId, setSelectedPropertyId] = useState<string | null>(null);
  const [scanVersions, setScanVersions] = useState<ScanVersion[]>([]);
  const [selectedScanVersionId, setSelectedScanVersionId] = useState<string | null>(null);
  
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<PipelineRunDetail | null>(null);
  const [selectedStage, setSelectedStage] = useState<number | null>(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Pose QA loader ─────────────────────────────────────────────────
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
      .catch((e) => setQaError(e instanceof Error ? e.message : String(e)));
  }, []);

  // ── Catalog loader ─────────────────────────────────────────────────
  const fetchCatalog = () => {
    setCatalogLoading(true);
    let url = "/api/catalog";
    const params = new URLSearchParams();
    if (catalogCategory !== "all") params.append("category", catalogCategory);
    if (catalogSearch) params.append("search", catalogSearch);
    if (params.toString()) url += `?${params.toString()}`;

    fetch(url)
      .then((res) => (res.ok ? res.json() : []))
      .then((data: CatalogItem[]) => setCatalogItems(data))
      .catch(() => setCatalogItems([]))
      .finally(() => setCatalogLoading(false));
  };

  useEffect(() => {
    fetchCatalog();
  }, [catalogCategory, catalogSearch]);

  const fetchProperties = useCallback(() => {
    fetch("/api/properties")
      .then((res) => (res.ok ? res.json() : []))
      .then((data: Property[]) => {
        setProperties(data);
        if (data.length > 0 && !selectedPropertyId) {
          setSelectedPropertyId(data[0].id);
        }
      })
      .catch(() => setProperties([]));
  }, [selectedPropertyId]);

  const fetchScanVersions = useCallback(() => {
    if (!selectedPropertyId) {
      setScanVersions([]);
      return;
    }
    fetch(`/api/properties/${selectedPropertyId}/scan-versions`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data: ScanVersion[]) => {
        setScanVersions(data);
        if (data.length > 0) {
          setSelectedScanVersionId(data[0].id);
        } else {
          setSelectedScanVersionId(null);
        }
      })
      .catch(() => {
        setScanVersions([]);
        setSelectedScanVersionId(null);
      });
  }, [selectedPropertyId]);

  const fetchRuns = useCallback(() => {
    let url = "/api/pipeline/runs";
    if (selectedScanVersionId) {
      url += `?scan_version_id=${selectedScanVersionId}`;
    }
    fetch(url)
      .then((res) => (res.ok ? res.json() : []))
      .then((data: PipelineRun[]) => {
        setPipelineRuns(data);
        // Auto-select the most recent run
        if (data.length > 0 && (!selectedRunId || !data.find((r) => r.id === selectedRunId))) {
          setSelectedRunId(data[0].id);
        } else if (data.length === 0) {
          setSelectedRunId(null);
        }
      })
      .catch(() => setPipelineRuns([]));
  }, [selectedScanVersionId, selectedRunId]);

  const fetchRunDetail = useCallback(() => {
    if (!selectedRunId) {
      setRunDetail(null);
      return;
    }
    fetch(`/api/pipeline/runs/${selectedRunId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: PipelineRunDetail | null) => setRunDetail(data))
      .catch(() => setRunDetail(null));
  }, [selectedRunId]);

  // Fetch properties when pipeline tab is active
  useEffect(() => {
    if (activeTab === "pipeline") {
      fetchProperties();
    }
  }, [activeTab, fetchProperties]);

  // Fetch scan versions when selected property changes
  useEffect(() => {
    fetchScanVersions();
  }, [selectedPropertyId, fetchScanVersions]);

  // Fetch runs when selected scan version changes
  useEffect(() => {
    if (activeTab === "pipeline") {
      fetchRuns();
    }
  }, [activeTab, selectedScanVersionId, fetchRuns]);

  // Fetch run detail when selected run changes
  useEffect(() => {
    fetchRunDetail();
  }, [selectedRunId, fetchRunDetail]);

  // Poll for updates when a run is active — SSE primary, interval fallback.
  const sseRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Clean up any existing SSE / poll
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    const isActive =
      activeTab === "pipeline" &&
      selectedRunId &&
      runDetail &&
      (runDetail.status === "running" || runDetail.status === "pending");

    if (!isActive || !selectedRunId) return;

    // Try SSE first — real-time updates via /api/pipeline/runs/{id}/events
    try {
      const es = new EventSource(`/api/pipeline/runs/${selectedRunId}/events`);
      sseRef.current = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.error || data.event === "done") {
            es.close();
            sseRef.current = null;
            fetchRuns();
            fetchRunDetail();
            return;
          }
          if (data.stages) {
            setRunDetail(data as PipelineRunDetail);
            // Also refresh the run list to update status chips
            fetchRuns();
          }
        } catch {
          // ignore parse errors
        }
      };

      es.onerror = () => {
        // SSE failed — fall back to polling
        es.close();
        sseRef.current = null;
        pollRef.current = setInterval(() => {
          fetchRunDetail();
          fetchRuns();
        }, 1500);
      };
    } catch {
      // EventSource not supported — fall back to polling
      pollRef.current = setInterval(() => {
        fetchRunDetail();
        fetchRuns();
      }, 1500);
    }

    return () => {
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [activeTab, runDetail?.status, selectedRunId, fetchRunDetail, fetchRuns]);

  // ── Pipeline actions ───────────────────────────────────────────────
  const handleStartRun = async () => {
    setPipelineLoading(true);
    try {
      if (!selectedScanVersionId) {
        alert("No scan version selected. Create a property and upload a scan first.");
        setPipelineLoading(false);
        return;
      }

      const res = await fetch("/api/pipeline/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scan_version_id: selectedScanVersionId }),
      });

      if (res.ok) {
        const run: PipelineRun = await res.json();
        setSelectedRunId(run.id);
        fetchRuns();
      } else {
        const err = await res.json();
        alert(`Failed to start pipeline: ${err.detail || "Unknown error"}`);
      }
    } catch (err) {
      alert(`Error starting pipeline: ${err}`);
    } finally {
      setPipelineLoading(false);
    }
  };

  const handleRetryStage = async (stageNumber: number) => {
    if (!selectedRunId) return;
    try {
      const res = await fetch(
        `/api/pipeline/runs/${selectedRunId}/stages/${stageNumber}/retry`,
        { method: "POST" },
      );
      if (res.ok) {
        fetchRunDetail();
      } else {
        const err = await res.json();
        alert(`Retry failed: ${err.detail || "Unknown error"}`);
      }
    } catch (err) {
      alert(`Retry error: ${err}`);
    }
  };

  const handleCancelRun = async () => {
    if (!selectedRunId) return;
    if (!confirm("Cancel this pipeline run?")) return;
    try {
      const res = await fetch(`/api/pipeline/runs/${selectedRunId}/cancel`, {
        method: "POST",
      });
      if (res.ok) {
        fetchRunDetail();
        fetchRuns();
      }
    } catch (err) {
      alert(`Cancel error: ${err}`);
    }
  };

  // ── Catalog actions ────────────────────────────────────────────────
  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || !sku || !productName) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("sku", sku);
    formData.append("product_name", productName);
    formData.append("manufacturer", manufacturer);
    formData.append("category", category);
    if (price) formData.append("price", price);

    try {
      const res = await fetch("/api/catalog/upload", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        setShowUploadModal(false);
        setSku("");
        setProductName("");
        setSelectedFile(null);
        fetchCatalog();
      } else {
        const err = await res.json();
        alert(`Upload failed: ${err.detail || "Unknown error"}`);
      }
    } catch (err) {
      alert(`Upload error: ${err}`);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteItem = async (id: string, itemSku: string) => {
    if (!confirm(`Delete catalog item ${itemSku}?`)) return;
    try {
      const res = await fetch(`/api/catalog/${id}`, { method: "DELETE" });
      if (res.ok) {
        fetchCatalog();
      }
    } catch (err) {
      alert(`Delete failed: ${err}`);
    }
  };

  // ── QA computed values ─────────────────────────────────────────────
  const visibleSweeps = useMemo(() => {
    if (!report) return [];
    let list = [...report.sweeps];
    if (qaFilter === "flagged") list = list.filter((s) => s.flagged);
    if (qaFilter === "spot") list = list.filter((s) => s.spot_check);
    if (qaFilter === "low_score") list = list.filter((s) => hasReason(s, "low_score"));
    if (qaFilter === "low_margin") list = list.filter((s) => hasReason(s, "low_margin"));
    const key = (s: SweepQA) => (qaFilter === "low_margin" ? s.margin ?? -1 : s.score ?? -1);
    return list.sort((a, b) => key(a) - key(b));
  }, [report, qaFilter]);

  const selected = report?.sweeps.find((s) => s.sweep_uuid === selectedUuid) ?? null;

  // Stage status map for DAG rendering
  const stageStatusMap = useMemo<Record<number, StageStatusType>>(() => {
    if (!runDetail) return {};
    const map: Record<number, StageStatusType> = {};
    for (const s of runDetail.stages) {
      map[s.stage_number] = s.status;
    }
    return map;
  }, [runDetail]);

  const selectedStageData = useMemo<StageExecution | null>(() => {
    if (!runDetail || selectedStage === null) return null;
    return runDetail.stages.find((s) => s.stage_number === selectedStage) ?? null;
  }, [runDetail, selectedStage]);

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div className="dashboard">
      <header>
        <div className="brand-title">
          <h1>Virtual Staging Platform</h1>
          <nav className="tab-nav">
            <button
              className={activeTab === "pipeline" ? "tab active" : "tab"}
              onClick={() => setActiveTab("pipeline")}
            >
              Pipeline
            </button>
            <button
              className={activeTab === "catalog" ? "tab active" : "tab"}
              onClick={() => setActiveTab("catalog")}
            >
              Furniture Catalog
            </button>
            <button
              className={activeTab === "qa" ? "tab active" : "tab"}
              onClick={() => setActiveTab("qa")}
            >
              Pose QA · Gate 1
            </button>
          </nav>
        </div>

        {activeTab === "qa" && report && (
          <div className="stats">
            <span>{report.sweep_count} sweeps</span>
            <span className="stat-flag">{report.flagged_count} flagged</span>
            <span>{report.spot_check_count} spot-check</span>
          </div>
        )}

        {activeTab === "catalog" && (
          <div className="stats">
            <span>{catalogItems.length} assets cataloged</span>
            <button className="primary-btn" onClick={() => setShowUploadModal(true)}>
              + Upload &amp; Normalize GLB
            </button>
          </div>
        )}

        {activeTab === "pipeline" && (
          <div className="pipeline-actions">
            <span style={{ fontSize: 13, color: "#b8bcc2" }}>
              {pipelineRuns.length} run{pipelineRuns.length !== 1 ? "s" : ""}
            </span>
            {runDetail &&
              (runDetail.status === "running" || runDetail.status === "pending") && (
                <button className="btn-warning" onClick={handleCancelRun}>
                  Cancel Run
                </button>
              )}
            <button
              className="primary-btn"
              onClick={handleStartRun}
              disabled={pipelineLoading}
            >
              {pipelineLoading ? "Starting..." : "▶ Start Pipeline"}
            </button>
          </div>
        )}
      </header>

      {/* ── Pipeline Tab ───────────────────────────────────────────── */}
      {activeTab === "pipeline" && (
        <div className="pipeline-container">
          {/* Property & Scan Selector */}
          <div style={{ display: "flex", gap: "10px", marginBottom: "1rem", alignItems: "center" }}>
            <span style={{ fontSize: 13, color: "#b8bcc2" }}>Property:</span>
            <select
              value={selectedPropertyId || ""}
              onChange={(e) => setSelectedPropertyId(e.target.value)}
              style={{ padding: "4px 8px", background: "#1c2128", color: "#adbac7", border: "1px solid #444c56", borderRadius: "4px" }}
            >
              {properties.length === 0 && <option value="">No properties found</option>}
              {properties.map(p => (
                <option key={p.id} value={p.id}>{p.name} {p.address ? `(${p.address})` : ""}</option>
              ))}
            </select>
            <span style={{ fontSize: 13, color: "#b8bcc2", marginLeft: "10px" }}>Scan Version:</span>
            <select
              value={selectedScanVersionId || ""}
              onChange={(e) => setSelectedScanVersionId(e.target.value)}
              style={{ padding: "4px 8px", background: "#1c2128", color: "#adbac7", border: "1px solid #444c56", borderRadius: "4px" }}
            >
              {scanVersions.length === 0 && <option value="">No scans found</option>}
              {scanVersions.map(sv => (
                <option key={sv.id} value={sv.id}>v{sv.version} ({sv.status})</option>
              ))}
            </select>
          </div>

          {/* Run history chips */}
          {pipelineRuns.length > 0 && (
            <div className="run-history">
              <span style={{ fontSize: 12, color: "#7d828a", marginRight: 4 }}>
                Runs:
              </span>
              {pipelineRuns.map((run) => (
                <button
                  key={run.id}
                  className={`run-chip ${run.id === selectedRunId ? "active" : ""}`}
                  onClick={() => {
                    setSelectedRunId(run.id);
                    setSelectedStage(null);
                  }}
                >
                  <span className={`run-status-dot ${run.status}`} />
                  <span>{new Date(run.created_at).toLocaleTimeString()}</span>
                  <span style={{ textTransform: "uppercase", fontSize: 10, opacity: 0.7 }}>
                    {run.status}
                  </span>
                </button>
              ))}
            </div>
          )}

          {runDetail ? (
            <>
              {/* DAG Visualization */}
              <div className="dag-panel">
                <h3>Pipeline DAG — {runDetail.status.toUpperCase()}</h3>
                <svg className="dag-svg" viewBox="0 0 900 220" xmlns="http://www.w3.org/2000/svg">
                  {/* Edges */}
                  {DAG_EDGES.map((edge, i) => {
                    const from = NODE_POSITIONS[edge.from];
                    const to = NODE_POSITIONS[edge.to];
                    if (!from || !to) return null;
                    const bothCompleted =
                      stageStatusMap[edge.from] === "completed" &&
                      stageStatusMap[edge.to] === "completed";
                    return (
                      <line
                        key={`edge-${i}`}
                        x1={from.x}
                        y1={from.y}
                        x2={to.x}
                        y2={to.y}
                        className={`dag-edge ${bothCompleted ? "completed" : ""}`}
                      />
                    );
                  })}

                  {/* Nodes */}
                  {Object.entries(NODE_POSITIONS).map(([numStr, pos]) => {
                    const num = parseInt(numStr, 10);
                    const status = stageStatusMap[num] || "pending";
                    return (
                      <g
                        key={`node-${num}`}
                        onClick={() => setSelectedStage(num)}
                        style={{ cursor: "pointer" }}
                      >
                        <circle
                          cx={pos.x}
                          cy={pos.y}
                          r={22}
                          strokeWidth={2}
                          className={`dag-node-circle ${status}`}
                        />
                        <text
                          x={pos.x}
                          y={pos.y + 1}
                          className="dag-node-number"
                        >
                          {statusIcon(status) || num}
                        </text>
                        <text
                          x={pos.x}
                          y={pos.y + 36}
                          className="dag-node-label"
                        >
                          {NODE_LABELS[num] || `Stage ${num}`}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Stage cards grid */}
              <div className="stage-grid">
                {runDetail.stages.map((stage) => (
                  <div
                    key={stage.stage_number}
                    className={`stage-card ${stage.status}${selectedStage === stage.stage_number ? " selected" : ""}`}
                    onClick={() => setSelectedStage(stage.stage_number)}
                  >
                    <div className="stage-header">
                      <span className="stage-number">Stage {stage.stage_number}</span>
                      <span className={`stage-status-badge ${stage.status}`}>
                        {stage.status}
                      </span>
                    </div>
                    <div className="stage-name">{stage.stage_name}</div>
                    <div className="stage-time">
                      {stage.status === "running"
                        ? `Running ${formatDuration(stage.started_at, null)}`
                        : stage.status === "completed"
                          ? `Done in ${formatDuration(stage.started_at, stage.finished_at)}`
                          : stage.status === "failed"
                            ? `Failed after ${formatDuration(stage.started_at, stage.finished_at)}`
                            : "Waiting"}
                    </div>
                  </div>
                ))}
              </div>

              {/* Stage detail panel */}
              {selectedStageData && (
                <div className="stage-detail-panel">
                  <h4>
                    Stage {selectedStageData.stage_number}: {selectedStageData.stage_name}
                  </h4>
                  <div className="detail-row">
                    <span className="label">Status</span>
                    <span className="value">
                      <span className={`stage-status-badge ${selectedStageData.status}`}>
                        {selectedStageData.status}
                      </span>
                    </span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Started</span>
                    <span className="value">{formatTime(selectedStageData.started_at)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Finished</span>
                    <span className="value">{formatTime(selectedStageData.finished_at)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Duration</span>
                    <span className="value">
                      {formatDuration(selectedStageData.started_at, selectedStageData.finished_at)}
                    </span>
                  </div>
                  {selectedStageData.log_path && (
                    <div className="detail-row">
                      <span className="label">Log</span>
                      <span className="value" style={{ fontFamily: "monospace", fontSize: 12 }}>
                        {selectedStageData.log_path}
                      </span>
                    </div>
                  )}

                  {selectedStageData.error_message && (
                    <div className="error-box">{selectedStageData.error_message}</div>
                  )}

                  <div className="stage-actions">
                    {selectedStageData.status === "failed" && (
                      <button
                        className="primary-btn"
                        onClick={() => handleRetryStage(selectedStageData.stage_number)}
                      >
                        ↻ Retry Stage
                      </button>
                    )}
                  </div>
                </div>
              )}
            </>
          ) : pipelineRuns.length === 0 ? (
            <div className="pipeline-empty">
              <h3>No Pipeline Runs Yet</h3>
              <p>
                Start a pipeline run to process a property scan through the 10-stage
                DAG: ingest → pose QA → point cloud → segmentation → planes → proxy →
                G-buffer → IBL → navigation → publish.
              </p>
              <button
                className="primary-btn"
                style={{ marginTop: 16 }}
                onClick={handleStartRun}
                disabled={pipelineLoading}
              >
                {pipelineLoading ? "Starting..." : "▶ Start First Pipeline Run"}
              </button>
            </div>
          ) : (
            <div className="overlay">Loading run details...</div>
          )}
        </div>
      )}

      {/* ── Catalog Tab ────────────────────────────────────────────── */}
      {activeTab === "catalog" && (
        <div className="catalog-container">
          <div className="catalog-toolbar">
            <div className="category-pills">
              {["all", "sofas", "chairs", "tables", "beds", "lighting", "decor"].map((cat) => (
                <button
                  key={cat}
                  className={catalogCategory === cat ? "pill active" : "pill"}
                  onClick={() => setCatalogCategory(cat)}
                >
                  {cat.toUpperCase()}
                </button>
              ))}
            </div>
            <div className="search-box">
              <input
                type="text"
                placeholder="Search SKU, name, or brand..."
                value={catalogSearch}
                onChange={(e) => setCatalogSearch(e.target.value)}
              />
            </div>
          </div>

          {catalogLoading ? (
            <div className="overlay">Loading catalog...</div>
          ) : catalogItems.length === 0 ? (
            <div className="empty-catalog">
              <h3>No catalog assets found</h3>
              <p>Upload a GLB asset to run the normalization pipeline (meters scale, bottom-center pivot, Draco/KTX2, LODs).</p>
              <button className="primary-btn" onClick={() => setShowUploadModal(true)}>
                Upload First Asset
              </button>
            </div>
          ) : (
            <div className="asset-grid">
              {catalogItems.map((item) => (
                <div key={item.id} className="asset-card">
                  <div className="thumb-container">
                    <img
                      src={item.thumbnail_url || "/data/TEST-001/thumbnail.png"}
                      alt={item.product_name}
                      onError={(e) => {
                        (e.target as HTMLImageElement).src =
                          "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%23667'><text x='20' y='50'>GLB</text></svg>";
                      }}
                    />
                    <span className="category-badge">{item.category}</span>
                  </div>
                  <div className="asset-info">
                    <div className="asset-header">
                      <h4>{item.product_name}</h4>
                      <span className="sku">{item.sku}</span>
                    </div>
                    <p className="manufacturer">{item.manufacturer}</p>

                    <div className="asset-meta-grid">
                      {item.dimensions_mm && (
                        <span>
                          <b>Dimensions:</b> {item.dimensions_mm.x}×{item.dimensions_mm.y}×
                          {item.dimensions_mm.z} mm
                        </span>
                      )}
                      {item.polygon_count !== undefined && (
                        <span>
                          <b>Polys:</b> {item.polygon_count?.toLocaleString() || "n/a"} tris
                        </span>
                      )}
                      {item.price !== null && item.price !== undefined && (
                        <span className="price">
                          ${Number(item.price).toFixed(2)} {item.currency || "USD"}
                        </span>
                      )}
                    </div>

                    <div className="asset-actions">
                      {item.glb_url && (
                        <a href={item.glb_url} download className="btn-secondary">
                          LOD0 GLB
                        </a>
                      )}
                      <button
                        className="btn-danger"
                        onClick={() => handleDeleteItem(item.id, item.sku)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {showUploadModal && (
            <div className="modal-backdrop">
              <div className="modal">
                <div className="modal-header">
                  <h3>Upload &amp; Normalize Catalog Asset</h3>
                  <button className="close-btn" onClick={() => setShowUploadModal(false)}>
                    ×
                  </button>
                </div>
                <form onSubmit={handleUploadSubmit} className="upload-form">
                  <label>
                    Select GLB / glTF File:
                    <input
                      type="file"
                      accept=".glb,.gltf"
                      required
                      onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    />
                  </label>

                  <div className="form-row">
                    <label>
                      SKU Code:
                      <input
                        type="text"
                        placeholder="SOFA-MOD-001"
                        required
                        value={sku}
                        onChange={(e) => setSku(e.target.value)}
                      />
                    </label>
                    <label>
                      Category:
                      <select value={category} onChange={(e) => setCategory(e.target.value)}>
                        <option value="sofas">Sofas</option>
                        <option value="chairs">Chairs</option>
                        <option value="tables">Tables</option>
                        <option value="beds">Beds</option>
                        <option value="lighting">Lighting</option>
                        <option value="decor">Decor</option>
                      </select>
                    </label>
                  </div>

                  <div className="form-row">
                    <label>
                      Product Name:
                      <input
                        type="text"
                        placeholder="Nordic 3-Seater Sofa"
                        required
                        value={productName}
                        onChange={(e) => setProductName(e.target.value)}
                      />
                    </label>
                    <label>
                      Manufacturer:
                      <input
                        type="text"
                        value={manufacturer}
                        onChange={(e) => setManufacturer(e.target.value)}
                      />
                    </label>
                  </div>

                  <label>
                    Price (USD):
                    <input
                      type="number"
                      step="0.01"
                      value={price}
                      onChange={(e) => setPrice(e.target.value)}
                    />
                  </label>

                  <div className="modal-footer">
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => setShowUploadModal(false)}
                    >
                      Cancel
                    </button>
                    <button type="submit" className="primary-btn" disabled={uploading}>
                      {uploading ? "Normalizing..." : "Start Normalization"}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Pose QA Tab ────────────────────────────────────────────── */}
      {activeTab === "qa" && (
        <div className="layout">
          {qaError ? (
            <div className="overlay error">{qaError}</div>
          ) : !report ? (
            <div className="overlay">Loading pose QA report…</div>
          ) : (
            <>
              <aside>
                <div className="filters">
                  {(
                    [
                      ["flagged", "Flagged"],
                      ["low_score", "Score"],
                      ["low_margin", "Margin"],
                      ["spot", "Spot"],
                      ["all", "All"],
                    ] as [QAFilter, string][]
                  ).map(([value, label]) => (
                    <button
                      key={value}
                      className={qaFilter === value ? "active" : ""}
                      onClick={() => setQaFilter(value)}
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
                          <span className="badge flag" title="low_score">
                            S
                          </span>
                        )}
                        {hasReason(sweep, "low_margin") && (
                          <span className="badge flag" title="low_margin">
                            M
                          </span>
                        )}
                      </button>
                    </li>
                  ))}
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
                            return (
                              <td key={face}>
                                {value !== undefined ? value.toFixed(3) : "—"}
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
                          Columns: panorama · OBJ render (normalized depth, oriented) · panorama with
                          rendered edges in red.
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
            </>
          )}
        </div>
      )}
    </div>
  );
}

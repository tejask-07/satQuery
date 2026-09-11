import type { QueryResponse } from "../../api/query";
import "./ResultsInsights.css";

export interface ResultsInsightsProps {
  result?: QueryResponse | null;
  onBack?: () => void;
  onViewLayers?: () => void;
  onNewAnalysis?: () => void;
}

type RecordValue = Record<string, unknown>;
interface Artifact { url: string; label: string; }

const asRecord = (value: unknown): RecordValue =>
  value && typeof value === "object" && !Array.isArray(value) ? value as RecordValue : {};
const textValue = (value: unknown): string | null => {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return null;
};
const numberValue = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const formatNumber = (value: number, digits = 2) => value.toFixed(digits);

const artifactLabels: Record<string, string> = {
  change_map: "Change Map",
  before: "Before Scene",
  after: "After Scene",
  optical: "Optical RGB",
  s1_vv: "SAR VV",
  s1_vh: "SAR VH",
  s1_composite: "SAR Composite",
};

const formatArtifactLabel = (value: string) =>
  artifactLabels[value.toLowerCase()] ||
  value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());

function ResultsInsights({ result, onBack, onViewLayers, onNewAnalysis }: ResultsInsightsProps) {
  if (!result) {
    return (
      <main className="results-workspace">
        <header className="results-page-header">
          <div className="results-header-left">
            {onBack && <button type="button" className="results-nav-button" onClick={onBack}>← ANALYSIS</button>}
            <h1>RESULTS &amp; INSIGHTS</h1>
          </div>
        </header>
        <div className="results-empty-state">
          <p>No analysis results are currently loaded.</p>
          {onNewAnalysis && <button type="button" className="download-report-button" onClick={onNewAnalysis}>START A QUERY →</button>}
        </div>
      </main>
    );
  }

  const statistics = asRecord(result.statistics);
  const plan = result.plan || {};
  const task = textValue(plan.task) || "N/A";
  const taskName = task.toLowerCase();
  const isVqa = taskName === "single_image_vqa";
  const isChange = statistics.changed_pixels !== undefined || statistics.mean_change !== undefined || statistics.change_ratio !== undefined;
  const metric = (textValue(statistics.metric) || textValue(plan.metric) || "").toUpperCase();
  const confidence = numberValue(result.confidence);
  const hasConfidence = confidence !== null && confidence >= 0 && confidence <= 1;
  const confidencePct = hasConfidence ? Math.round(confidence * 100) : null;
  const model = asRecord(result.model);
  const executionSummary = asRecord(result.execution_summary);
  const interpretation = asRecord(result.interpretation || statistics.interpretation);
  const spatial = asRecord(result.spatial_analysis || statistics.spatial_analysis);
  const vqa = asRecord(statistics.vqa);

  const resolveUrl = (raw: unknown): string | null => {
    const value = textValue(raw);
    if (!value || value.startsWith("data:")) return null;
    if (/^https?:\/\//i.test(value)) return value;
    const base = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
    return `${base}${value.startsWith("/") ? value : `/${value}`}`;
  };
  const artifacts: Artifact[] = [];
  const addArtifact = (rawUrl: unknown, label: unknown) => {
    const url = resolveUrl(rawUrl);
    const name = textValue(label);
    if (url && name && !artifacts.some((item) => item.url === url)) artifacts.push({ url, label: name });
  };
  for (const layer of result.layers || []) {
    const item = asRecord(layer);
    addArtifact(item.visualization_url || item.url, item.name || item.id || item.type);
  }
  for (const [key, rawUrl] of Object.entries(result.images || {})) addArtifact(rawUrl, formatArtifactLabel(key));
  const evidenceImages = (result.evidence || []).flatMap((item) => {
    const images = asRecord(item).images;
    return Array.isArray(images) ? images : [];
  }).map(asRecord);
  for (const image of evidenceImages) {
    const date = textValue(image.date);
    for (const [kind, rawVisualization] of Object.entries(asRecord(image.visualizations))) {
      const visualization = asRecord(rawVisualization);
      addArtifact(visualization.url || visualization.filename, `${kind.replace(/_/g, " ")}${date ? ` (${date})` : ""}`);
    }
  }
  for (const [group, rawGroup] of Object.entries(asRecord(result.layer_package))) {
    for (const [kind, rawItem] of Object.entries(asRecord(rawGroup))) {
      addArtifact(asRecord(rawItem).url, `${formatArtifactLabel(group)} ${formatArtifactLabel(kind)}`);
    }
  }

  const cloudValues = evidenceImages.map((item) => numberValue(item.cloud_cover) ?? numberValue(asRecord(item.metadata).cloud_cover)).filter((value): value is number => value !== null);
  const cloudCover = cloudValues.length ? (cloudValues[0] <= 1 ? cloudValues[0] * 100 : cloudValues[0]) : null;
  const beforeDate = textValue(evidenceImages[0]?.date) || textValue(plan.time_start);
  const afterDate = textValue(evidenceImages[1]?.date) || textValue(plan.time_end);
  const observationRange = beforeDate || afterDate ? `${beforeDate || "N/A"}${afterDate ? ` → ${afterDate}` : ""}` : null;

  const areaHectares = numberValue(statistics.area_hectares) ?? numberValue(spatial.total_candidate_area_hectares);
  const validPixels = numberValue(statistics.valid_pixels) ?? numberValue(statistics.total_pixels);
  const changedPixels = numberValue(statistics.changed_pixels);
  const meanChange = numberValue(statistics.mean_change);
  const meanValue = numberValue(statistics.mean);
  const areaDisplay = areaHectares === null ? "N/A" : `${formatNumber(areaHectares)} ha`;
  const averageDisplay = isChange ? meanChange === null ? "N/A" : `${meanChange >= 0 ? "+" : ""}${formatNumber(meanChange, 4)}` : meanValue === null ? "N/A" : formatNumber(meanValue, 4);
  const pixelsDisplay = validPixels === null ? "N/A" : `${validPixels} px`;

  const findings: string[] = [];
  if (changedPixels !== null && numberValue(statistics.valid_pixels) !== null) {
    const ratio = numberValue(statistics.change_ratio);
    findings.push(`${metric || "Index"}: ${changedPixels} of ${statistics.valid_pixels} valid pixels changed${ratio === null ? "." : ` (${formatNumber(ratio * 100, 2)}%).`}`);
  }
  const changeType = textValue(statistics.change_type);
  if (changeType) findings.push(`Observed change type: ${changeType.replace(/_/g, " ")}.`);
  if (metric && meanChange !== null) findings.push(`${metric} mean change: ${meanChange >= 0 ? "+" : ""}${formatNumber(meanChange, 4)}.`);
  const candidateCount = numberValue(spatial.region_count);
  if (candidateCount !== null) findings.push(`${candidateCount} spatial candidate regions were identified.`);
  const modalities = Array.isArray(plan.modalities) ? plan.modalities.filter((value): value is string => typeof value === "string") : [];
  if (modalities.length) findings.push(`Modalities used: ${modalities.join(", ")}.`);
  const indicators = [
    ...(Array.isArray(plan.primary_indicators) ? plan.primary_indicators : []),
    ...(Array.isArray(plan.supporting_indicators) ? plan.supporting_indicators : []),
  ].filter((value): value is string => typeof value === "string");
  const trace = Array.isArray(result.execution_trace) ? result.execution_trace : [];
  const summarySteps = Array.isArray(executionSummary.steps) ? executionSummary.steps.map(asRecord) : [];
  const executionSteps = trace.length ? trace : summarySteps.map((step) => `${textValue(step.tool) || "Unknown tool"}${textValue(step.status) ? ` (${step.status})` : ""}`);
  const answer = textValue(result.answer) || textValue(statistics.explanation) || "N/A";
  const metricLabels = isVqa ? ["Area affected", "Average index/change", "Pixels analyzed", "Model confidence"] : ["Area affected", isChange ? `Avg ${metric || "index"} change` : `Mean ${metric || "index"}`, "Pixels analyzed", "Model confidence"];
  const metricValues = isVqa ? ["N/A", "N/A", "N/A", hasConfidence ? `${confidencePct}%` : "N/A"] : [areaDisplay, averageDisplay, pixelsDisplay, hasConfidence ? `${confidencePct}%` : "N/A"];

  return (
    <main className="results-workspace">
      <header className="results-page-header">
        <div className="results-header-left">
          <div className="results-nav-actions">
            {onBack && <button type="button" className="results-nav-button" onClick={onBack}>← ANALYSIS</button>}
            {onViewLayers && <button type="button" className="results-nav-button accent" onClick={onViewLayers}>LAYERS VIEW →</button>}
            {onNewAnalysis && <button type="button" className="results-nav-button" onClick={onNewAnalysis}>NEW QUERY ↺</button>}
          </div>
          <h1>RESULTS &amp; INSIGHTS</h1>
        </div>
        <div className="results-header-meta"><div className="results-page-number">04.</div><div>STATUS: {(result.status || "N/A").toUpperCase()}</div></div>
      </header>

      <section className="results-card">
        <div className="results-card-header">RESULT SUMMARY — {task.toUpperCase()}</div>
        <div className="results-left">
          <section className="results-metrics"><div className="results-section-label">DETERMINISTIC MEASUREMENTS</div><div className="results-metric-grid">{metricValues.map((value, index) => <div className="results-metric" key={metricLabels[index]}><div className="results-metric-value">{value}</div><div className="results-metric-label">{metricLabels[index]}</div></div>)}</div></section>
          <section className="results-highlights"><div className="results-section-label">KEY FINDINGS &amp; HIGHLIGHTS</div>{isVqa ? <div className="highlight-item"><span className="highlight-dot red" /><p>Visual question answering result is shown in the interpretation panel.</p></div> : findings.length ? findings.map((finding, index) => <div className="highlight-item" key={finding}><span className={`highlight-dot ${index % 3 === 0 ? "red" : index % 3 === 1 ? "orange" : "yellow"}`} /><p>{finding}</p></div>) : <div className="highlight-item"><span className="highlight-dot yellow" /><p>No structured findings were returned by the backend.</p></div>}</section>
          {artifacts.length > 0 && <section className="results-evidence"><div className="results-section-label">OBSERVATIONAL EVIDENCE &amp; ARTIFACTS</div><div className="evidence-grid">{artifacts.map((artifact) => <div className="evidence-item" key={artifact.url}><div className="evidence-image"><img src={artifact.url} alt={artifact.label} /></div><div className="evidence-label">{artifact.label}</div></div>)}</div></section>}
          {executionSteps.length > 0 && <section className="execution-summary-section"><div className="results-section-label">EXECUTION AUDIT &amp; SELECTED TOOLS</div><div className="execution-step-grid">{executionSteps.map((step, index) => <div key={`${step}-${index}`} className="execution-step-chip completed"><span>⚡</span><span>{step}</span></div>)}</div>{textValue(model.name) && <div className="execution-model-badge">Model: <strong>{textValue(model.name)}</strong></div>}</section>}
        </div>

        <div className="results-right">
          <section className="confidence-section"><div className="results-section-label">CONFIDENCE ASSESSMENT</div><div className={`confidence-ring ${!hasConfidence ? "unavailable" : ""}`} style={hasConfidence ? { background: `conic-gradient(#277f8a 0deg ${(confidencePct || 0) * 3.6}deg, #dce2d9 ${(confidencePct || 0) * 3.6}deg 360deg)` } : undefined}><div className="confidence-ring-inner"><span className={!hasConfidence ? "unavailable" : ""}>{hasConfidence ? `${confidencePct}%` : "N/A"}</span></div></div><div className={`confidence-status ${!hasConfidence ? "unavailable" : ""}`}>{hasConfidence ? textValue(model.name) ? `Model confidence (${textValue(model.name)})` : "Model confidence" : "Not provided for this analysis."}</div></section>
          {(observationRange || cloudCover !== null || indicators.length > 0 || candidateCount !== null) && <section className="based-on-section"><div className="results-section-label">SCIENTIFIC GROUNDS &amp; PROVENANCE</div>{observationRange && <div className="check-item"><span>Observation range: {observationRange}</span></div>}{cloudCover !== null && <div className="check-item"><span>Cloud cover: {formatNumber(cloudCover, 1)}%</span></div>}{indicators.length > 0 && <div className="check-item"><span>Indicators: {indicators.join(", ")}</span></div>}{candidateCount !== null && <div className="check-item"><span>Spatial candidates: {candidateCount} regions</span></div>}</section>}
          <section className="interpretation-section"><div className="results-section-label">{isVqa ? "VQA ANSWER" : "INTERPRETATION"}</div>{isVqa && textValue(vqa.question) && <div className="interpretation-question">Question: {textValue(vqa.question)}</div>}<p>{answer}</p>{isVqa && textValue(vqa.scene_date) && <div className="interpretation-meta">Scene date: {textValue(vqa.scene_date)}</div>}{textValue(interpretation.summary) && textValue(interpretation.summary) !== answer && <p>{textValue(interpretation.summary)}</p>}</section>
          <button type="button" className="download-report-button" onClick={() => window.print()}><span>PRINT / SAVE REPORT</span><span>↓</span></button>
        </div>
      </section>
    </main>
  );
}

export default ResultsInsights;

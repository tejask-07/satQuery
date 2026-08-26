interface AnalysisPanelProps {
  result: any;
}

function AnalysisPanel({ result }: AnalysisPanelProps) {
  if (!result) {
    return (
      <section className="analysis-panel empty">
        <div className="section-label">AI ANALYSIS</div>

        <div className="empty-message">
          <h2>Ready for analysis</h2>
          <p>
            Submit a natural-language query to see the generated analysis
            plan and results.
          </p>
        </div>
      </section>
    );
  }

  const plan = result.plan;

  return (
    <section className="analysis-panel">
      <div className="section-label">AI ANALYSIS</div>

      <div className="analysis-status">
        <span>{result.status}</span>
      </div>

      <h2>{result.answer}</h2>

      {plan && (
        <div className="plan">
          <div className="plan-row">
            <span>Task</span>
            <strong>{plan.task}</strong>
          </div>

          <div className="plan-row">
            <span>Target</span>
            <strong>{plan.target ?? "—"}</strong>
          </div>

          <div className="plan-row">
            <span>Time</span>
            <strong>
              {plan.time_start ?? "—"} → {plan.time_end ?? "—"}
            </strong>
          </div>

          <div className="plan-row">
            <span>Analysis</span>
            <strong>{plan.analysis?.join(" · ") || "—"}</strong>
          </div>
        </div>
      )}
    </section>
  );
}

export default AnalysisPanel;
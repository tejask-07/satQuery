export interface QueryPlan {
  task: string;
  target: string;
  time_start: string;
  time_end: string;
  modalities: string[];
  intent?: string;
  metric?: string;
  direction?: string;
  analysis: string[];
  output: string[];
  outputs?: string[];
  primary_indicators?: string[];
  supporting_indicators?: string[];
  aoi?: unknown;
}

export interface QueryResponse {
  status?: string;
  answer?: string;
  confidence?: number | null;
  plan: QueryPlan;
  statistics?: Record<string, unknown>;
  layers?: unknown[];
  evidence?: unknown[];
  execution_trace?: string[];
  visualization_url?: string | null;
  classified_visualization_url?: string | null;
  bounds?: [number, number][] | number[][] | null;
  images?: {
    before?: string;
    after?: string;
    change_map?: string;
  };
  layer_package?: Record<string, any>;
  multi_index_evidence?: Record<string, any>;
  evidence_package?: Record<string, unknown>;
  candidates?: Record<string, any>[];
  candidate_package?: Record<string, any>;
  interpretation?: Record<string, any>;
  spatial_analysis?: Record<string, any>;
  temporal_analysis?: Record<string, any>;
  calibration?: Record<string, any>;
  model?: Record<string, any>;
  execution_summary?: Record<string, any>;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function submitQuery(
  query: string,
  aoi?: unknown,
  timeStart?: string,
  timeEnd?: string
): Promise<QueryResponse> {
  const payload: {
    query: string;
    aoi?: unknown;
    time_start?: string;
    time_end?: string;
  } = { query };

  if (aoi) {
    payload.aoi = aoi;
  }
  if (timeStart && timeStart.trim()) {
    payload.time_start = timeStart.trim();
  }
  if (timeEnd && timeEnd.trim()) {
    payload.time_end = timeEnd.trim();
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      let errorDetail = `Query failed with status ${response.status}`;
      try {
        const errJson = await response.json();
        if (Array.isArray(errJson?.detail)) {
          errorDetail = errJson.detail
            .map((d: any) => `${d.loc ? d.loc.join(".") + ": " : ""}${d.msg || JSON.stringify(d)}`)
            .join(", ");
        } else if (errJson?.detail) {
          errorDetail =
            typeof errJson.detail === "object"
              ? (errJson.detail.message || JSON.stringify(errJson.detail))
              : String(errJson.detail);
        } else if (errJson?.message) {
          errorDetail = String(errJson.message);
        }
      } catch {
        // ignore
      }
      throw new Error(errorDetail);
    }

    return await response.json();
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error(
        "Query request timed out after 300 seconds. The satellite imagery search or index calculation is taking longer than expected."
      );
    }
    throw new Error(err.message || "Failed to process query");
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function fetchBenchmarkSummary(): Promise<Record<string, any>> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/benchmark/summary`);
    if (!response.ok) return {};
    return await response.json();
  } catch {
    return {};
  }
}

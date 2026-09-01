export interface QueryPlan {
  task: string;
  target: string;
  time_start: string;
  time_end: string;
  modalities: string[];
  metric?: string;
  direction?: string;
  analysis: string[];
  output: string[];
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
  evidence_package?: Record<string, unknown>;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function submitQuery(
  query: string,
  aoi?: unknown
): Promise<QueryResponse> {
  const payload: { query: string; aoi?: unknown } = { query };
  if (aoi) {
    payload.aoi = aoi;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 90000);

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
        if (errJson?.detail) {
          errorDetail = String(errJson.detail);
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
        "Query request timed out after 90 seconds. The satellite imagery search or index calculation is taking longer than expected."
      );
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}
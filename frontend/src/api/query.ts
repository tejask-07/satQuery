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
}

const API_BASE_URL = "http://127.0.0.1:8000";

export async function submitQuery(
  query: string
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Query failed with status ${response.status}`
    );
  }

  return response.json();
}
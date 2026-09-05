export interface AOIGeometry {
  type:
    | "polygon"
    | "rectangle"
    | "circle";

  coordinates?: [
    number,
    number
  ][];

  center?: [
    number,
    number
  ];

  radius?: number;
}

export interface QueryRequest {
  query: string;
  aoi?: AOIGeometry;
  time_start?: string;
  time_end?: string;
}

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

  layer_package?: Record<string, any>;
  evidence_package?: Record<string, unknown>;
  interpretation?: Record<string, any>;
  spatial_analysis?: Record<string, any>;
  temporal_analysis?: Record<string, any>;
  calibration?: Record<string, any>;
}

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

export async function submitQuery(
  query: string,
  options?: {
    aoi?: AOIGeometry;
    time_start?: string;
    time_end?: string;
  }
): Promise<QueryResponse> {
  const payload: QueryRequest = {
    query,
    ...(options ?? {}),
  };

  const controller =
    new AbortController();

  const timeoutId = setTimeout(
    () => controller.abort(),
    300000
  );

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/query`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify(
          payload
        ),

        signal: controller.signal,
      }
    );

    if (!response.ok) {
      let errorDetail =
        `Query failed with status ${response.status}`;

      try {
        const errorBody =
          await response.json();

        if (errorBody?.detail) {
          errorDetail =
            typeof errorBody.detail ===
            "string"
              ? errorBody.detail
              : JSON.stringify(
                  errorBody.detail
                );
        }
      } catch {
        // Keep default error message.
      }

      throw new Error(
        "Query request timed out after 5 minutes. The satellite imagery analysis is taking longer than expected."
      );
    }

    return await response.json();
  } catch (err: any) {
    if (
      err?.name ===
      "AbortError"
    ) {
      throw new Error(
        "Query request timed out after 90 seconds. The satellite imagery search or index calculation is taking longer than expected."
      );
    }

    throw new Error(
      err?.message ||
        "Failed to process query"
    );
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function fetchBenchmarkSummary(): Promise<
  Record<string, any>
> {
  try {
    const response =
      await fetch(
        `${API_BASE_URL}/api/benchmark/summary`
      );

    if (!response.ok) {
      return {};
    }

    return await response.json();
  } catch {
    return {};
  }
}
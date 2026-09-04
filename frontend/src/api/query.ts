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

  start_date?: string;

  end_date?: string;
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
}

export interface QueryResponse {
  status?: string;

  answer?: string;

  confidence?: number | null;

  plan: QueryPlan;

  statistics?: Record<
    string,
    unknown
  >;

  layers?: unknown[];

  evidence?: unknown[];

  execution_trace?: string[];
}

const API_BASE_URL =
  "http://127.0.0.1:8000";

/* =========================================================
   SUBMIT QUERY
   ========================================================= */

export async function submitQuery(
  query: string,
  options?: {
    aoi?: AOIGeometry;
    start_date?: string;
    end_date?: string;
  }
): Promise<QueryResponse> {
  const payload: QueryRequest = {
    query,
    ...(options ?? {}),
  };

  const response =
    await fetch(
      `${API_BASE_URL}/api/query`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            payload
          ),
      }
    );

  if (!response.ok) {
    let message =
      `Query failed with status ${response.status}`;

    try {
      const errorBody =
        await response.json();

      if (
        errorBody?.detail
      ) {
        message =
          typeof errorBody.detail ===
          "string"
            ? errorBody.detail
            : JSON.stringify(
                errorBody.detail
              );
      }
    } catch {
      // Keep the default error message.
    }

    throw new Error(
      message
    );
  }

  return response.json();
}
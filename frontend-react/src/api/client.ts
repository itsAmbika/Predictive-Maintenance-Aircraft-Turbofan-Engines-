import type { FleetPredictionResponse, ModelInfo } from "../types";

// Dev server proxies /api -> http://localhost:8000 (see vite.config.ts).
// In production this is served by the same FastAPI process, so a relative
// path works unmodified in both cases.
const API_BASE = "";

export class ApiError extends Error {}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/api/model-info`);
  if (!res.ok) throw new ApiError(`model-info failed: ${res.statusText}`);
  return res.json();
}

export async function fetchHealth(): Promise<{ status: string; subset: string; model_name: string }> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new ApiError(`health check failed: ${res.statusText}`);
  return res.json();
}

export async function uploadSensorFile(file: File): Promise<FleetPredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/predict/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.detail || res.statusText);
  }
  return res.json();
}

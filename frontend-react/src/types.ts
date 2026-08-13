export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface HealthPoint {
  cycle: number;
  health_indicator: number;
}

export interface ShapFactor {
  feature: string;
  impact: number;
  direction: "raises_rul" | "lowers_rul";
}

export interface EnginePrediction {
  unit_number: number;
  last_cycle: number;
  predicted_rul: number;
  rul_low: number | null;
  rul_high: number | null;
  health_score: number;
  risk: RiskLevel;
  risk_action: string;
  fail_within_20_proba: number | null;
  top_factors: ShapFactor[];
  health_trend: HealthPoint[];
}

export interface FleetPredictionResponse {
  subset: string;
  n_engines: number;
  engines: EnginePrediction[];
}

export interface ModelInfo {
  subset: string;
  model_name: string;
  n_features: number;
  metrics: {
    MAE: number;
    RMSE: number;
    R2: number;
    NASA_score: number;
  };
}

export type SortKey = "unit_number" | "last_cycle" | "predicted_rul" | "health_score" | "risk";
export type SortDir = "asc" | "desc";
export type RiskFilter = "ALL" | RiskLevel;

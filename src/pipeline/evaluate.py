"""Stage 4 -- score the persisted best model on the untouched official test set.

Two metric sets are reported, because they answer different questions and are
easy to confuse:

  * ``official`` -- the *last* cycle of each test engine vs. RUL_FD00x.txt. This
    is how the PHM08 challenge scores a submission (100 engines for FD001).
  * ``all_rows`` -- every test row. Optimistic by comparison, since most rows sit
    early in a trajectory where RUL is easy.

The configured quality gate is applied to ``official``: a model that fails it is
not eligible for registry promotion.

    uv run python -m src.pipeline.evaluate
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from omegaconf import DictConfig

from src import evaluate as ev
from src import tracking
from src.config import load_config, resolve, set_seeds


def _gate(metrics: dict, gate_cfg) -> tuple[bool, list[str]]:
    failures = []
    if metrics["MAE"] > gate_cfg.max_mae:
        failures.append(f"MAE {metrics['MAE']:.2f} > {gate_cfg.max_mae}")
    if metrics["RMSE"] > gate_cfg.max_rmse:
        failures.append(f"RMSE {metrics['RMSE']:.2f} > {gate_cfg.max_rmse}")
    if metrics["R2"] < gate_cfg.min_r2:
        failures.append(f"R2 {metrics['R2']:.3f} < {gate_cfg.min_r2}")
    return not failures, failures


def evaluate(cfg: DictConfig) -> dict:
    subset = cfg.subset
    processed = resolve(cfg, "processed")
    models_dir = resolve(cfg, "models")
    reports = resolve(cfg, "reports")

    model = joblib.load(models_dir / f"best_model_{subset}.joblib")
    meta = joblib.load(models_dir / f"best_model_{subset}_meta.joblib")
    feature_cols = meta["feature_cols"]

    test = pd.read_parquet(processed / f"test_{subset}_features.parquet")
    y_all = test[cfg.target.name]
    pred_all = model.predict(test[feature_cols])

    last = test.sort_values(["unit_number", "cycle"]).groupby("unit_number", as_index=False).tail(1)
    y_last = last[cfg.target.name]
    pred_last = model.predict(last[feature_cols])

    official = ev.regression_report(y_last, pred_last)
    all_rows = ev.regression_report(y_all, pred_all)
    by_bin = ev.evaluate_by_rul_bin(y_last, pred_last)
    passed, failures = _gate(official, cfg.evaluation.gate)

    payload = {
        "subset": subset,
        "model_name": meta["model_name"],
        "trained_at": meta.get("trained_at"),
        "git_sha": meta.get("git_sha"),
        "official_test_last_cycle": official,
        "all_test_rows": all_rows,
        "gate_passed": passed,
        "gate_failures": failures,
    }
    (reports / f"metrics_{subset}.json").write_text(json.dumps(payload, indent=2, default=float))
    by_bin.to_csv(reports / f"error_by_rul_bin_{subset}.csv")

    with tracking.run(cfg, stage="evaluate") as run:
        mlflow = tracking.setup(cfg)
        mlflow.set_tags({"model_name": meta["model_name"], "train_run_id": meta.get("mlflow_run_id") or "unknown"})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in official.items()})
        mlflow.log_metrics({f"test_allrows_{k}": float(v) for k, v in all_rows.items()})
        mlflow.log_metric("gate_passed", int(passed))
        mlflow.log_artifact(str(reports / f"metrics_{subset}.json"))
        mlflow.log_artifact(str(reports / f"error_by_rul_bin_{subset}.csv"))
        payload["eval_run_id"] = run.info.run_id

    print(
        f"[evaluate] {meta['model_name']} on official test set ({official['n']} engines): "
        f"MAE={official['MAE']:.2f} RMSE={official['RMSE']:.2f} R2={official['R2']:.3f} "
        f"NASA={official['NASA_score']:.0f}"
    )
    print(f"[evaluate] quality gate: {'PASS' if passed else 'FAIL -- ' + '; '.join(failures)}")
    return payload


def main() -> None:
    cfg = load_config(cli_overrides=True)
    set_seeds(cfg.project.seed)
    result = evaluate(cfg)
    raise SystemExit(0 if result["gate_passed"] else 1)


if __name__ == "__main__":
    main()

"""Stage 6 -- promote the trained model into the MLflow Model Registry.

Only runs if the evaluate stage's quality gate passed, so a worse model can't
silently take over the ``champion`` alias. The Model Registry needs a
database-backed tracking server (see ``docker-compose.yml``); against a local
file store this stage warns and exits cleanly rather than failing the pipeline.

    uv run python -m src.pipeline.register mlflow.register=true
"""

from __future__ import annotations

import json

import joblib
from omegaconf import DictConfig

from src import tracking
from src.config import load_config, resolve


def register(cfg: DictConfig) -> dict | None:
    subset = cfg.subset
    reports = resolve(cfg, "reports")
    models_dir = resolve(cfg, "models")

    metrics_path = reports / f"metrics_{subset}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} missing -- run `python -m src.pipeline.evaluate` first")
    metrics = json.loads(metrics_path.read_text())
    if not metrics["gate_passed"]:
        print(f"[register] quality gate failed ({'; '.join(metrics['gate_failures'])}) -- not registering")
        return None

    meta = joblib.load(models_dir / f"best_model_{subset}_meta.joblib")
    run_id = meta.get("mlflow_run_id")
    if not run_id:
        raise ValueError("model meta has no mlflow_run_id -- retrain with `python -m src.pipeline.train`")

    mlflow = tracking.setup(cfg)
    uri = mlflow.get_tracking_uri()
    if uri.startswith("file:") or uri.startswith("/") or uri[1:3] == ":\\":
        print(f"[register] tracking store {uri} is a file store, which has no Model Registry.")
        print("[register] start the MLflow server (docker compose up mlflow) and set MLFLOW_TRACKING_URI.")
        return None

    name = f"{cfg.mlflow.registered_model_name}-{subset.lower()}"
    # The train stage logs each candidate as a nested run; the champion's model
    # artifact lives under the nested run whose model_name tag matches.
    client = mlflow.tracking.MlflowClient()
    children = client.search_runs(
        experiment_ids=[client.get_run(run_id).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}' and tags.model_name = '{meta['model_name']}'",
        max_results=1,
    )
    source_run = children[0].info.run_id if children else run_id

    version = mlflow.register_model(f"runs:/{source_run}/model", name)
    client.set_registered_model_alias(name, "champion", version.version)
    client.set_model_version_tag(name, version.version, "subset", subset)
    client.set_model_version_tag(name, version.version, "test_mae", f"{metrics['official_test_last_cycle']['MAE']:.3f}")

    print(f"[register] {name} v{version.version} registered and aliased as @champion")
    return {"name": name, "version": version.version}


def main() -> None:
    cfg = load_config(cli_overrides=True)
    register(cfg)


if __name__ == "__main__":
    main()

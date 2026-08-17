"""Run the whole DAG end to end: raw data -> registered model.

    uv run python -m src.pipeline.run_all
    uv run python -m src.pipeline.run_all subset=FD003

Stages run in dependency order and share one config, so an override like
``target.train_on=RUL_capped`` reaches every stage that cares about it.
"""

from __future__ import annotations

import time

from src.config import load_config, set_seeds
from src.pipeline import build_features, evaluate, prepare_data, register, serving_extras, train


def main() -> None:
    cfg = load_config(cli_overrides=True)
    set_seeds(cfg.project.seed)

    started = time.perf_counter()
    prepare_data.prepare(cfg)
    build_features.build(cfg)
    train.train(cfg)
    result = evaluate.evaluate(cfg)
    serving_extras.fit_extras(cfg)
    if cfg.mlflow.register:
        register.register(cfg)

    print(f"[run_all] finished in {time.perf_counter() - started:.1f}s")
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Reproducible pipeline stages, one module per stage.

    prepare_data    data/raw            -> data/interim
    build_features  data/interim        -> data/processed + artifacts/scalers
    train           data/processed      -> models/ + reports/model_leaderboard.csv
    evaluate        models/             -> reports/metrics_<subset>.json (official test set)
    serving_extras  data/processed      -> failure-probability + RUL-interval models
    register        models/             -> MLflow Model Registry

Every stage is runnable on its own (`python -m src.pipeline.<stage>`) and the
whole DAG runs with `python -m src.pipeline.run_all`. The notebooks stay as
narrative/EDA; these modules are the source of truth for artifacts.
"""

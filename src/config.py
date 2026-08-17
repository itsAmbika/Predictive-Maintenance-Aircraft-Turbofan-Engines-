"""Typed configuration loading for the whole project.

`conf/config.yaml` is the single source of truth for every parameter that can
change a model: split sizes, feature windows, hyperparameters, the RUL cap, the
serving thresholds. Code reads them from here rather than hardcoding literals,
so training and serving can never disagree about, say, which lags exist.

Usage from a pipeline stage:

    from src.config import load_config
    cfg = load_config()                      # conf/config.yaml
    cfg = load_config(cli_overrides=True)    # + `key=value` args from sys.argv

CLI overrides use OmegaConf dotlist syntax:

    uv run python -m src.pipeline.train subset=FD002 models.xgboost.max_depth=8
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


# ---------------------------------------------------------------------------
# Schema -- OmegaConf validates the YAML against these dataclasses, so a typo
# like `featurs.lags` fails loudly at load time instead of silently doing nothing.
# ---------------------------------------------------------------------------


@dataclass
class ProjectCfg:
    name: str = "aircraft-rul-prognostics"
    seed: int = 42


@dataclass
class PathsCfg:
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    artifacts: str = "artifacts/scalers"
    models: str = "models"
    reports: str = "reports"


@dataclass
class SplitCfg:
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class FeaturesCfg:
    lags: list[int] = field(default_factory=lambda: [1, 2, 3])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    ema_spans: list[int] = field(default_factory=lambda: [5, 10])
    diff_periods: list[int] = field(default_factory=lambda: [1])
    near_constant_std_threshold: float = 1e-3


@dataclass
class TargetCfg:
    name: str = "RUL"
    cap: int | None = 125
    train_on: str = "RUL"


@dataclass
class ModelsCfg:
    candidates: list[str] = field(default_factory=lambda: ["xgboost"])
    selection_metric: str = "MAE"
    selection_direction: str = "min"
    linear: dict[str, Any] = field(default_factory=dict)
    decision_tree: dict[str, Any] = field(default_factory=dict)
    random_forest: dict[str, Any] = field(default_factory=dict)
    xgboost: dict[str, Any] = field(default_factory=dict)
    lightgbm: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServingExtrasCfg:
    failure_thresholds: list[int] = field(default_factory=lambda: [10, 20, 30])
    quantile_low: float = 0.1
    quantile_high: float = 0.9
    quantile_params: dict[str, Any] = field(default_factory=dict)
    classifier_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateCfg:
    max_mae: float = 30.0
    max_rmse: float = 40.0
    min_r2: float = 0.65


@dataclass
class EvaluationCfg:
    gate: GateCfg = field(default_factory=GateCfg)


@dataclass
class MLflowCfg:
    tracking_uri: str | None = None
    experiment: str = "aircraft-rul-prognostics"
    registered_model_name: str = "aircraft-rul"
    register: bool = False


@dataclass
class ServingCfg:
    rul_healthy: float = 125
    risk_high_below: float = 30
    risk_medium_below: float = 60
    failure_horizon: int = 20
    shap_top_n: int = 5


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    subset: str = "FD001"
    paths: PathsCfg = field(default_factory=PathsCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    target: TargetCfg = field(default_factory=TargetCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    serving_extras: ServingExtrasCfg = field(default_factory=ServingExtrasCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    mlflow: MLflowCfg = field(default_factory=MLflowCfg)
    serving: ServingCfg = field(default_factory=ServingCfg)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
    cli_overrides: bool = False,
) -> DictConfig:
    """Load conf/config.yaml, validated against the schema above.

    ``overrides`` / ``cli_overrides`` accept OmegaConf dotlist entries
    (``models.xgboost.max_depth=8``). CLI args that are not ``key=value`` pairs
    are ignored, so this composes with argparse-free stage entrypoints.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(path))

    dotlist: list[str] = list(overrides or [])
    if cli_overrides:
        dotlist += [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    # An env override is handy in containers where editing YAML is awkward.
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri and cfg.mlflow.tracking_uri is None:
        cfg.mlflow.tracking_uri = env_uri

    return cfg  # type: ignore[return-value]


def resolve(cfg: DictConfig, key: str) -> Path:
    """Absolute path for one of ``cfg.paths`` entries, created if missing."""
    p = PROJECT_ROOT / str(cfg.paths[key])
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seeds(seed: int) -> None:
    """Seed every RNG the pipeline touches, so a re-run reproduces the artifacts."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is only needed by the sequence models
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dep today, optional tomorrow
        pass


def config_to_dict(cfg: DictConfig) -> dict:
    """Plain nested dict -- for joblib manifests and MLflow params."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

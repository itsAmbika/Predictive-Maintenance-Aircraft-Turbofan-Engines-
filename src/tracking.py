"""Thin MLflow wrapper.

Keeps every stage's tracking code to two lines and keeps MLflow an optional
import: the API image is built without the dev dependency group, so nothing in
``api/`` may import this module.

Tracking URI resolution order: ``cfg.mlflow.tracking_uri`` -> ``MLFLOW_TRACKING_URI``
-> a local ``mlruns/`` file store, so a fresh clone gets experiment tracking with
zero setup. The file store has no Model Registry (and MLflow 3 considers it
legacy); point ``MLFLOW_TRACKING_URI`` at the tracking server from
``docker-compose.yml`` when you want the registry.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from omegaconf import DictConfig

from src.config import PROJECT_ROOT, config_to_dict


def git_sha() -> str | None:
    """Short commit the run was produced from -- logged so a model traces back to code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def setup(cfg: DictConfig):
    """Point MLflow at the configured store + experiment. Returns the mlflow module."""
    import mlflow

    uri = str(cfg.mlflow.tracking_uri or f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}")
    if uri.startswith("file:"):
        # MLflow 3 refuses the file backend unless you opt in explicitly. It stays
        # the local default here because it needs no server; the registry lives on
        # the compose-hosted server instead.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    if uri.startswith("http"):
        mlflow.set_registry_uri(uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    return mlflow


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v
    return flat


@contextmanager
def run(cfg: DictConfig, stage: str, run_name: str | None = None) -> Iterator[Any]:
    """Start an MLflow run tagged with stage/subset/git sha and the full config."""
    mlflow = setup(cfg)
    with mlflow.start_run(run_name=run_name or f"{stage}-{cfg.subset}") as active:
        mlflow.set_tags(
            {
                "stage": stage,
                "subset": cfg.subset,
                "git_sha": git_sha() or "unknown",
                "project": cfg.project.name,
            }
        )
        params = _flatten(config_to_dict(cfg))
        # MLflow rejects params over 500 chars; long lists are logged as tags instead.
        mlflow.log_params({k: v for k, v in params.items() if len(str(v)) <= 500})
        yield active

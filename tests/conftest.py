"""Shared fixtures.

Tests split into two kinds:

  * pure unit tests -- run anywhere, no artifacts needed (the bulk of the suite,
    and what CI runs on every push)
  * ``@pytest.mark.slow`` -- need built feature tables / trained models on disk,
    i.e. a local `python -m src.pipeline.run_all`. They skip cleanly when the
    artifacts are absent instead of failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src.config import PROJECT_ROOT, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


def make_raw_frame(n_engines: int = 3, n_cycles: int = 40, seed: int = 0) -> pd.DataFrame:
    """A synthetic C-MAPSS-shaped frame: 26 columns, monotonic cycles per engine.

    Sensor values drift with age so that degradation-driven features (rolling
    means, diffs, the PCA health indicator) have something real to pick up.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        life = n_cycles + unit  # engines fail at different ages, as in the real data
        for cycle in range(1, life + 1):
            wear = cycle / life
            row = {"unit_number": unit, "cycle": cycle}
            for i, col in enumerate(dl.OP_SETTING_COLS):
                row[col] = [0.0, 0.0, 100.0][i]
            for j, col in enumerate(dl.SENSOR_COLS):
                base = 500.0 + 10.0 * j
                # sensors 1, 5, 10, 16, 18, 19 are constant in FD001; mimic that
                drift = (
                    0.0
                    if col in {"sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"}
                    else 20.0 * wear
                )
                noise = 0.0 if drift == 0.0 else rng.normal(0, 0.5)
                row[col] = base + drift + noise
            rows.append(row)
    return pd.DataFrame(rows)[dl.ALL_COLS]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture(scope="session")
def raw_cmapss_file(project_root):
    """A small slice of the real test file -- 5 engines, in the on-disk format."""
    path = project_root / "data" / "raw" / "test_FD001.txt"
    if not path.exists():
        pytest.skip("data/raw/test_FD001.txt not available")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=dl.ALL_COLS)
    return df[df["unit_number"] <= 5].reset_index(drop=True)


@pytest.fixture(scope="session")
def artifacts(cfg, project_root):
    """Loaded serving artifacts, or skip if the pipeline hasn't been run here."""
    model_path = project_root / str(cfg.paths.models) / f"best_model_{cfg.subset}.joblib"
    if not model_path.exists():
        pytest.skip("trained artifacts missing -- run `python -m src.pipeline.run_all`")
    from api import inference

    return inference.load_artifacts(cfg.subset, cfg=cfg)

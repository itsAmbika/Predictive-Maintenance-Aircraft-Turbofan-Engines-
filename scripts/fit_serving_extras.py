"""Deprecated shim -- kept so the command in older docs keeps working.

The real implementation is now a pipeline stage, config-driven and MLflow-logged:

    uv run python -m src.pipeline.serving_extras

It fits and persists the two model families notebook 08 computes but never saves:
the per-horizon failure-probability classifiers and the RUL-interval quantile
regressors, both of which the API loads at startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seeds  # noqa: E402
from src.pipeline.serving_extras import fit_extras  # noqa: E402


def main() -> None:
    print("[deprecated] use `python -m src.pipeline.serving_extras` -- running it for you now\n")
    cfg = load_config(cli_overrides=True)
    set_seeds(cfg.project.seed)
    fit_extras(cfg)


if __name__ == "__main__":
    main()

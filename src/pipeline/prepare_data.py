"""Stage 1 -- raw C-MAPSS text files -> interim parquet with the RUL target.

Replaces the hand-run parts of notebooks/01_data_understanding.ipynb.

    uv run python -m src.pipeline.prepare_data
    uv run python -m src.pipeline.prepare_data subset=FD003
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from src import data_loader as dl
from src import preprocessing as pp
from src.config import load_config, resolve, set_seeds


def prepare(cfg: DictConfig) -> dict:
    subset = cfg.subset
    raw_dir = resolve(cfg, "raw")
    interim = resolve(cfg, "interim")

    train_raw = dl.load_train(subset, raw_dir)
    test_raw = dl.load_test(subset, raw_dir)
    rul_raw = dl.load_rul(subset, raw_dir)

    train = pp.add_rul(train_raw)
    # Test trajectories stop before failure; RUL_FD00x.txt supplies what is left
    # after the last observed cycle of each engine.
    test = pp.add_rul_test(test_raw, rul_raw)

    train.to_parquet(interim / f"train_{subset}_with_rul.parquet")
    test.to_parquet(interim / f"test_{subset}_with_rul.parquet")
    rul_raw.to_frame(name="RUL").to_parquet(interim / f"rul_{subset}.parquet")

    stats = {
        "subset": subset,
        "train_rows": int(len(train)),
        "train_engines": int(train["unit_number"].nunique()),
        "test_rows": int(len(test)),
        "test_engines": int(test["unit_number"].nunique()),
        "train_max_rul": int(train["RUL"].max()),
        "median_engine_life": float(train.groupby("unit_number")["cycle"].max().median()),
    }
    (resolve(cfg, "reports") / f"data_stats_{subset}.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    cfg = load_config(cli_overrides=True)
    set_seeds(cfg.project.seed)
    stats = prepare(cfg)
    print(f"[prepare_data] {cfg.subset}: " + ", ".join(f"{k}={v}" for k, v in stats.items() if k != "subset"))
    print(f"[prepare_data] wrote interim parquet to {resolve(cfg, 'interim')}")


if __name__ == "__main__":
    main()

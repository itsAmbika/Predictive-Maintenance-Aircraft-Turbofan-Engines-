"""
Sliding-window sequence construction for the LSTM / GRU models (PHASE 11-12).

Two builders are provided:

- ``build_training_sequences``: every possible window of length ``seq_len``
  within each engine, paired with the RUL at the *last* cycle of that window.
  Used for train/validation.
- ``build_last_window_per_engine``: exactly one window per engine - the most
  recent ``seq_len`` cycles - used for the C-MAPSS test set, which is
  evaluated at a single point in time per engine (matching RUL_FD00x.txt).

Engines shorter than ``seq_len`` are handled by left-padding (repeating the
first observed row) rather than being silently dropped, so that no engine -
including short ones, which are common in FD002/FD004 - disappears from
training or evaluation. A boolean padding mask is returned alongside the
tensors in case you want to mask padded steps in the loss/model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pad_to_length(arr: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad a (T, F) array up to (seq_len, F) by repeating the first row."""
    t = arr.shape[0]
    if t >= seq_len:
        return arr[-seq_len:], np.ones(seq_len, dtype=bool)
    pad_rows = np.repeat(arr[[0]], seq_len - t, axis=0)
    padded = np.concatenate([pad_rows, arr], axis=0)
    mask = np.concatenate([np.zeros(seq_len - t, dtype=bool), np.ones(t, dtype=bool)])
    return padded, mask


def build_training_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    target_col: str = "RUL",
    id_col: str = "unit_number",
    time_col: str = "cycle",
) -> dict:
    """Build every sliding window of length ``seq_len`` per engine.

    Returns a dict with:
        X: float32 array (n_samples, seq_len, n_features)
        y: float32 array (n_samples,)               -- RUL at the window's last cycle
        mask: bool array (n_samples, seq_len)        -- True where the step is real (not padding)
        engine_ids: int array (n_samples,)
        end_cycle: int array (n_samples,)             -- last cycle number in each window
    """
    X_list, y_list, mask_list, id_list, end_cycle_list = [], [], [], [], []

    for engine_id, g in df.sort_values([id_col, time_col]).groupby(id_col):
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_col].to_numpy(dtype=np.float32)
        cycles = g[time_col].to_numpy()
        n = feats.shape[0]

        if n < seq_len:
            # Engine shorter than the requested window: one padded window using all its data.
            padded, mask = _pad_to_length(feats, seq_len)
            X_list.append(padded)
            y_list.append(targets[-1])
            mask_list.append(mask)
            id_list.append(engine_id)
            end_cycle_list.append(cycles[-1])
            continue

        for end in range(seq_len - 1, n):
            start = end - seq_len + 1
            X_list.append(feats[start : end + 1])
            y_list.append(targets[end])
            mask_list.append(np.ones(seq_len, dtype=bool))
            id_list.append(engine_id)
            end_cycle_list.append(cycles[end])

    return {
        "X": np.stack(X_list).astype(np.float32),
        "y": np.array(y_list, dtype=np.float32),
        "mask": np.stack(mask_list),
        "engine_ids": np.array(id_list),
        "end_cycle": np.array(end_cycle_list),
    }


def build_last_window_per_engine(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    id_col: str = "unit_number",
    time_col: str = "cycle",
) -> dict:
    """Build exactly one (most-recent) window per engine - for test-time inference.

    Returns X (n_engines, seq_len, n_features), mask (n_engines, seq_len), and
    engine_ids (n_engines,) in the same order.
    """
    X_list, mask_list, id_list = [], [], []
    for engine_id, g in df.sort_values([id_col, time_col]).groupby(id_col):
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        padded, mask = _pad_to_length(feats, seq_len)
        X_list.append(padded)
        mask_list.append(mask)
        id_list.append(engine_id)

    return {
        "X": np.stack(X_list).astype(np.float32),
        "mask": np.stack(mask_list),
        "engine_ids": np.array(id_list),
    }

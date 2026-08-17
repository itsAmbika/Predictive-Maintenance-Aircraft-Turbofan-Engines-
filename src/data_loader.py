"""
Loading utilities for the NASA C-MAPSS turbofan degradation simulation dataset
(Saxena, Goebel, Simon & Eklund, "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM08, 2008).

Every raw ``train_FD00x.txt`` / ``test_FD00x.txt`` file is whitespace-delimited with
26 columns and no header:

    1  unit number            (engine id)
    2  time, in cycles
    3  operational setting 1  (altitude)
    4  operational setting 2  (Mach number)
    5  operational setting 3  (throttle resolver angle)
    6-26  sensor measurement 1-21

``RUL_FD00x.txt`` has one column: the true remaining useful life (in cycles) of the
*last* cycle present for each engine in the corresponding test file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

ID_COLS = ["unit_number", "cycle"]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]  # sensor_1 ... sensor_21
ALL_COLS = ID_COLS + OP_SETTING_COLS + SENSOR_COLS

# ---------------------------------------------------------------------------
# Physical meaning of each column (from the PHM08 paper, Table 1 & 2).
# Useful for EDA narration and later for SHAP interpretation.
# ---------------------------------------------------------------------------

OP_SETTING_INFO = {
    "op_setting_1": {"symbol": "Altitude", "description": "Flight altitude", "units": "ft"},
    "op_setting_2": {"symbol": "Mach", "description": "Mach number", "units": "-"},
    "op_setting_3": {"symbol": "TRA", "description": "Throttle Resolver Angle", "units": "deg"},
}

SENSOR_INFO = {
    "sensor_1": {"symbol": "T2", "description": "Total temperature at fan inlet", "units": "R"},
    "sensor_2": {"symbol": "T24", "description": "Total temperature at LPC outlet", "units": "R"},
    "sensor_3": {"symbol": "T30", "description": "Total temperature at HPC outlet", "units": "R"},
    "sensor_4": {"symbol": "T50", "description": "Total temperature at LPT outlet", "units": "R"},
    "sensor_5": {"symbol": "P2", "description": "Pressure at fan inlet", "units": "psia"},
    "sensor_6": {"symbol": "P15", "description": "Total pressure in bypass duct", "units": "psia"},
    "sensor_7": {"symbol": "P30", "description": "Total pressure at HPC outlet", "units": "psia"},
    "sensor_8": {"symbol": "Nf", "description": "Physical fan speed", "units": "rpm"},
    "sensor_9": {"symbol": "Nc", "description": "Physical core speed", "units": "rpm"},
    "sensor_10": {"symbol": "epr", "description": "Engine pressure ratio (P50/P2)", "units": "-"},
    "sensor_11": {"symbol": "Ps30", "description": "Static pressure at HPC outlet", "units": "psia"},
    "sensor_12": {"symbol": "phi", "description": "Ratio of fuel flow to Ps30", "units": "pps/psi"},
    "sensor_13": {"symbol": "NRf", "description": "Corrected fan speed", "units": "rpm"},
    "sensor_14": {"symbol": "NRc", "description": "Corrected core speed", "units": "rpm"},
    "sensor_15": {"symbol": "BPR", "description": "Bypass ratio", "units": "-"},
    "sensor_16": {"symbol": "farB", "description": "Burner fuel-air ratio", "units": "-"},
    "sensor_17": {"symbol": "htBleed", "description": "Bleed enthalpy", "units": "-"},
    "sensor_18": {"symbol": "Nf_dmd", "description": "Demanded fan speed", "units": "rpm"},
    "sensor_19": {"symbol": "PCNfR_dmd", "description": "Demanded corrected fan speed", "units": "rpm"},
    "sensor_20": {"symbol": "W31", "description": "HPT coolant bleed", "units": "lbm/s"},
    "sensor_21": {"symbol": "W32", "description": "LPT coolant bleed", "units": "lbm/s"},
}

# ---------------------------------------------------------------------------
# Dataset metadata (from readme.txt)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetInfo:
    train_trajectories: int
    test_trajectories: int
    conditions: int
    fault_modes: int
    description: str


SUBSET_INFO = {
    "FD001": SubsetInfo(100, 100, 1, 1, "One operating condition (sea level), one fault mode (HPC degradation)."),
    "FD002": SubsetInfo(260, 259, 6, 1, "Six operating conditions, one fault mode (HPC degradation)."),
    "FD003": SubsetInfo(
        100, 100, 1, 2, "One operating condition (sea level), two fault modes (HPC + Fan degradation)."
    ),
    "FD004": SubsetInfo(248, 249, 6, 2, "Six operating conditions, two fault modes (HPC + Fan degradation)."),
}


def _read_raw_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)
    return df


def load_train(subset: str, data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load a train_FD00x.txt file. ``subset`` is e.g. 'FD001'."""
    path = Path(data_dir) / f"train_{subset}.txt"
    return _read_raw_txt(path)


def load_test(subset: str, data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load a test_FD00x.txt file (trajectories truncated before failure)."""
    path = Path(data_dir) / f"test_{subset}.txt"
    return _read_raw_txt(path)


def load_rul(subset: str, data_dir: str | Path = "data/raw") -> pd.Series:
    """Load the ground-truth RUL (at the last cycle of each test engine)."""
    path = Path(data_dir) / f"RUL_{subset}.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit_number")
    return rul["RUL"]


def load_subset(subset: str, data_dir: str | Path = "data/raw") -> dict:
    """Convenience loader returning train/test/rul for one subset."""
    return {
        "train": load_train(subset, data_dir),
        "test": load_test(subset, data_dir),
        "rul": load_rul(subset, data_dir),
    }

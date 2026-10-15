"""Shared settings for the experiment scripts.

The scripts import the ``machine_allocation`` package, so install it first
(``pip install -e .`` from the repository root).
"""
from __future__ import annotations

import json
import os
import pathlib
import pickle

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUICK = bool(os.environ.get("QUICK"))          # smoke-test mode: tiny budgets, separate folders
RESULTS = ROOT / ("results_quick" if QUICK else "results")
FIGURES = ROOT / ("figures_quick" if QUICK else "figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_JOBS = 50
JOBS_SEED = 11
PLANT_SEED = 7
PENALTY_PER_H = 60.0          # EUR per weighted hour of lateness
PEN = PENALTY_PER_H / 60.0    # per minute


def budget(x: float) -> float:
    """Deterministic solver time (CP-SAT work units, not seconds)."""
    return 2.0 if QUICK else x


def reps(n: int) -> int:
    return 20 if QUICK else n


MAIN_BUDGET = budget(240.0)   # reference plan
SIDE_BUDGET = budget(40.0)    # default for the other solves


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def save_json(name: str, obj) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=1, default=_default))


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


def save_pickle(name: str, obj) -> None:
    (RESULTS / name).write_bytes(pickle.dumps(obj))


def load_pickle(name: str):
    return pickle.loads((RESULTS / name).read_bytes())


def scalars(d: dict) -> dict:
    return {k: v for k, v in d.items() if not isinstance(v, (np.ndarray, list, dict))}

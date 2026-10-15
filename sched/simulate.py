"""Schedule execution and scoring.

A plan is a dict ``machine -> sequence``. A sequence lists job indices in the order they
run; a planned stop (maintenance, breakdown) is written as ``-(k+1)`` for window ``k`` of
that machine. Execution preserves the order and starts everything as early as possible.
"""
from __future__ import annotations

import numpy as np

from .instance import SETUP_COST_MIN, Instance


def is_window(item: int) -> bool:
    return item < 0


def execute(inst: Instance, seqs: dict, dur: np.ndarray, windows: dict | None = None,
            not_before: np.ndarray | None = None):
    """Run the sequences with durations ``dur[j]`` (minutes on the assigned machine).

    Rules, identical to the constraint model: a changeover occupies the machine and may be
    done before the material arrives; a stop resets the machine to a cold state; a stop
    begins at its planned time, or when the running job finishes if that is later.
    Returns (start, end, setup_minutes_per_job, assign).
    """
    windows = inst.windows if windows is None else windows
    pl = inst.plant
    N = inst.N
    start = np.full(N, -1.0)
    end = np.full(N, -1.0)
    setup = np.zeros(N)
    assign = np.full(N, -1, dtype=int)
    for m, seq in seqs.items():
        t, prev = 0.0, None
        for item in seq:
            if is_window(item):
                a, b = windows[m][-item - 1]
                t = max(t, a) + (b - a)
                prev = None
                continue
            j = item
            f = inst.family[j]
            s = pl.cold[m, f] if prev is None else pl.setup[m, prev, f]
            st = max(t + s, inst.release[j])
            if not_before is not None:
                st = max(st, not_before[j])
            start[j], end[j], setup[j], assign[j] = st, st + dur[j], s, m
            t, prev = end[j], f
    return start, end, setup, assign


def score(inst: Instance, end: np.ndarray, setup: np.ndarray, assign: np.ndarray,
          dur: np.ndarray, penalty_per_min: float = 1.0) -> dict:
    """Cost breakdown of an executed schedule (EUR)."""
    tard = np.maximum(0.0, end - inst.due)
    wt_min = float((inst.weight * tard).sum())
    run = float((inst.plant.cost_h[assign] * dur / 60.0).sum())
    setup_cost = float(setup.sum() * SETUP_COST_MIN)
    return {
        "weighted_tardiness_h": wt_min / 60.0,
        "tardiness_cost": wt_min * penalty_per_min,
        "setup_h": float(setup.sum() / 60.0),
        "setup_cost": setup_cost,
        "run_cost": run,
        "operating_cost": setup_cost + run,
        "total": wt_min * penalty_per_min + setup_cost + run,
        "late_jobs": int((tard > 0).sum()),
        "max_tardiness_h": float(tard.max() / 60.0),
        "makespan_h": float(end.max() / 60.0),
    }


def durations_on_plan(P: np.ndarray, seqs: dict, N: int) -> np.ndarray:
    dur = np.zeros(N)
    for m, seq in seqs.items():
        for item in seq:
            if not is_window(item):
                dur[item] = P[item, m]
    return dur


def evaluate(inst: Instance, seqs: dict, P: np.ndarray, penalty_per_min: float = 1.0,
             windows: dict | None = None, not_before: np.ndarray | None = None) -> dict:
    dur = durations_on_plan(P, seqs, inst.N)
    start, end, setup, assign = execute(inst, seqs, dur, windows, not_before)
    out = score(inst, end, setup, assign, dur, penalty_per_min)
    out.update(start=start, end=end, assign=assign, setup=setup)
    return out


def utilisation(inst: Instance, res: dict) -> np.ndarray:
    """Share of the planning horizon each machine spends running or changing over."""
    u = np.zeros(inst.M)
    for m in range(inst.M):
        idx = res["assign"] == m
        busy = (res["end"][idx] - res["start"][idx]).sum() + res["setup"][idx].sum()
        avail = inst.horizon - sum(b - a for a, b in inst.windows.get(m, []))
        u[m] = busy / avail
    return u

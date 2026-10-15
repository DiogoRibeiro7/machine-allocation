"""Assertions shared by the tests."""
from __future__ import annotations

import numpy as np

from machine_allocation.instance import Instance
from machine_allocation.simulate import evaluate


def assert_valid_plan(inst: Instance, seqs: dict, windows: dict | None = None) -> None:
    """Every job runs exactly once on an eligible machine, and every stop of every machine
    appears exactly once, in time order."""
    windows = inst.windows if windows is None else windows
    jobs = [j for m, seq in seqs.items() for j in seq if j >= 0]
    assert sorted(jobs) == list(range(inst.N))
    for m, seq in seqs.items():
        assert all(inst.eligible[j, m] for j in seq if j >= 0), f"ineligible job on machine {m}"
        stops = [-it - 1 for it in seq if it < 0]
        assert stops == list(range(len(windows.get(m, [])))), f"stops out of order on machine {m}"


def assert_clear_of_stops(inst: Instance, seqs: dict, P: np.ndarray, windows: dict | None = None) -> None:
    """Executed with the planning times, no job overlaps a stop on its machine."""
    windows = inst.windows if windows is None else windows
    ev = evaluate(inst, seqs, P, windows=windows)
    for m, wins in windows.items():
        for a, b in wins:
            on_m = ev["assign"] == m
            overlap = on_m & (ev["start"] < b) & (ev["end"] > a)
            assert not overlap.any(), f"jobs {np.flatnonzero(overlap)} overlap stop {(a, b)} on machine {m}"

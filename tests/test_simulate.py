from __future__ import annotations

import numpy as np
import pytest

from machine_allocation.heuristics import edd_list
from machine_allocation.simulate import evaluate, execute, is_window, score, utilisation


def _two_jobs_on(inst, m):
    """Two jobs of different families that machine m can run."""
    jobs = np.flatnonzero(inst.eligible[:, m])
    j1 = int(jobs[0])
    j2 = int(next(j for j in jobs[1:] if inst.family[j] != inst.family[j1]))
    return j1, j2


def test_is_window():
    assert is_window(-1) and not is_window(0)


def test_execute_applies_cold_start_setups_and_release_dates(week, P_week):
    pl, m = week.plant, 3                                   # M4 has no planned stop
    j1, j2 = _two_jobs_on(week, m)
    dur = P_week[:, m].astype(float)
    start, end, setup, assign = execute(week, {m: [j1, j2]}, dur)
    s1 = max(pl.cold[m, week.family[j1]], week.release[j1])
    s2 = max(s1 + dur[j1] + pl.setup[m, week.family[j1], week.family[j2]], week.release[j2])
    assert (start[j1], end[j1]) == (s1, s1 + dur[j1])
    assert (start[j2], end[j2]) == (s2, s2 + dur[j2])
    assert setup[j1] == pl.cold[m, week.family[j1]]
    assert setup[j2] == pl.setup[m, week.family[j1], week.family[j2]]
    assert assign[j1] == assign[j2] == m
    others = np.setdiff1d(np.arange(week.N), [j1, j2])
    assert (assign[others] == -1).all() and (start[others] == -1).all()


def test_a_stop_resets_the_machine_to_cold(week, P_week):
    pl, m = week.plant, 0                                   # M1 stops at hour 30 for six hours
    (a, b), = week.windows[m]
    j1, j2 = _two_jobs_on(week, m)
    dur = P_week[:, m].astype(float)
    start, end, setup, _ = execute(week, {m: [j1, -1, j2]}, dur)
    resume = max(end[j1], a) + (b - a)
    assert setup[j2] == pl.cold[m, week.family[j2]]
    assert start[j2] == max(resume + setup[j2], week.release[j2])


def test_not_before_delays_starts(week, P_week):
    m = 3
    j1, _ = _two_jobs_on(week, m)
    dur = P_week[:, m].astype(float)
    late = np.full(week.N, 5000)
    start, *_ = execute(week, {m: [j1]}, dur, not_before=late)
    assert start[j1] == 5000


def test_score_adds_up(week, P_week):
    ev = evaluate(week, edd_list(week, P_week), P_week, penalty_per_min=1.5)
    assert ev["operating_cost"] == pytest.approx(ev["setup_cost"] + ev["run_cost"])
    assert ev["total"] == pytest.approx(ev["tardiness_cost"] + ev["operating_cost"])
    assert ev["tardiness_cost"] == pytest.approx(1.5 * 60 * ev["weighted_tardiness_h"])
    assert ev["late_jobs"] == int((ev["end"] > week.due).sum())
    assert ev["makespan_h"] == pytest.approx(ev["end"].max() / 60)


def test_score_of_an_on_time_schedule_has_no_lateness(week):
    end = week.due.astype(float) - 1
    dur = np.ones(week.N)
    sc = score(week, end, np.zeros(week.N), np.zeros(week.N, dtype=int), dur)
    assert sc["late_jobs"] == 0 and sc["tardiness_cost"] == 0
    assert sc["run_cost"] == pytest.approx(week.plant.cost_h[0] * week.N / 60)


def test_utilisation_is_a_share_of_available_time(week, P_week):
    u = utilisation(week, evaluate(week, edd_list(week, P_week), P_week))
    assert u.shape == (week.M,)
    assert ((u > 0) & (u < 1.5)).all()

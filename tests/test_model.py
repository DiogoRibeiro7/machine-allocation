from __future__ import annotations

import numpy as np
import pytest

from helpers import assert_clear_of_stops, assert_valid_plan
from machine_allocation.heuristics import best_baseline
from machine_allocation.model import solve, solve_isolated
from machine_allocation.simulate import evaluate

BUDGET = 10.0          # deterministic work units; the small instance needs well under one


@pytest.fixture(scope="module")
def warm(small, P_small):
    return best_baseline(small, P_small, penalty_per_min=2.5)[0]


@pytest.fixture(scope="module")
def solved(small, P_small, warm):
    return solve(small, P_small, time_limit=BUDGET, penalty_per_min=2.5, hint_seqs=warm)


def test_solver_proves_optimality_on_the_small_instance(solved):
    assert solved["status"] == "OPTIMAL"
    assert solved["bound"] == pytest.approx(solved["objective"])


def test_solver_returns_a_valid_plan(small, P_small, solved):
    assert_valid_plan(small, solved["seqs"])
    assert_clear_of_stops(small, solved["seqs"], P_small)


def test_simulator_agrees_with_the_model_objective(small, P_small, solved):
    """The CP model and the order-preserving simulator price a plan the same way."""
    ev = evaluate(small, solved["seqs"], P_small, penalty_per_min=2.5)
    assert ev["total"] == pytest.approx(solved["objective"])
    np.testing.assert_array_equal(ev["assign"], solved["assign"])


def test_solver_beats_its_warm_start(small, P_small, warm, solved):
    assert solved["objective"] < evaluate(small, warm, P_small, penalty_per_min=2.5)["total"]


def test_solve_is_deterministic(small, P_small, warm, solved):
    again = solve(small, P_small, time_limit=BUDGET, penalty_per_min=2.5, hint_seqs=warm)
    assert again["seqs"] == solved["seqs"]


def test_exhausted_budget_hands_back_the_warm_start(small, P_small, warm):
    res = solve(small, P_small, time_limit=1e-9, penalty_per_min=2.5, hint_seqs=warm)
    assert res["status"] == "HINT"
    assert res["seqs"] == warm
    assert res["objective"] == pytest.approx(evaluate(small, warm, P_small, penalty_per_min=2.5)["total"])


def test_exhausted_budget_without_a_warm_start_reports_the_status(small, P_small):
    res = solve(small, P_small, time_limit=1e-9)
    assert res["status"] == "UNKNOWN"
    assert "seqs" not in res


# Rescheduling: M1 breaks down at minute 1000 for four hours; work started before then is frozen.
T_BREAK, DOWN, BROKEN = 1000, 240, 0


@pytest.fixture(scope="module")
def breakdown(small, P_small):
    base = solve(small, P_small, time_limit=BUDGET)
    ev0 = evaluate(small, base["seqs"], P_small)
    windows = {m: list(w) for m, w in small.windows.items()}
    windows[BROKEN] = sorted([(T_BREAK, T_BREAK + DOWN), *windows[BROKEN]])
    frozen = {j: (int(ev0["assign"][j]), int(ev0["start"][j])) for j in range(small.N)
              if ev0["start"][j] < T_BREAK and not (ev0["assign"][j] == BROKEN and ev0["end"][j] > T_BREAK)}
    assert 0 < len(frozen) < small.N
    return ev0, windows, frozen


def _resolve(small, P_small, breakdown, max_moves):
    ev0, windows, frozen = breakdown
    res = solve(small, P_small, time_limit=BUDGET, windows=windows, frozen=frozen, not_before=T_BREAK,
                ref_assign=ev0["assign"], max_moves=max_moves)
    assert res["status"] == "OPTIMAL"
    not_before = np.array([0 if j in frozen else T_BREAK for j in range(small.N)])
    return res, evaluate(small, res["seqs"], P_small, windows=windows, not_before=not_before)


def test_rescheduling_keeps_frozen_work_and_delays_the_rest(small, P_small, breakdown):
    _, windows, frozen = breakdown
    res, ev = _resolve(small, P_small, breakdown, max_moves=None)
    assert_valid_plan(small, res["seqs"], windows)
    assert_clear_of_stops(small, res["seqs"], P_small, windows)
    for j in range(small.N):
        if j in frozen:
            assert (ev["assign"][j], ev["start"][j]) == frozen[j]
        else:
            assert ev["start"][j] >= T_BREAK


@pytest.mark.parametrize("cap", [0, 1])
def test_rescheduling_respects_the_move_cap(small, P_small, breakdown, cap):
    ev0, _, frozen = breakdown
    _, ev = _resolve(small, P_small, breakdown, max_moves=cap)
    moved = [j for j in range(small.N) if j not in frozen and ev["assign"][j] != ev0["assign"][j]]
    assert len(moved) <= cap


def test_rescheduling_with_more_freedom_never_costs_more(small, P_small, breakdown):
    totals = [_resolve(small, P_small, breakdown, max_moves=k)[0]["objective"] for k in (0, 1, None)]
    assert totals[0] >= totals[1] >= totals[2]


def test_isolated_solve_matches_in_process_solve(small, P_small, warm, solved):
    """Runs the solver in a child interpreter, which imports this package by module name."""
    res = solve_isolated(small, P_small, time_limit=BUDGET, penalty_per_min=2.5, hint_seqs=warm)
    assert res["status"] == solved["status"]
    assert res["seqs"] == solved["seqs"]
    assert res["objective"] == pytest.approx(solved["objective"])

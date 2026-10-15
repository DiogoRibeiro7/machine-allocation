from __future__ import annotations

import numpy as np
import pytest

from helpers import assert_clear_of_stops, assert_valid_plan
from machine_allocation.heuristics import K1_GRID, K2_GRID, atcs, atcs_tuned, best_baseline, edd_list, greedy_cost
from machine_allocation.simulate import evaluate

RULES = {
    "edd": lambda inst, P: edd_list(inst, P),
    "atcs": lambda inst, P: atcs(inst, P, k1=1.0, k2=0.5),
    "atcs_tuned": lambda inst, P: atcs_tuned(inst, P)[0],
    "greedy": lambda inst, P: greedy_cost(inst, P),
}


@pytest.mark.parametrize("rule", RULES)
def test_rules_build_valid_plans(rule, week, P_week):
    seqs = RULES[rule](week, P_week)
    assert_valid_plan(week, seqs)
    assert_clear_of_stops(week, seqs, P_week)


@pytest.mark.parametrize("rule", RULES)
def test_rules_handle_an_extra_stop(rule, week, P_week):
    windows = {m: list(w) for m, w in week.windows.items()}
    windows[1] = sorted([(40 * 60, 52 * 60), *windows[1]])
    seqs = {
        "edd": lambda: edd_list(week, P_week, windows),
        "atcs": lambda: atcs(week, P_week, 1.0, 0.5, windows),
        "atcs_tuned": lambda: atcs_tuned(week, P_week, windows=windows)[0],
        "greedy": lambda: greedy_cost(week, P_week, windows=windows),
    }[rule]()
    assert_valid_plan(week, seqs, windows)
    assert_clear_of_stops(week, seqs, P_week, windows)


def test_rules_are_deterministic(week, P_week):
    for rule in RULES.values():
        assert rule(week, P_week) == rule(week, P_week)


def test_tuned_atcs_is_at_least_as_good_as_any_grid_point(week, P_week):
    seqs, par = atcs_tuned(week, P_week)
    assert par["k1"] in K1_GRID and par["k2"] in K2_GRID
    tuned = evaluate(week, seqs, P_week)["total"]
    for k1, k2 in [(K1_GRID[0], K2_GRID[0]), (1.0, 0.5), (K1_GRID[-1], K2_GRID[-1])]:
        assert tuned <= evaluate(week, atcs(week, P_week, k1, k2), P_week)["total"]


def test_online_atcs_follows_the_actual_run_times(week, P_week):
    rng = np.random.default_rng(0)
    actual = week.true_mean * np.exp(rng.normal(0.0, 0.2, week.true_mean.shape))
    seqs = atcs(week, P_week, 1.0, 0.5, actual=actual)
    assert_valid_plan(week, seqs)
    assert seqs != atcs(week, P_week, 1.0, 0.5)


def test_best_baseline_picks_the_cheapest_rule(week, P_week):
    seqs, name = best_baseline(week, P_week, penalty_per_min=1.0)
    totals = {
        "ATCS": evaluate(week, atcs_tuned(week, P_week)[0], P_week)["total"],
        "EDD": evaluate(week, edd_list(week, P_week), P_week)["total"],
        "greedy": evaluate(week, greedy_cost(week, P_week), P_week)["total"],
    }
    assert name == min(totals, key=totals.__getitem__)
    assert evaluate(week, seqs, P_week)["total"] == pytest.approx(min(totals.values()))

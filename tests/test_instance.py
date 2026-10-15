from __future__ import annotations

import numpy as np
import pytest

from machine_allocation.instance import (
    ELIG,
    FAMILIES,
    MACHINES,
    MAINT_120,
    describe,
    int_times,
    make_instance,
    make_plant,
    run_cost,
)


def test_plant_is_reproducible(plant):
    again = make_plant(7)
    for name in ("rate_nom", "delta", "setup", "cold", "elig_mf"):
        np.testing.assert_array_equal(getattr(plant, name), getattr(again, name))
    assert not np.array_equal(plant.rate_nom, make_plant(8).rate_nom)


def test_plant_matches_eligibility_table(plant):
    for m, mn in enumerate(MACHINES):
        assert "".join(f for i, f in enumerate(FAMILIES) if plant.elig_mf[m, i]) == ELIG[mn]


def test_setups_are_whole_five_minute_blocks(plant):
    assert plant.setup.shape == (plant.M, plant.F, plant.F)
    assert (plant.setup % 5 == 0).all() and (plant.cold % 5 == 0).all()
    assert (plant.setup > 0).all() and (plant.cold > 0).all()
    # A lot change within a family is cheaper than any change of family.
    for m in range(plant.M):
        diag = np.diag(plant.setup[m])
        off = plant.setup[m][~np.eye(plant.F, dtype=bool)]
        assert diag.max() <= off.min()


def test_instance_shapes_and_ranges(week):
    N, M = week.N, week.M
    assert (N, M) == (50, 6)
    for name in ("family", "qty", "grade", "weight", "release", "due"):
        assert getattr(week, name).shape == (N,)
    for name in ("eligible", "nominal", "true_mean"):
        assert getattr(week, name).shape == (N, M)
    assert set(np.unique(week.weight)) <= {1, 2, 5}
    assert ((week.grade >= 0) & (week.grade <= 1)).all()
    assert (week.release % 60 == 0).all() and (week.due % 30 == 0).all()
    assert (week.due > week.release).all()
    np.testing.assert_array_equal(week.eligible, week.plant.elig_mf[:, week.family].T)
    assert (week.true_mean[week.eligible] > 0).all()


def test_instance_is_reproducible(plant, week):
    again = make_instance(50, jobs_seed=11, plant=plant)
    for name in ("family", "qty", "release", "due", "true_mean"):
        np.testing.assert_array_equal(getattr(week, name), getattr(again, name))


@pytest.mark.parametrize("horizon_h", [120.0, 60.0])
def test_maintenance_scales_with_the_horizon(plant, horizon_h):
    inst = make_instance(10, horizon_h=horizon_h, plant=plant)
    scale = horizon_h / 120.0
    expected = {MACHINES.index(mn): [(round(a * scale) * 60, round(a * scale) * 60 + d * 60)]
                for mn, (a, d) in MAINT_120.items()}
    assert inst.windows == expected
    assert inst.horizon == int(horizon_h * 60)


def test_int_times_marks_ineligible_pairs(week):
    P = int_times(week, week.true_mean)
    assert P.dtype.kind == "i"
    assert (P[~week.eligible] == -1).all()
    assert (P[week.eligible] >= 1).all()


def test_run_cost_is_hourly_rate_times_hours(plant, week):
    m = 0
    assert run_cost(week, m, 90) == pytest.approx(plant.cost_h[m] * 1.5)


def test_describe_reports_capacity_net_of_maintenance(week):
    d = describe(week)
    stop_h = sum(dur for _, dur in MAINT_120.values())
    assert d["jobs"] == 50
    assert d["capacity_h"] == pytest.approx(6 * 120 - stop_h)
    assert d["work_h_fastest_machine"] < d["work_h_mean_eligible"] < d["capacity_h"]

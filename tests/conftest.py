from __future__ import annotations

import numpy as np
import pytest

from machine_allocation.instance import Instance, Plant, int_times, make_instance, make_plant


@pytest.fixture(scope="session")
def plant() -> Plant:
    return make_plant(7)


@pytest.fixture(scope="session")
def week(plant: Plant) -> Instance:
    """The order book of the experiments: 50 jobs over a 120-hour week."""
    return make_instance(50, jobs_seed=11, plant=plant)


@pytest.fixture(scope="session")
def small(plant: Plant) -> Instance:
    """Eight jobs over 30 hours, which CP-SAT solves to proven optimality in about a second."""
    return make_instance(8, horizon_h=30.0, jobs_seed=2, plant=plant)


@pytest.fixture(scope="session")
def P_week(week: Instance) -> np.ndarray:
    return int_times(week, week.true_mean)


@pytest.fixture(scope="session")
def P_small(small: Instance) -> np.ndarray:
    return int_times(small, small.true_mean)

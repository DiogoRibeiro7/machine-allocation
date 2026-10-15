from __future__ import annotations

import numpy as np
import pytest

from machine_allocation.instance import draw_jobs, times_for
from machine_allocation.times import Estimators, make_history

KINDS = ["nameplate", "cell mean", "boosted mean", "boosted q75"]


@pytest.fixture(scope="module")
def history(plant):
    return make_history(plant, n_runs=1500, seed=3)


@pytest.fixture(scope="module")
def estimators(plant, history):
    return Estimators(plant, history)


@pytest.fixture(scope="module")
def fresh(plant):
    """Jobs the estimators have not seen, with their true mean times."""
    fam, qty, grade = draw_jobs(plant, 2000, np.random.default_rng(101))
    _, true_mean = times_for(plant, fam, qty, grade)
    return fam, qty, grade, true_mean, plant.elig_mf[:, fam].T


def test_history_only_uses_eligible_machines(plant, history):
    assert len(history["actual"]) == 1500
    assert plant.elig_mf[history["machine"], history["family"]].all()
    assert (history["actual"] > 0).all()


def test_history_has_rarely_seen_cells(plant, history):
    rare = history["rare_cells"]
    assert rare.shape == (5, 2)
    assert plant.elig_mf[rare[:, 0], rare[:, 1]].all()
    counts = np.zeros((plant.M, plant.F), dtype=int)
    np.add.at(counts, (history["machine"], history["family"]), 1)
    typical = np.median(counts[plant.elig_mf])
    assert all(counts[m, f] < typical / 4 for m, f in rare)


def test_history_is_reproducible(plant, history):
    again = make_history(plant, n_runs=1500, seed=3)
    np.testing.assert_array_equal(history["actual"], again["actual"])


def test_cell_counts_cover_the_history(estimators, history):
    assert estimators.count.sum() == len(history["actual"])


def test_nameplate_matrix_is_the_nominal_time(plant, estimators, fresh):
    fam, qty, grade, _, _ = fresh
    nominal, _ = times_for(plant, fam, qty, grade)
    np.testing.assert_allclose(estimators.matrix("nameplate", fam, qty, grade), nominal)


@pytest.mark.parametrize("kind", KINDS)
def test_matrices_are_positive_and_finite(estimators, fresh, kind):
    fam, qty, grade, _, _ = fresh
    out = estimators.matrix(kind, fam, qty, grade)
    assert out.shape == (len(fam), 6)
    assert np.isfinite(out).all() and (out > 0).all()


def test_unknown_estimator_is_rejected(estimators, fresh):
    fam, qty, grade, _, _ = fresh
    with pytest.raises(ValueError, match="median"):
        estimators.matrix("median", fam, qty, grade)


def test_learned_estimates_beat_nameplate(estimators, fresh):
    """The ranking experiment 3 relies on: nameplate rates underestimate real run times,
    the boosted mean is close to unbiased and the most accurate, the upper quantile errs long."""
    fam, qty, grade, true_mean, el = fresh
    err = {k: np.log(estimators.matrix(k, fam, qty, grade)[el] / true_mean[el]) for k in KINDS}
    rmse = {k: float(np.sqrt(np.mean(e ** 2))) for k, e in err.items()}
    bias = {k: float(np.mean(e)) for k, e in err.items()}
    assert rmse["boosted mean"] < rmse["cell mean"] < rmse["nameplate"]
    assert bias["nameplate"] < -0.05
    assert abs(bias["boosted mean"]) < 0.02
    assert bias["boosted q75"] > 0.03

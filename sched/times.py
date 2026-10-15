"""Run-time data: a historical production log and four ways of turning it into the
processing-time matrix the scheduler needs."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

from .instance import MCLASS, Plant, draw_jobs, systematic_log_ratio

HIST_PREF = {"new": 1.6, "mid": 1.0, "old": 0.6}   # how often planners used each class


def make_history(plant: Plant, n_runs: int = 4000, seed: int = 3, n_rare: int = 5):
    """Past runs. Planners chose machines by habit, so some machine-family pairs are
    well documented and a few are almost never seen."""
    rng = np.random.default_rng(seed)
    fam, qty, grade = draw_jobs(plant, n_runs, rng)
    pref = np.array([[HIST_PREF[c]] * plant.F for c in MCLASS]) * plant.elig_mf
    cells = np.argwhere(plant.elig_mf)
    rare = cells[rng.choice(len(cells), size=n_rare, replace=False)]
    for m, f in rare:
        pref[m, f] *= 0.04
    mach = np.array([rng.choice(plant.M, p=pref[:, f] / pref[:, f].sum()) for f in fam])
    nominal = 60.0 * qty / plant.rate_nom[mach, fam]
    lr = systematic_log_ratio(plant, mach, fam, qty, grade)
    actual = nominal * np.exp(lr + rng.normal(0.0, plant.sigma[mach]))
    return {"machine": mach, "family": fam, "qty": qty, "grade": grade,
            "nominal": nominal, "actual": actual, "rare_cells": rare}


def _features(plant: Plant, mach, fam, qty, grade):
    return np.column_stack([mach, fam, mach * plant.F + fam,
                            np.log(qty / plant.q_ref[fam]), grade])


class Estimators:
    """nameplate / cell mean / boosted mean / boosted upper quantile."""

    def __init__(self, plant: Plant, hist: dict, quantile: float = 0.75):
        self.plant = plant
        ratio = hist["actual"] / hist["nominal"]
        self.cell = np.full((plant.M, plant.F), np.nan)
        self.count = np.zeros((plant.M, plant.F), dtype=int)
        for m in range(plant.M):
            mm = hist["machine"] == m
            for f in range(plant.F):
                idx = mm & (hist["family"] == f)
                self.count[m, f] = idx.sum()
                if idx.sum() >= 1:
                    self.cell[m, f] = ratio[idx].mean()
                elif mm.any():
                    self.cell[m, f] = ratio[mm].mean()
        X = _features(plant, hist["machine"], hist["family"], hist["qty"], hist["grade"])
        y = np.log(ratio)
        kw = dict(categorical_features=[0, 1, 2], max_iter=300, learning_rate=0.05,
                  max_leaf_nodes=8, min_samples_leaf=15, random_state=0)
        self.gbm = HistGradientBoostingRegressor(**kw).fit(X, y)
        oof = cross_val_predict(HistGradientBoostingRegressor(**kw), X, y, cv=5)
        self.smear = float(np.mean(np.exp(y - oof)))          # Duan (1983)
        self.gbq = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw).fit(X, y)

    def matrix(self, kind: str, fam, qty, grade) -> np.ndarray:
        pl = self.plant
        n = len(fam)
        out = np.zeros((n, pl.M))
        for m in range(pl.M):
            nominal = 60.0 * qty / pl.rate_nom[m, fam]
            if kind == "nameplate":
                out[:, m] = nominal
            elif kind == "cell mean":
                r = self.cell[m, fam]
                out[:, m] = nominal * np.where(np.isnan(r), 1.0, r)
            else:
                X = _features(pl, np.full(n, m), fam, qty, grade)
                if kind == "boosted mean":
                    out[:, m] = nominal * np.exp(self.gbm.predict(X)) * self.smear
                elif kind == "boosted q75":
                    out[:, m] = nominal * np.exp(self.gbq.predict(X))
                else:
                    raise ValueError(kind)
        return out

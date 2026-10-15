"""Plant and order-book generator.

Everything is in integer minutes. The plant (machines, rates, setups, the way real run
times deviate from nameplate) is fixed by ``plant_seed``; an order book is drawn on top
of it with ``jobs_seed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAMILIES = ["A", "B", "C", "D", "E"]
MACHINES = ["M1", "M2", "M3", "M4", "M5", "M6"]
MCLASS = ["new", "new", "mid", "mid", "old", "old"]

# Which families each machine can run.
ELIG = {
    "M1": "ABCDE",
    "M2": "ABCDE",
    "M3": "ABCD",
    "M4": "BCDE",
    "M5": "ABC",
    "M6": "ACD",
}
SPEED = {"new": 1.25, "mid": 1.00, "old": 0.75}          # nameplate speed factor
COST_H = {"new": 95.0, "mid": 70.0, "old": 48.0}          # running cost, EUR / hour
SETUP_MULT = {"new": 0.70, "mid": 1.00, "old": 1.30}      # changeover speed
SETUP_COST_MIN = 1.0                                       # EUR per setup minute
# How reality departs from nameplate, by machine class (log scale).
DELTA_MU = {"new": 0.06, "mid": 0.10, "old": 0.16}
DELTA_SD = {"new": 0.07, "mid": 0.10, "old": 0.13}
GAMMA = {"new": 0.10, "mid": 0.25, "old": 0.45}           # sensitivity to material grade
RHO = {"new": 0.04, "mid": 0.07, "old": 0.11}             # short-lot ramp-up penalty
SIGMA = {"new": 0.08, "mid": 0.12, "old": 0.18}           # run-to-run noise
FAMILY_P = np.array([0.25, 0.25, 0.20, 0.18, 0.12])
# Planned maintenance on a 120-hour week: machine -> (start hour, duration hours).
MAINT_120 = {"M1": (30, 6), "M3": (54, 8), "M5": (78, 6), "M2": (96, 4)}


@dataclass
class Plant:
    rate_nom: np.ndarray      # (M, F) nameplate units per hour
    delta: np.ndarray         # (M, F) systematic log deviation of real time from nameplate
    gamma: np.ndarray         # (M,)
    rho: np.ndarray           # (M,)
    sigma: np.ndarray         # (M,)
    cost_h: np.ndarray        # (M,)
    setup: np.ndarray         # (M, F, F) minutes, from family f to family g
    cold: np.ndarray          # (M, F) minutes, first job / first job after maintenance
    elig_mf: np.ndarray       # (M, F) bool
    q_ref: np.ndarray         # (F,) typical lot size

    @property
    def M(self) -> int:
        return len(MACHINES)

    @property
    def F(self) -> int:
        return len(FAMILIES)


def make_plant(plant_seed: int = 7) -> Plant:
    rng = np.random.default_rng(plant_seed)
    M, F = len(MACHINES), len(FAMILIES)
    base_rate = np.array([100.0, 80.0, 120.0, 90.0, 60.0])   # units/hour on a mid machine
    rate = np.zeros((M, F))
    delta = np.zeros((M, F))
    for m, cls in enumerate(MCLASS):
        rate[m] = base_rate * SPEED[cls] * rng.uniform(0.88, 1.12, F)
        delta[m] = rng.normal(DELTA_MU[cls], DELTA_SD[cls], F)
    # Family-to-family changeover on a mid machine, asymmetric.
    base = rng.integers(45, 181, size=(F, F)).astype(float)
    base = 15.0 * np.round(base / 15.0)
    np.fill_diagonal(base, 15.0)                              # same family: lot change only
    setup = np.zeros((M, F, F), dtype=int)
    cold = np.zeros((M, F), dtype=int)
    for m, cls in enumerate(MCLASS):
        setup[m] = np.round(base * SETUP_MULT[cls] / 5.0) * 5
        off = base.copy()
        np.fill_diagonal(off, np.nan)
        cold[m] = np.round(np.nanmean(off, axis=0) * SETUP_MULT[cls] / 5.0) * 5
    elig = np.array([[f in ELIG[mn] for f in FAMILIES] for mn in MACHINES])
    q_ref = base_rate * 9.0                                   # about nine hours on a mid machine
    return Plant(
        rate_nom=rate, delta=delta,
        gamma=np.array([GAMMA[c] for c in MCLASS]),
        rho=np.array([RHO[c] for c in MCLASS]),
        sigma=np.array([SIGMA[c] for c in MCLASS]),
        cost_h=np.array([COST_H[c] for c in MCLASS]),
        setup=setup, cold=cold, elig_mf=elig, q_ref=q_ref,
    )


def systematic_log_ratio(plant: Plant, m, fam, qty, grade):
    """log(E-ish real time / nameplate time) before noise, vectorised over runs."""
    ramp = np.clip(plant.q_ref[fam] / qty - 1.0, -0.5, 2.0)
    return plant.delta[m, fam] + plant.gamma[m] * (grade - 0.5) + plant.rho[m] * ramp


@dataclass
class Instance:
    plant: Plant
    family: np.ndarray        # (N,) int
    qty: np.ndarray           # (N,) units
    grade: np.ndarray         # (N,) in [0, 1]
    weight: np.ndarray        # (N,) int priority weight
    release: np.ndarray       # (N,) minutes, material availability
    due: np.ndarray           # (N,) minutes
    eligible: np.ndarray      # (N, M) bool
    nominal: np.ndarray       # (N, M) nameplate minutes (float)
    true_mean: np.ndarray     # (N, M) expected real minutes (float)
    windows: dict = field(default_factory=dict)   # machine -> list of (start, end) minutes
    horizon: int = 7200

    @property
    def N(self) -> int:
        return len(self.family)

    @property
    def M(self) -> int:
        return self.plant.M


def draw_jobs(plant: Plant, n: int, rng: np.random.Generator):
    fam = rng.choice(len(FAMILIES), size=n, p=FAMILY_P)
    qty = plant.q_ref[fam] * np.clip(rng.lognormal(0.0, 0.45, n), 0.3, 2.4)
    qty = np.round(qty / 10.0) * 10.0
    grade = rng.beta(2.0, 2.0, n)
    return fam, qty, grade


def times_for(plant: Plant, fam, qty, grade):
    """Nameplate and expected real minutes for every job on every machine."""
    n, M = len(fam), plant.M
    nominal = np.zeros((n, M))
    true_mean = np.zeros((n, M))
    for m in range(M):
        nominal[:, m] = 60.0 * qty / plant.rate_nom[m, fam]
        lr = systematic_log_ratio(plant, m, fam, qty, grade)
        true_mean[:, m] = nominal[:, m] * np.exp(lr + 0.5 * plant.sigma[m] ** 2)
    return nominal, true_mean


def make_instance(n_jobs: int = 50, horizon_h: float = 120.0, jobs_seed: int = 11,
                  plant: Plant | None = None, due_lo: float = 0.15) -> Instance:
    plant = plant or make_plant()
    rng = np.random.default_rng(jobs_seed)
    H = int(round(horizon_h * 60))
    fam, qty, grade = draw_jobs(plant, n_jobs, rng)
    nominal, true_mean = times_for(plant, fam, qty, grade)
    eligible = plant.elig_mf[:, fam].T.copy()
    weight = rng.choice([1, 2, 5], size=n_jobs, p=[0.6, 0.3, 0.1])
    late_mat = rng.random(n_jobs) < 0.35
    release = np.where(late_mat, rng.uniform(6 * 60, 0.6 * H, n_jobs), 0.0)
    release = (np.round(release / 60.0) * 60).astype(int)
    pbar = np.array([true_mean[j, eligible[j]].mean() for j in range(n_jobs)])
    due = np.maximum(release + 1.5 * pbar, rng.uniform(due_lo * H, 1.0 * H, n_jobs))
    due = (np.round(due / 30.0) * 30).astype(int)
    scale = horizon_h / 120.0
    windows = {}
    for mn, (a, d) in MAINT_120.items():
        m = MACHINES.index(mn)
        a_min = int(round(a * scale)) * 60
        windows[m] = [(a_min, a_min + d * 60)]
    return Instance(plant=plant, family=fam, qty=qty, grade=grade, weight=weight,
                    release=release, due=due, eligible=eligible, nominal=nominal,
                    true_mean=true_mean, windows=windows, horizon=H)


def int_times(inst: Instance, P: np.ndarray) -> np.ndarray:
    """Round a duration matrix to integer minutes; ineligible pairs get -1."""
    out = np.maximum(1, np.round(P)).astype(int)
    out[~inst.eligible] = -1
    return out


def run_cost(inst: Instance, m: int, minutes) -> np.ndarray:
    return inst.plant.cost_h[m] * np.asarray(minutes, dtype=float) / 60.0


def describe(inst: Instance) -> dict:
    P = inst.true_mean
    best = np.where(inst.eligible, P, np.inf).min(axis=1)
    cap = inst.M * inst.horizon - sum(b - a for w in inst.windows.values() for a, b in w)
    return {
        "jobs": inst.N,
        "horizon_h": inst.horizon / 60,
        "work_h_fastest_machine": float(best.sum() / 60),
        "work_h_mean_eligible": float(np.mean([P[j, inst.eligible[j]].mean() for j in range(inst.N)]) * inst.N / 60),
        "capacity_h": cap / 60,
        "late_material_jobs": int((inst.release > 0).sum()),
        "eligible_machines_per_job": float(inst.eligible.sum(axis=1).mean()),
    }

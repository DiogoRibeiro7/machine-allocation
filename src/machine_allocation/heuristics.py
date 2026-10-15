"""Constructive baselines: an EDD list rule and a tuned ATCS dispatching rule."""
from __future__ import annotations

import itertools

import numpy as np

from .instance import SETUP_COST_MIN, Instance
from .simulate import evaluate


class _State:
    def __init__(self, inst: Instance, windows: dict):
        self.inst = inst
        self.t = np.zeros(inst.M)
        self.prev = [None] * inst.M
        self.pending = {m: list(range(len(windows.get(m, [])))) for m in range(inst.M)}
        self.windows = windows
        self.seqs: dict[int, list[int]] = {m: [] for m in range(inst.M)}

    def place(self, j: int, m: int, P: np.ndarray, commit: bool = False,
              actual: np.ndarray | None = None):
        """Earliest (start, end, setup) for job j appended on machine m, judged with the
        planning times P. When ``actual`` is given the machine is released at the real
        finishing time, which is how the rule behaves on a live shop floor."""
        inst, pl = self.inst, self.inst.plant
        t, prev, pend = self.t[m], self.prev[m], list(self.pending[m])
        f = inst.family[j]
        crossed = []
        while True:
            s = pl.cold[m, f] if prev is None else pl.setup[m, prev, f]
            st = max(t + s, inst.release[j])
            en = st + P[j, m]
            if pend and en > self.windows[m][pend[0]][0]:
                a, b = self.windows[m][pend[0]]
                t = max(t, a) + (b - a)
                prev = None
                crossed.append(pend.pop(0))
                continue
            break
        if commit:
            for k in crossed:
                self.seqs[m].append(-(k + 1))
            self.seqs[m].append(j)
            if actual is not None:
                en = st + actual[j, m]
            self.t[m], self.prev[m], self.pending[m] = en, f, pend
        return st, en, s

    def finish(self):
        for m in range(self.inst.M):
            for k in self.pending[m]:
                self.seqs[m].append(-(k + 1))
            self.pending[m] = []
        return self.seqs


def edd_list(inst: Instance, P: np.ndarray, windows: dict | None = None) -> dict:
    """Earliest due date first; each job goes to the machine that finishes it soonest."""
    windows = inst.windows if windows is None else windows
    st = _State(inst, windows)
    for j in np.lexsort((-inst.weight, inst.due)):
        ms = np.flatnonzero(inst.eligible[j]).tolist()
        best = min(ms, key=lambda m: st.place(j, m, P)[1])
        st.place(j, best, P, commit=True)
    return st.finish()


def atcs(inst: Instance, P: np.ndarray, k1: float, k2: float,
         windows: dict | None = None, actual: np.ndarray | None = None) -> dict:
    """Apparent Tardiness Cost with Setups (Lee, Bhaskaran and Pinedo, 1997), run as a
    dispatching rule: whenever a machine falls idle it takes the waiting job with the
    highest index. Processing times are machine-specific, which is the usual extension
    to unrelated machines."""
    windows = inst.windows if windows is None else windows
    pl = inst.plant
    st = _State(inst, windows)
    todo = set(range(inst.N))
    pbar = float(np.mean(P[inst.eligible]))
    sbar = float(pl.setup.mean())
    done_m = np.zeros(inst.M, dtype=bool)
    while todo:
        cand_m = [m for m in range(inst.M) if not done_m[m]]
        m = min(cand_m, key=lambda q: st.t[q])
        elig = [j for j in todo if inst.eligible[j, m]]
        if not elig:
            done_m[m] = True
            continue
        t, prev = st.t[m], st.prev[m]
        setup = {j: pl.cold[m, inst.family[j]] if prev is None else pl.setup[m, prev, inst.family[j]]
                 for j in elig}
        # A job counts as waiting if its material is in, or arrives during its changeover.
        ready = [j for j in elig if inst.release[j] <= t + setup[j]]
        if not ready:
            st.t[m] = min(inst.release[j] - setup[j] for j in elig)
            continue
        best, best_idx = ready[0], -np.inf
        for j in ready:
            s = setup[j]
            p = P[j, m]
            slack = max(inst.due[j] - p - t, 0.0)
            idx = (inst.weight[j] / p) * np.exp(-slack / (k1 * pbar)) * np.exp(-s / (k2 * sbar))
            if idx > best_idx:
                best, best_idx = j, idx
        st.place(best, m, P, commit=True, actual=actual)
        todo.remove(best)
    return st.finish()


K1_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
K2_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def atcs_tuned(inst: Instance, P: np.ndarray, penalty_per_min: float = 1.0,
               windows: dict | None = None):
    """Grid-search the two look-ahead parameters on the instance itself, which is the
    most favourable way to run the rule."""
    best_tot, best_seqs, best_par = np.inf, {}, {}
    for k1, k2 in itertools.product(K1_GRID, K2_GRID):
        seqs = atcs(inst, P, k1, k2, windows)
        tot = evaluate(inst, seqs, P, penalty_per_min, windows)["total"]
        if tot < best_tot:
            best_tot, best_seqs, best_par = tot, seqs, {"k1": k1, "k2": k2}
    return best_seqs, best_par


def greedy_cost(inst: Instance, P: np.ndarray, penalty_per_min: float = 1.0,
                windows: dict | None = None) -> dict:
    """A cost-aware list rule: jobs in due-date order, each sent to the machine where its
    own running cost, changeover cost and lateness penalty add up to the least. It sees
    the same cost function as the solver but decides one job at a time."""
    windows = inst.windows if windows is None else windows
    st = _State(inst, windows)
    for j in np.lexsort((-inst.weight, inst.due)):
        ms = np.flatnonzero(inst.eligible[j]).tolist()
        best, best_c = ms[0], np.inf
        for m in ms:
            _, en, su = st.place(j, m, P)
            c = (inst.plant.cost_h[m] * P[j, m] / 60.0 + SETUP_COST_MIN * su
                 + penalty_per_min * inst.weight[j] * max(0.0, en - inst.due[j]))
            if c < best_c:
                best, best_c = m, c
        st.place(j, best, P, commit=True)
    return st.finish()


def best_baseline(inst: Instance, P: np.ndarray, penalty_per_min: float = 1.0,
                  windows: dict | None = None):
    """The best of the three rules on this instance; used to warm-start the solver."""
    cands = {
        "ATCS": atcs_tuned(inst, P, penalty_per_min, windows)[0],
        "EDD": edd_list(inst, P, windows),
        "greedy": greedy_cost(inst, P, penalty_per_min, windows),
    }
    name = min(cands, key=lambda k: evaluate(inst, cands[k], P, penalty_per_min, windows)["total"])
    return cands[name], name

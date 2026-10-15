"""CP-SAT model: unrelated parallel machines, eligibility, release dates, sequence-dependent
setups, planned stops, weighted tardiness plus operating cost."""
from __future__ import annotations

import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
from ortools.sat.python import cp_model

from .instance import SETUP_COST_MIN, Instance
from .simulate import evaluate


def solve(inst: Instance, P: np.ndarray, *, time_limit: float = 60.0,
          penalty_per_min: float = 1.0, cost_weight: float = 1.0,
          hint_seqs: dict | None = None, windows: dict | None = None,
          frozen: dict | None = None, not_before: int = 0,
          ref_assign: np.ndarray | None = None, max_moves: int | None = None,
          workers: int = 4, seed: int = 1, deterministic: bool = True, log: bool = False,
          hint_not_before: np.ndarray | None = None, trace_progress: bool = False):
    """Minimise  penalty * sum_j w_j T_j  +  cost_weight * (setup cost + running cost).

    P[j, m]   integer minutes, -1 where machine m cannot run job j.
    windows   machine -> [(start, end)] stops; a stop resets the machine to cold.
    frozen    job -> (machine, start): work already done or under way (rescheduling).
    not_before  earliest start for every job that is not frozen (rescheduling).
    max_moves   cap on the number of unfrozen jobs that leave their ref_assign machine.
    Objective coefficients are scaled to integers (1/600 EUR).
    """
    windows = inst.windows if windows is None else windows
    frozen = frozen or {}
    pl = inst.plant
    N, M = inst.N, inst.M
    ub = int(max(inst.horizon, inst.release.max()) + P.max(axis=1).sum() / 2 + 3000)
    mdl = cp_model.CpModel()

    start = [mdl.new_int_var(0, ub, f"s{j}") for j in range(N)]
    end = [mdl.new_int_var(0, ub, f"e{j}") for j in range(N)]
    tard = [mdl.new_int_var(0, ub, f"t{j}") for j in range(N)]
    x = {}
    obj = []
    scale = 600                      # 1/600 EUR: keeps per-minute penalties integral

    for j in range(N):
        mdl.add(start[j] >= int(inst.release[j]))
        if j not in frozen and not_before:
            mdl.add(start[j] >= int(not_before))
        mdl.add(tard[j] >= end[j] - int(inst.due[j]))
        obj.append(int(round(scale * penalty_per_min * inst.weight[j])) * tard[j])
        lits = []
        for m in range(M):
            if P[j, m] < 0:
                continue
            x[j, m] = mdl.new_bool_var(f"x{j}_{m}")
            lits.append(x[j, m])
            mdl.add(end[j] == start[j] + int(P[j, m])).only_enforce_if(x[j, m])
            c = pl.cost_h[m] * P[j, m] / 60.0
            obj.append(int(round(scale * cost_weight * c)) * x[j, m])
        mdl.add_exactly_one(lits)
        if j in frozen:
            fm, fs = frozen[j]
            mdl.add(x[j, fm] == 1)
            mdl.add(start[j] == int(fs))

    arc_lits = {}
    for m in range(M):
        jobs = [j for j in range(N) if (j, m) in x]
        wins = windows.get(m, [])
        # Node 0 is the depot; jobs are 1..n; stops follow.
        node = {j: i + 1 for i, j in enumerate(jobs)}
        wnode = {k: len(jobs) + 1 + k for k in range(len(wins))}
        arcs = []
        intervals = []
        for j in jobs:
            fj = inst.family[j]
            intervals.append(mdl.new_optional_interval_var(
                start[j], int(P[j, m]), end[j], x[j, m], f"iv{j}_{m}"))
            arcs.append((node[j], node[j], ~x[j, m]))            # job not on this machine
            first = mdl.new_bool_var(f"f{j}_{m}")
            arcs.append((0, node[j], first))
            arc_lits[m, "D", j] = first
            cold = int(pl.cold[m, fj])
            mdl.add(start[j] >= cold).only_enforce_if(first)
            obj.append(int(round(scale * cost_weight * SETUP_COST_MIN * cold)) * first)
            arcs.append((node[j], 0, mdl.new_bool_var(f"l{j}_{m}")))
            for i in jobs:
                if i == j:
                    continue
                lit = mdl.new_bool_var(f"a{i}_{j}_{m}")
                arcs.append((node[i], node[j], lit))
                arc_lits[m, i, j] = lit
                s = int(pl.setup[m, inst.family[i], fj])
                mdl.add(start[j] >= end[i] + s).only_enforce_if(lit)
                obj.append(int(round(scale * cost_weight * SETUP_COST_MIN * s)) * lit)
            for k, (a, b) in enumerate(wins):
                before = mdl.new_bool_var(f"jb{j}_{k}_{m}")
                after = mdl.new_bool_var(f"ja{j}_{k}_{m}")
                arcs.append((node[j], wnode[k], before))
                arcs.append((wnode[k], node[j], after))
                arc_lits[m, j, ("W", k)] = before
                arc_lits[m, ("W", k), j] = after
                mdl.add(end[j] <= int(a)).only_enforce_if(before)
                mdl.add(start[j] >= int(b) + cold).only_enforce_if(after)
                obj.append(int(round(scale * cost_weight * SETUP_COST_MIN * cold)) * after)
        for k, (a, b) in enumerate(wins):
            intervals.append(mdl.new_fixed_size_interval_var(int(a), int(b - a), f"w{k}_{m}"))
            lit0 = mdl.new_bool_var(f"dw{k}_{m}")
            arcs.append((0, wnode[k], lit0))
            arc_lits[m, "D", ("W", k)] = lit0
            arcs.append((wnode[k], 0, mdl.new_bool_var(f"wd{k}_{m}")))
            for k2 in range(len(wins)):
                if k2 != k and wins[k2][0] >= b:
                    lit = mdl.new_bool_var(f"ww{k}_{k2}_{m}")
                    arcs.append((wnode[k], wnode[k2], lit))
                    arc_lits[m, ("W", k), ("W", k2)] = lit
        if not wins:
            arcs.append((0, 0, mdl.new_bool_var(f"empty{m}")))
        mdl.add_circuit(arcs)
        mdl.add_no_overlap(intervals)

    if max_moves is not None and ref_assign is not None:
        moved = [1 - x[j, int(ref_assign[j])] for j in range(N) if j not in frozen]
        mdl.add(sum(moved) <= int(max_moves))

    mdl.minimize(sum(obj))

    if hint_seqs is not None:
        ev = evaluate(inst, hint_seqs, P, penalty_per_min, windows, hint_not_before)
        for j in range(N):
            if j in frozen:
                continue
            mdl.add_hint(start[j], int(ev["start"][j]))
            mdl.add_hint(end[j], int(ev["end"][j]))
            mdl.add_hint(tard[j], int(max(0, ev["end"][j] - inst.due[j])))
            for m in range(M):
                if (j, m) in x:
                    mdl.add_hint(x[j, m], int(ev["assign"][j] == m))
        for m, seq in hint_seqs.items():
            prev = "D"
            used = set()
            for item in seq:
                cur = ("W", -item - 1) if item < 0 else item
                if (m, prev, cur) in arc_lits:
                    mdl.add_hint(arc_lits[m, prev, cur], 1)
                    used.add((m, prev, cur))
                prev = cur
            for key, lit in arc_lits.items():
                if key[0] == m and key not in used:
                    mdl.add_hint(lit, 0)

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log
    if deterministic:
        # Interleaved search with a deterministic-time budget makes a multi-worker run
        # repeatable; ``time_limit`` is then in solver work units, not seconds.
        solver.parameters.interleave_search = True
        solver.parameters.max_deterministic_time = time_limit
    else:
        solver.parameters.max_time_in_seconds = time_limit
    t0 = time.time()
    trace = []

    class _Trace(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            trace.append((time.time() - t0, self.objective_value / scale,
                          self.best_objective_bound / scale))

    status = solver.solve(mdl, _Trace()) if trace_progress else solver.solve(mdl)
    wall = time.time() - t0
    name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if hint_seqs is None:
            return {"status": name, "wall": wall}
        # Budget too small to improve on the starting plan: hand it back unchanged.
        ev = evaluate(inst, hint_seqs, P, penalty_per_min, windows, hint_not_before)
        tot = (ev["weighted_tardiness_h"] * 60 * penalty_per_min + cost_weight * ev["operating_cost"])
        return {"status": "HINT", "wall": wall, "objective": tot,
                "bound": solver.best_objective_bound / scale, "seqs": hint_seqs,
                "solver_start": ev["start"], "assign": ev["assign"], "trace": trace}

    st = np.array([solver.value(v) for v in start])
    assign = np.array([next(m for m in range(M) if (j, m) in x and solver.value(x[j, m]))
                       for j in range(N)])
    seqs = {}
    for m in range(M):
        items = [(st[j], 1, j) for j in range(N) if assign[j] == m]
        items += [(a, 0, -(k + 1)) for k, (a, b) in enumerate(windows.get(m, []))]
        # A job that starts at or after a stop begins must follow it.
        seqs[m] = [it for _, _, it in sorted(items)]
    return {
        "status": name, "wall": wall,
        "objective": solver.objective_value / scale,
        "bound": solver.best_objective_bound / scale,
        "seqs": seqs, "solver_start": st, "assign": assign, "trace": trace,
    }


def solve_isolated(*args, retries: int = 3, **kwargs):
    """Run ``solve`` in a fresh interpreter and retry if the process dies.

    A long batch of solves is only as reliable as its least reliable native call; one
    aborted run should cost one solve, not the batch. With a deterministic budget a retry
    returns the same plan the failed attempt would have returned.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.pkl"), os.path.join(tmp, "out.pkl")
        with open(fin, "wb") as fh:
            pickle.dump((args, kwargs), fh)
        for attempt in range(retries):
            proc = subprocess.run([sys.executable, "-m", "sched.model", fin, fout], env=env)
            if proc.returncode == 0 and os.path.exists(fout):
                with open(fout, "rb") as fh:
                    return pickle.load(fh)
            print(f"solver process exited with code {proc.returncode}; retrying", flush=True)
    raise RuntimeError("the solver process failed on every attempt")


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as _fh:
        _args, _kwargs = pickle.load(_fh)
    _res = solve(*_args, **_kwargs)
    with open(sys.argv[2], "wb") as _fh:
        pickle.dump(_res, _fh)

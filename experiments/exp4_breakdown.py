"""Experiment 4: a machine fails mid-week. Right-shift, or re-optimise with a cap on how
many jobs may change machine."""
import numpy as np

from common import JOBS_SEED, N_JOBS, PEN, PLANT_SEED, budget, load_pickle, save_json, scalars
from machine_allocation.instance import int_times, make_instance, make_plant
from machine_allocation.model import solve_isolated as solve
from machine_allocation.simulate import evaluate

BM, T_B, DOWN = 1, 40 * 60, 12 * 60          # M2 fails at hour 40 for twelve hours
inst = make_instance(N_JOBS, jobs_seed=JOBS_SEED, plant=make_plant(PLANT_SEED))
P = int_times(inst, inst.true_mean)
base = load_pickle("exp1_plans.pkl")["CP-SAT"]
ev0 = evaluate(inst, base, P, PEN)

windows = {m: list(w) for m, w in inst.windows.items()}
windows[BM] = sorted([(T_B, T_B + DOWN), *windows.get(BM, [])])
frozen, interrupted = {}, None
for j in range(inst.N):
    if ev0["start"][j] < T_B:
        if ev0["assign"][j] == BM and ev0["end"][j] > T_B:
            interrupted = j                    # work in progress is lost and must be redone
        else:
            frozen[j] = (int(ev0["assign"][j]), int(ev0["start"][j]))
nb = np.array([0 if j in frozen else T_B for j in range(inst.N)])
free = [j for j in range(inst.N) if j not in frozen]

# Right-shift: same machines, same order, everything on M2 slides.
rs = {m: list(s) for m, s in base.items()}
seq, placed = [], False
for it in base[BM]:
    if it < 0:
        seq.append(-2)                          # planned maintenance is now window 1
        continue
    if not placed and it not in frozen:
        seq.append(-1)
        placed = True
    seq.append(it)
if not placed:
    seq.append(-1)
rs[BM] = seq

rows = []


def record(label, seqs, extra=None):
    ev = evaluate(inst, seqs, P, PEN, windows, nb)
    moved = int(sum(ev["assign"][j] != ev0["assign"][j] for j in free))
    shift = float(np.mean([abs(ev["start"][j] - ev0["start"][j]) for j in free]) / 60)
    row = {"policy": label, "moved_jobs": moved, "mean_start_shift_h": shift, **scalars(ev), **(extra or {})}
    rows.append(row)
    print(label, round(ev["total"]), round(ev["weighted_tardiness_h"], 1), ev["late_jobs"], moved,
          round(shift, 1), flush=True)
    return ev


record("right-shift", rs)
best = rs
for K in (0, 2, 5, 10, None):
    res = solve(inst, P, time_limit=budget(20.0), penalty_per_min=PEN, hint_seqs=best, windows=windows,
                frozen=frozen, not_before=T_B, ref_assign=ev0["assign"], max_moves=K,
                hint_not_before=nb)
    label = "re-optimise, any moves" if K is None else f"re-optimise, at most {K} moves"
    record(label, res["seqs"], {"bound": res["bound"], "wall_s": res["wall"], "status": res["status"]})
    best = res["seqs"]
save_json("exp4_breakdown.json", {
    "machine": "M2", "t_break_h": T_B / 60, "down_h": DOWN / 60, "interrupted_job": interrupted,
    "frozen_jobs": len(frozen), "free_jobs": len(free), "base_total": ev0["total"],
    "base_wt_h": ev0["weighted_tardiness_h"], "rows": rows})

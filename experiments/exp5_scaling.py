"""Experiment 5: the same plant with a longer order book. Fixed solver budget."""
from common import JOBS_SEED, N_JOBS, PEN, PLANT_SEED, budget, save_json
from machine_allocation.heuristics import atcs_tuned, edd_list, greedy_cost
from machine_allocation.instance import int_times, make_instance, make_plant
from machine_allocation.model import solve_isolated as solve
from machine_allocation.simulate import evaluate

BUDGET = budget(40.0)
rows = []
for n in (20, 35, 50, 80, 120):
    inst = make_instance(n, horizon_h=120.0 * n / 50.0, jobs_seed=JOBS_SEED if n == N_JOBS else JOBS_SEED + n,
                         plant=make_plant(PLANT_SEED))
    P = int_times(inst, inst.true_mean)
    e = evaluate(inst, edd_list(inst, P), P, PEN)
    hint, par = atcs_tuned(inst, P, PEN)
    a = evaluate(inst, hint, P, PEN)
    gs = greedy_cost(inst, P, PEN)
    g = evaluate(inst, gs, P, PEN)
    cands = [(a["total"], hint), (e["total"], edd_list(inst, P)), (g["total"], gs)]
    res = solve(inst, P, time_limit=BUDGET, penalty_per_min=PEN,
                hint_seqs=min(cands, key=lambda c: c[0])[1])
    c = evaluate(inst, res["seqs"], P, PEN)
    rows.append({"jobs": n, "horizon_h": inst.horizon / 60, "edd_total": e["total"], "atcs_total": a["total"],
                 "greedy_total": g["total"], "greedy_wt_h": g["weighted_tardiness_h"],
                 "cp_total": c["total"], "bound": res["bound"], "gap": 1 - res["bound"] / res["objective"],
                 "status": res["status"], "wall_s": res["wall"],
                 "edd_wt_h": e["weighted_tardiness_h"], "atcs_wt_h": a["weighted_tardiness_h"],
                 "cp_wt_h": c["weighted_tardiness_h"],
                 "saving_vs_best_rule": 1 - c["total"] / min(a["total"], e["total"], g["total"])})
    print(rows[-1], flush=True)
    save_json("exp5_scaling.json", {"budget_det": BUDGET, "rows": rows})

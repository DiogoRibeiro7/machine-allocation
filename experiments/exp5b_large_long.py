"""Experiment 5b: the 120-job instance again with six times the solver budget."""
from common import *  # noqa: F401,F403

from sched.heuristics import best_baseline
from sched.instance import int_times, make_instance, make_plant
from sched.model import solve_isolated as solve
from sched.simulate import evaluate

n = 120
inst = make_instance(n, horizon_h=120.0 * n / 50.0, jobs_seed=JOBS_SEED + n, plant=make_plant(PLANT_SEED))
P = int_times(inst, inst.true_mean)
hint, name = best_baseline(inst, P, PEN)
h = evaluate(inst, hint, P, PEN)
res = solve(inst, P, time_limit=MAIN_BUDGET, penalty_per_min=PEN, hint_seqs=hint, trace_progress=True)
c = evaluate(inst, res["seqs"], P, PEN)
out = {"jobs": n, "budget_det": MAIN_BUDGET, "warm_start": name, "start_total": h["total"], "cp_total": c["total"],
       "cp_wt_h": c["weighted_tardiness_h"], "bound": res["bound"], "gap": 1 - res["bound"] / res["objective"],
       "saving_vs_best_rule": 1 - c["total"] / h["total"], "wall_s": res["wall"], "status": res["status"],
       "trace": res["trace"]}
print({k: v for k, v in out.items() if k != "trace"})
save_json("exp5b_large_long.json", out)

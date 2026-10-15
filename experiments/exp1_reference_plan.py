"""Experiment 1: the reference week. EDD list rule, tuned ATCS dispatching, CP-SAT."""
from common import JOBS_SEED, MAIN_BUDGET, N_JOBS, PEN, PENALTY_PER_H, PLANT_SEED, save_json, save_pickle, scalars
from machine_allocation.heuristics import atcs_tuned, edd_list, greedy_cost
from machine_allocation.instance import describe, int_times, make_instance, make_plant
from machine_allocation.model import solve_isolated as solve
from machine_allocation.simulate import evaluate, utilisation

inst = make_instance(N_JOBS, jobs_seed=JOBS_SEED, plant=make_plant(PLANT_SEED))
P = int_times(inst, inst.true_mean)
out = {"instance": describe(inst), "penalty_per_weighted_hour": PENALTY_PER_H}

plans = {}
plans["EDD list"] = edd_list(inst, P)
plans["ATCS (tuned)"], par = atcs_tuned(inst, P, PEN)
out["atcs_params"] = par
plans["Cost-aware greedy"] = greedy_cost(inst, P, PEN)
start = min(plans, key=lambda k: evaluate(inst, plans[k], P, PEN)["total"])
out["warm_start"] = start
res = solve(inst, P, time_limit=MAIN_BUDGET, penalty_per_min=PEN, hint_seqs=plans[start],
            trace_progress=True)
plans["CP-SAT"] = res["seqs"]
out["solver"] = {"status": res["status"], "wall_s": res["wall"], "objective": res["objective"],
                 "bound": res["bound"], "gap": 1 - res["bound"] / res["objective"],
                 "budget_det": MAIN_BUDGET, "trace": res["trace"]}
out["policies"] = {}
for name, seqs in plans.items():
    ev = evaluate(inst, seqs, P, PEN)
    row = scalars(ev)
    row["utilisation"] = utilisation(inst, ev)
    out["policies"][name] = row
    print(name, {k: round(v, 1) for k, v in scalars(ev).items()})
print(out["solver"]["status"], out["solver"]["wall_s"], out["solver"]["objective"], out["solver"]["bound"])
save_json("exp1_reference_plan.json", out)
save_pickle("exp1_plans.pkl", plans)

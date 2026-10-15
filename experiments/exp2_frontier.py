"""Experiment 2: what an hour of lateness is worth. Sweep the lateness penalty and trace
the trade-off between operating cost and weighted tardiness.

VOLUME=1.1 python exp2_frontier.py  repeats the sweep with every job 10% larger, where
capacity binds; the grid is finer there because that is where the curve bends.
"""
import dataclasses

from common import *  # noqa: F401,F403

from sched.heuristics import atcs_tuned, edd_list, greedy_cost
from sched.instance import int_times, make_instance, make_plant
from sched.model import solve_isolated as solve
from sched.simulate import evaluate

VOLUME = float(os.environ.get("VOLUME", "1.0"))
inst = make_instance(N_JOBS, jobs_seed=JOBS_SEED, plant=make_plant(PLANT_SEED))
if VOLUME == 1.0:
    P = int_times(inst, inst.true_mean)
    base = load_pickle("exp1_plans.pkl")
    LAMBDAS = [2, 10, 30, 60, 120, 300, 1200]          # EUR per weighted hour late
    pool = {"EDD list": base["EDD list"], "ATCS (tuned)": base["ATCS (tuned)"],
            "Cost-aware greedy": base["Cost-aware greedy"], "ref": base["CP-SAT"]}
    OUT = "exp2_frontier.json"
else:
    inst = dataclasses.replace(inst, true_mean=inst.true_mean * VOLUME, nominal=inst.nominal * VOLUME,
                               qty=inst.qty * VOLUME)
    P = int_times(inst, inst.true_mean)
    LAMBDAS = [2, 5, 10, 20, 30, 60, 120, 300, 1200]
    pool = {"EDD list": edd_list(inst, P), "ATCS (tuned)": atcs_tuned(inst, P, PEN)[0],
            "Cost-aware greedy": greedy_cost(inst, P, PEN)}
    OUT = f"exp2_frontier_v{round(VOLUME * 100)}.json"
rows = []
for lam in LAMBDAS:
    pen = lam / 60.0
    cand = dict(pool)
    cand[f"atcs@{lam}"] = atcs_tuned(inst, P, pen)[0]
    cand[f"greedy@{lam}"] = greedy_cost(inst, P, pen)
    pool[f"greedy@{lam}"] = cand[f"greedy@{lam}"]
    hint = min(cand.values(), key=lambda s: evaluate(inst, s, P, pen)["total"])
    res = solve(inst, P, time_limit=budget(30.0), penalty_per_min=pen, hint_seqs=hint)
    pool[f"cp@{lam}"] = res["seqs"]
    ev = evaluate(inst, res["seqs"], P, pen)
    rows.append({"lambda": lam, "objective": res["objective"], "bound": res["bound"],
                 "gap": 1 - res["bound"] / res["objective"], "wall_s": res["wall"], **scalars(ev)})
    print(lam, round(ev["operating_cost"]), round(ev["weighted_tardiness_h"], 1),
          ev["late_jobs"], round(rows[-1]["gap"], 3), flush=True)

points = {}
for name, seqs in pool.items():
    ev = evaluate(inst, seqs, P, 1.0)
    points[name] = {"operating_cost": ev["operating_cost"], "weighted_tardiness_h": ev["weighted_tardiness_h"],
                    "setup_h": ev["setup_h"], "run_cost": ev["run_cost"], "late_jobs": ev["late_jobs"]}
cp = {k: v for k, v in points.items() if k.startswith("cp@") or k == "ref"}
front = [k for k, v in cp.items() if not any(
    (w["operating_cost"] <= v["operating_cost"] and w["weighted_tardiness_h"] <= v["weighted_tardiness_h"]
     and (w["operating_cost"] < v["operating_cost"] or w["weighted_tardiness_h"] < v["weighted_tardiness_h"]))
    for w in cp.values())]
save_json(OUT, {"volume": VOLUME, "sweep": rows, "points": points, "nondominated": front})
print("non-dominated:", front)

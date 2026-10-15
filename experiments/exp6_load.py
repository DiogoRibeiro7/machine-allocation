"""Experiment 6: the same order book at different volumes. How much of the planned
punctuality survives run-to-run variability as the machines fill up."""
import dataclasses

from common import *  # noqa: F401,F403

from sched.heuristics import best_baseline
from sched.instance import int_times, make_instance, make_plant
from sched.model import solve_isolated as solve
from sched.simulate import evaluate, execute, score, utilisation

R = reps(2000)
plant = make_plant(PLANT_SEED)
base = make_instance(N_JOBS, jobs_seed=JOBS_SEED, plant=plant)
z = np.random.default_rng(2024).normal(size=(R, base.N))
rows = []
for phi in (0.70, 0.80, 0.90, 1.00, 1.10):
    inst = dataclasses.replace(base, true_mean=base.true_mean * phi, nominal=base.nominal * phi,
                               qty=base.qty * phi)
    P = int_times(inst, inst.true_mean)
    hint, _ = best_baseline(inst, P, PEN)
    res = solve(inst, P, time_limit=SIDE_BUDGET, penalty_per_min=PEN, hint_seqs=hint)
    ev = evaluate(inst, res["seqs"], P, PEN)
    a = ev["assign"]
    s = plant.sigma[a]
    mu = inst.true_mean[np.arange(inst.N), a]
    wt, tot = np.zeros(R), np.zeros(R)
    for r in range(R):
        dur = mu * np.exp(s * z[r] - 0.5 * s ** 2)
        st, en, su, aa = execute(inst, res["seqs"], dur)
        sc = score(inst, en, su, aa, dur, PEN)
        wt[r], tot[r] = sc["weighted_tardiness_h"], sc["total"]
    u = utilisation(inst, ev)
    rows.append({"volume": phi, "utilisation_mean": float(u.mean()), "utilisation_max": float(u.max()),
                 "planned_wt_h": ev["weighted_tardiness_h"], "planned_total": ev["total"],
                 "bound": res["bound"], "real_wt_mean": wt.mean(), "real_wt_p90": np.quantile(wt, 0.9),
                 "real_wt_p10": np.quantile(wt, 0.1), "real_total_mean": tot.mean(),
                 "planned_late": ev["late_jobs"]})
    print({k: round(float(v), 3) for k, v in rows[-1].items()}, flush=True)
    save_json("exp6_load.json", {"rows": rows, "replications": R})

"""Experiment 3: processing times are estimates. Plan with five different duration
matrices, then execute every plan against the same simulated reality.

Each estimator is planned three times with different solver seeds, because two solves of
the same instance land on different near-optimal plans and that variation has to be told
apart from the effect of the estimates. Finished solves are cached, so the script can be
interrupted and resumed.
"""
from common import *  # noqa: F401,F403

from sched.heuristics import atcs, atcs_tuned, best_baseline, greedy_cost
from sched.instance import draw_jobs, int_times, make_instance, make_plant, times_for
from sched.model import solve_isolated as solve
from sched.simulate import evaluate, execute, score
from sched.times import Estimators, make_history

BUDGET = budget(80.0)
R = reps(2000)
SEEDS = (1, 2, 3)
CACHE = RESULTS / "exp3_cache"
CACHE.mkdir(exist_ok=True)

plant = make_plant(PLANT_SEED)
inst = make_instance(N_JOBS, jobs_seed=JOBS_SEED, plant=plant)
hist = make_history(plant)
est = Estimators(plant, hist)
P_true = inst.true_mean
out = {"history_runs": len(hist["actual"]), "cell_counts": est.count, "smear": est.smear,
       "rare_cells": hist["rare_cells"], "budget_det": BUDGET, "replications": R, "seeds": SEEDS}

# --- prediction accuracy on fresh jobs -------------------------------------------------
rng = np.random.default_rng(101)
f2, q2, g2 = draw_jobs(plant, 5000, rng)
_, tm2 = times_for(plant, f2, q2, g2)
el2 = plant.elig_mf[:, f2].T
KINDS = ["nameplate", "cell mean", "boosted mean", "boosted q75"]
mats = {k: est.matrix(k, inst.family, inst.qty, inst.grade) for k in KINDS}
mats["oracle mean"] = P_true
out["accuracy"] = {}
for k in KINDS:
    e = np.log(est.matrix(k, f2, q2, g2)[el2] / tm2[el2])
    out["accuracy"][k] = {"mape": float(np.mean(np.abs(np.exp(e) - 1))), "mean_log_error": float(e.mean()),
                          "rmse_log": float(np.sqrt((e ** 2).mean()))}
    print(k, out["accuracy"][k], flush=True)

# --- execution against sampled run times -----------------------------------------------
z = np.random.default_rng(2024).normal(size=(R, inst.N))      # common random numbers
sig = plant.sigma
rows = np.arange(inst.N)


def realise(seqs, assign):
    tot, wt, late = np.zeros(R), np.zeros(R), np.zeros(R)
    s = sig[assign]
    mu = P_true[rows, assign]
    for r in range(R):
        dur = mu * np.exp(s * z[r] - 0.5 * s ** 2)
        st, en, su, a = execute(inst, seqs, dur)
        sc = score(inst, en, su, a, dur, PEN)
        tot[r], wt[r], late[r] = sc["total"], sc["weighted_tardiness_h"], sc["late_jobs"]
    return tot, wt, late


def summarise(seqs, Pk):
    """Promise, cost at the true means, and realised cost of one plan."""
    Pi = int_times(inst, Pk)
    planned = evaluate(inst, seqs, Pi, PEN)
    at_truth = evaluate(inst, seqs, P_true, PEN)
    a = planned["assign"]
    tot, wt, late = realise(seqs, a)
    lr = np.log(P_true / np.where(inst.eligible, Pk, 1.0))
    w = inst.weight
    return {
        # Priority weight of the jobs planned to finish less than two hours before they
        # are due: as the plan believes, and as the true mean times would have it.
        "tight_weight_plan": int(w[inst.due - planned["end"] < 120].sum()),
        "tight_weight_true": int(w[inst.due - at_truth["end"] < 120].sum()),
        "planned_total": planned["total"], "planned_wt_h": planned["weighted_tardiness_h"],
        "truth_total": at_truth["total"], "truth_wt_h": at_truth["weighted_tardiness_h"],
        "real_total_mean": tot.mean(), "real_total_p90": np.quantile(tot, 0.9),
        "real_wt_mean": wt.mean(), "real_wt_p90": np.quantile(wt, 0.9), "real_late_mean": late.mean(),
        "underestimate_all_pairs": float(np.exp(lr[inst.eligible].mean()) - 1),
        "underestimate_chosen_pairs": float(np.exp(lr[rows, a].mean()) - 1),
        "share_minutes_on_old": float(P_true[rows, a][a >= 4].sum() / P_true[rows, a].sum()),
    }, tot, wt


FIELDS = ["planned_total", "planned_wt_h", "truth_total", "truth_wt_h", "real_total_mean", "real_total_p90",
          "real_wt_mean", "real_wt_p90", "real_late_mean", "underestimate_chosen_pairs", "share_minutes_on_old",
          "tight_weight_plan", "tight_weight_true"]
out["plans"] = {}
for k, Pk in mats.items():
    Pi = int_times(inst, Pk)
    hint, hint_name = best_baseline(inst, Pi, PEN)
    per_seed, tots, wts = [], [], []
    for seed in SEEDS:
        f = CACHE / f"{k.replace(' ', '_')}_seed{seed}.pkl"
        if f.exists() and not QUICK:
            res = pickle.loads(f.read_bytes())
        else:
            res = solve(inst, Pi, time_limit=BUDGET, penalty_per_min=PEN, hint_seqs=hint, seed=seed)
            res = {q: res[q] for q in ("seqs", "bound", "wall", "status", "objective")}
            f.write_bytes(pickle.dumps(res))
        row, tot, wt = summarise(res["seqs"], Pk)
        row.update(seed=seed, bound=res["bound"], warm_start=hint_name)
        per_seed.append(row)
        tots.append(tot)
        wts.append(wt)
        print(k, seed, {q: round(float(row[q]), 3) for q in FIELDS}, flush=True)
    out["plans"][k] = {
        "per_seed": per_seed,
        "mean": {q: float(np.mean([r[q] for r in per_seed])) for q in FIELDS},
        "min": {q: float(np.min([r[q] for r in per_seed])) for q in FIELDS},
        "max": {q: float(np.max([r[q] for r in per_seed])) for q in FIELDS},
        "underestimate_all_pairs": per_seed[0]["underestimate_all_pairs"],
        "total_samples": np.concatenate(tots), "wt_samples": np.concatenate(wts),
    }

# --- the reference plan of experiment 1 (true means, three times the budget) ------------
ref = load_pickle("exp1_plans.pkl")["CP-SAT"]
row, tot, wt = summarise(ref, P_true)
out["reference_plan"] = row
print("reference plan", {q: round(float(row[q]), 3) for q in FIELDS}, flush=True)

# --- the cost-aware list rule, planned on the boosted estimates and executed as is ------
Pi = int_times(inst, mats["boosted mean"])
row, tot, wt = summarise(greedy_cost(inst, Pi, PEN), mats["boosted mean"])
out["greedy_plan"] = row
print("greedy plan", {q: round(float(row[q]), 3) for q in FIELDS}, flush=True)

# --- a reactive rule on the same reality ------------------------------------------------
_, par = atcs_tuned(inst, Pi, PEN)
tot, wt, late = np.zeros(R), np.zeros(R), np.zeros(R)
for r in range(R):
    A = P_true * np.exp(sig[None, :] * z[r][:, None] - 0.5 * sig[None, :] ** 2)
    seqs = atcs(inst, Pi, par["k1"], par["k2"], actual=A)
    assign = np.zeros(inst.N, dtype=int)
    for m, sq in seqs.items():
        for it in sq:
            if it >= 0:
                assign[it] = m
    dur = A[rows, assign]
    st, en, su, a = execute(inst, seqs, dur)
    sc = score(inst, en, su, a, dur, PEN)
    tot[r], wt[r], late[r] = sc["total"], sc["weighted_tardiness_h"], sc["late_jobs"]
out["online_atcs"] = {"params": par, "real_total_mean": tot.mean(), "real_total_p90": np.quantile(tot, 0.9),
                      "real_wt_mean": wt.mean(), "real_wt_p90": np.quantile(wt, 0.9),
                      "real_late_mean": late.mean()}
print("online ATCS", round(tot.mean()), round(wt.mean(), 1), flush=True)
save_json("exp3_run_times.json", out)

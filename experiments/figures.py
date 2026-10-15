"""Draw every figure in the article from the saved results."""
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import FIGURES, JOBS_SEED, N_JOBS, PEN, PLANT_SEED, load_json, load_pickle
from machine_allocation.instance import FAMILIES, MACHINES, int_times, make_instance, make_plant
from machine_allocation.simulate import evaluate

matplotlib.use("Agg")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]      # categorical slots 1-5
NEUTRAL = "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
})
CREDIT = "Figure by the author; data simulated (see repository)."


def finish(fig, name, credit_y=0.005):
    fig.text(0.995, credit_y, CREDIT, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def euro(x, _=None):
    return f"€{x:,.0f}"


# --------------------------------------------------------------------------------------
def fig_gantt():
    inst = make_instance(N_JOBS, jobs_seed=JOBS_SEED, plant=make_plant(PLANT_SEED))
    P = int_times(inst, inst.true_mean)
    plans = load_pickle("exp1_plans.pkl")
    res = load_json("exp1_reference_plan.json")["policies"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    for ax, name in zip(axes, ["Cost-aware greedy", "CP-SAT"], strict=True):
        ev = evaluate(inst, plans[name], P, PEN)
        for j in range(inst.N):
            m, s, e, su = ev["assign"][j], ev["start"][j] / 60, ev["end"][j] / 60, ev["setup"][j] / 60
            y = inst.M - 1 - m
            ax.barh(y, su - 0.08, left=s - su + 0.04, height=0.30, color=NEUTRAL, zorder=2)
            ax.barh(y, (e - s) - 0.16, left=s + 0.08, height=0.62, color=C[inst.family[j]], zorder=3)
            if e - s > 3.2:
                ax.text((s + e) / 2, y, FAMILIES[inst.family[j]], ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold", zorder=4)
            if e > inst.due[j] / 60 + 1e-6:
                ax.plot([(s + e) / 2], [y + 0.43], marker="v", ms=5, color=INK, zorder=5, lw=0)
        for m, wins in inst.windows.items():
            for a, b in wins:
                ax.barh(inst.M - 1 - m, (b - a) / 60, left=a / 60, height=0.62, color=SURFACE,
                        edgecolor=MUTED, hatch="////", lw=0.6, zorder=3)
        ax.axvline(inst.horizon / 60, color=MUTED, lw=1)
        if name == "Cost-aware greedy":
            ax.text(inst.horizon / 60 + 0.8, -0.5, "end of week", fontsize=8, color=INK2, va="bottom")
        r = res[name]
        ax.set_title(f"{name}:  €{r['total']:,.0f} total  ·  {r['weighted_tardiness_h']:.1f} weighted hours late"
                     f"  ·  {r['late_jobs']} late jobs  ·  {r['setup_h']:.1f} h of changeovers")
        ax.set_yticks(range(inst.M))
        ax.set_yticklabels(MACHINES[::-1])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 136)
        ax.set_ylim(-0.6, inst.M - 0.3)
        ax.tick_params(length=0)
    axes[1].set_xlabel("Hour of the week")
    axes[1].set_xticks(range(0, 121, 24))
    handles = [Patch(color=C[f], label=f"Family {FAMILIES[f]}") for f in range(5)]
    handles += [Patch(color=NEUTRAL, label="Changeover"),
                Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="////", label="Planned maintenance"),
                Line2D([], [], marker="v", color=INK, lw=0, ms=5, label="Finishes after its due date")]
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.005), fontsize=8.5,
               handlelength=1.2, columnspacing=1.3)
    fig.subplots_adjust(hspace=0.22, bottom=0.12)
    finish(fig, "fig1_gantt.png", credit_y=-0.035)


def fig_anytime():
    d = load_json("exp1_reference_plan.json")
    tr = np.array(d["solver"]["trace"]).reshape(-1, 3)
    if not len(tr):          # the solver never improved on its warm start (QUICK=1 budgets)
        tr = np.array([[0.0, d["solver"]["objective"], d["solver"]["bound"]]])
    wall = d["solver"]["wall_s"]
    t = np.append(tr[:, 0], wall) / 60
    obj = np.append(tr[:, 1], tr[-1, 1])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.step(t, obj, where="post", color=C[0], lw=2, label="Best plan found so far")
    ax.step(t, np.append(tr[:, 2], d["solver"]["bound"]), where="post", color=INK2, lw=2,
            label="Proven lower bound")
    for name, col in (("EDD list", C[1]), ("ATCS (tuned)", C[2]), ("Cost-aware greedy", C[3])):
        v = d["policies"][name]["total"]
        ax.axhline(v, color=col, lw=2)
        ax.text(wall / 60, v + 130, f"{name}  {euro(v)}", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.text(wall / 60, obj[-1] + 150, f"CP-SAT  {euro(obj[-1])}", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.text(wall / 60, d["solver"]["bound"] - 180, f"bound  {euro(d['solver']['bound'])}", ha="right", va="top",
            fontsize=9, color=INK2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(euro))
    ax.set_xlabel("Solver wall-clock time, minutes (two cores)")
    ax.set_ylabel("Weekly cost of the plan")
    ax.set_title("The plan improves for as long as the solver is allowed to run")
    ax.set_xlim(0, wall / 60 * 1.01)
    ax.set_ylim(36000, 49000)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.59))
    finish(fig, "fig2_anytime.png", credit_y=-0.03)


def fig_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (fname, title) in zip(axes, [("exp2_frontier.json", "Reference order book"),
                                         ("exp2_frontier_v110.json", "Every job 10% larger")], strict=True):
        d = load_json(fname)
        pts = d["points"]
        lams = sorted(r["lambda"] for r in d["sweep"])
        gx = [pts[f"greedy@{lam}"]["weighted_tardiness_h"] for lam in lams]
        gy = [pts[f"greedy@{lam}"]["operating_cost"] for lam in lams]
        front = sorted(d["nondominated"], key=lambda k: pts[k]["weighted_tardiness_h"])
        cx = [pts[k]["weighted_tardiness_h"] for k in front]
        cy = [pts[k]["operating_cost"] for k in front]
        ax.plot(cx, cy, color=C[0], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=4,
                label="CP-SAT, best plans found in the sweep")
        ax.plot(gx, gy, color=C[1], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3,
                label="Cost-aware greedy, one plan per penalty")
        for lam, x, y in zip(lams, gx, gy, strict=True):
            if lam in (lams[0], 30, lams[-1]):
                ax.annotate(f"€{lam:,}/h", (x, y), xytext=(8, 4), textcoords="offset points", fontsize=8.5, color=INK2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(euro))
        ax.set_xlabel("Weighted hours late")
        ax.set_title(title)
        ax.set_xlim(left=-15)
    axes[0].set_ylabel("Operating cost of the week")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(wspace=0.22, bottom=0.14)
    finish(fig, "fig3_frontier.png", credit_y=-0.12)


EST = [("nameplate", "Nameplate rates"), ("cell mean", "Historical cell mean"),
       ("boosted mean", "Boosted model, mean"), ("boosted q75", "Boosted model, 75th percentile"),
       ("oracle mean", "True mean times")]


def fig_times():
    d = load_json("exp3_run_times.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ys = np.arange(len(EST))[::-1]
    panels = [("planned_total", "real_total_mean", "total_samples", "Weekly cost", euro),
              ("planned_wt_h", "real_wt_mean", "wt_samples", "Weighted hours late", lambda v, _=None: f"{v:,.0f}")]
    for ax, (pk, rk, sk, title, fmt) in zip(axes, panels, strict=True):
        for y, (k, _) in zip(ys, EST, strict=True):
            r = d["plans"][k]
            smp = np.array(r[sk])
            lo, hi = np.quantile(smp, 0.1), np.quantile(smp, 0.9)
            ax.plot([lo, hi], [y, y], color=C[0], lw=2, solid_capstyle="round", zorder=2)
            ax.plot([r["mean"][rk]], [y], marker="o", ms=10, color=C[0], mec=SURFACE, mew=2, zorder=4)
            ax.plot([q[rk] for q in r["per_seed"]], [y + 0.26] * len(r["per_seed"]), marker="|", ms=8,
                    mew=1.6, color=INK2, lw=0, zorder=3)
            ax.plot([r["mean"][pk]], [y], marker="D", ms=8, color=C[1], mec=SURFACE, mew=2, zorder=5)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(EST) - 0.3)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab in EST])
    handles = [Line2D([], [], marker="D", color=C[1], lw=0, ms=8, mec=SURFACE, mew=2, label="Promised by the plan"),
               Line2D([], [], marker="o", color=C[0], lw=2, ms=10, mec=SURFACE, mew=2,
                      label="Delivered: mean and 10th to 90th percentile of the simulated weeks"),
               Line2D([], [], marker="|", color=INK2, lw=0, ms=8, mew=1.6, label="Delivered mean, each solver seed")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.subplots_adjust(wspace=0.08, bottom=0.14)
    finish(fig, "fig4_run_times.png", credit_y=-0.12)


def fig_slack():
    d = load_json("exp3_run_times.json")
    marks = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for (k, lab), col, mk in zip(EST, C, marks, strict=True):
        rs = d["plans"][k]["per_seed"]
        ax.plot([r["tight_weight_true"] for r in rs], [r["real_wt_mean"] for r in rs], marker=mk, ms=9, color=col,
                mec=SURFACE, mew=2, lw=0, label=lab, zorder=3)
    for key, lab, dx in (("reference_plan", "reference plan", 8), ("greedy_plan", "greedy plan", 8)):
        r = d[key]
        ax.plot([r["tight_weight_true"]], [r["real_wt_mean"]], marker="X", ms=9, color=INK2, mec=SURFACE, mew=2,
                lw=0, zorder=3)
        ax.annotate(lab, (r["tight_weight_true"], r["real_wt_mean"]), xytext=(-dx, -3), textcoords="offset points",
                    fontsize=8.5, color=INK2, ha="right")
    ax.set_xlabel("Priority weight of jobs that would finish less than two hours before their due date\n"
                  "if every run took exactly its true mean time")
    ax.set_ylabel("Weighted hours late, mean of 2,000 simulated weeks")
    ax.set_title("Delivered lateness follows the number of jobs planned without slack")
    ax.set_ylim(bottom=-1.5)
    ax.set_xlim(left=-0.8)
    ax.legend(loc="upper left", title="Plan built on", alignment="left")
    finish(fig, "fig5_slack.png", credit_y=-0.07)


def fig_load():
    d = load_json("exp6_load.json")["rows"]
    u = np.array([r["utilisation_mean"] for r in d]) * 100
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.fill_between(u, [r["real_wt_p10"] for r in d], [r["real_wt_p90"] for r in d], color=C[1], alpha=0.12, lw=0)
    ax.plot(u, [r["planned_wt_h"] for r in d], color=C[0], lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
            label="Planned")
    ax.plot(u, [r["real_wt_mean"] for r in d], color=C[1], lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
            label="Delivered, mean of 2,000 weeks (band: 10th to 90th percentile)")
    for r, x in zip(d, u, strict=True):
        ax.annotate(f"{r['volume']:.0%} volume", (x, r["real_wt_mean"]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Mean planned machine utilisation, %")
    ax.set_ylabel("Weighted hours late")
    ax.set_title("Lateness arrives before the machines are full")
    ax.set_ylim(bottom=-3)
    ax.set_xlim(right=u.max() + 3)
    ax.legend(loc="upper left")
    finish(fig, "fig6_load.png", credit_y=-0.03)


FIGS = {"gantt": fig_gantt, "anytime": fig_anytime, "frontier": fig_frontier, "times": fig_times,
        "slack": fig_slack, "load": fig_load}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    unknown = [w for w in which if w not in FIGS]
    if unknown:
        sys.exit(f"unknown figure(s): {', '.join(unknown)}; choose from {', '.join(FIGS)}")
    for w in which:
        FIGS[w]()

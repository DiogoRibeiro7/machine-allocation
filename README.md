# machine-allocation

Companion code for the article *Machine Allocation on a Production Line*.

A simulated plant with six unrelated parallel machines, machine eligibility, material
release dates, sequence-dependent changeovers and planned maintenance; a CP-SAT model that
minimises lateness penalties plus operating cost; three constructive rules to compare it
with; and five experiments around it.

```
sched/
  instance.py     plant and order-book generator
  model.py        CP-SAT model (optional intervals + circuit constraint per machine)
  heuristics.py   EDD list rule, tuned ATCS dispatching, cost-aware greedy
  simulate.py     order-preserving execution and scoring
  times.py        production log and the four processing-time estimators
experiments/
  exp1_reference_plan.py   the reference week: three rules and the solver
  exp2_frontier.py         operating cost against lateness, by penalty (VOLUME=1.1 for the larger order book)
  exp3_run_times.py        plans built on five processing-time matrices, three solver seeds each, executed
  exp4_breakdown.py        a machine fails mid-week: right-shift or re-optimise
  exp5_scaling.py          20 to 120 jobs at a fixed solver budget
  exp5b_large_long.py      the 120-job instance with six times the budget
  exp6_load.py             the same order book at different volumes
  figures.py               every figure
results/                   JSON written by the experiments
figures/                   PNG written by figures.py
```

## Reproducing

```
pip install -r requirements.txt
./run_all.sh
python experiments/figures.py
```

`QUICK=1 ./run_all.sh` is a smoke test with tiny budgets; it writes to `results_quick/`.

The solver runs with four interleaved workers and a deterministic-time budget, so a run
returns the same plans on any machine with the same OR-Tools version (9.15 here). Wall-clock
time is machine-dependent: the full set is 49 solves and took just under three hours on two
cores. `exp3_run_times.py` caches each finished solve in `results/exp3_cache/` and can be
interrupted and resumed; delete that folder to recompute.

Every solve runs in its own process (`solve_isolated` in `sched/model.py`) and is retried if
the process dies. That is not decoration: the native solver aborted twice with a memory
fault while these results were being produced, and both times the retry returned the plan.

All data are simulated. The plant parameters are in `sched/instance.py`.

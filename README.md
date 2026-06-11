# agentrel

**Reliability & reproducibility statistics for stochastic agent / tool-use evals.**

[![CI](https://github.com/yongzhe2160cs/agent-eval-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/yongzhe2160cs/agent-eval-reliability/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

## The problem

Agent / tool-use evals — multi-turn, partial-credit, trajectory-scored — are the
hottest and *least reproducible* corner of LLM evaluation. The same agent on the
same task scores differently run to run, because the agent is stochastic. That
breaks the habits people carry over from single-shot benchmarks:

- **A single run is a coin flip, not a measurement.** Reporting one pass/fail per
  task throws away the run-to-run variance that dominates these evals.
- **A headline number with no interval invites false conclusions.** "Agent A beats
  agent B by 3 points" is meaningless without knowing the 3 points clears the
  noise — and partial-credit, multi-run scores aren't binomial, so the textbook
  proportion CI doesn't apply.
- **`pass@k` quietly conflates capability with luck.** Best-of-k looks great while
  the agent silently fails most of the time; the reliability question ("does it
  succeed *every* time?") needs `pass^k`, which is rarely reported.
- **"How many runs do I need?" usually goes unasked**, so half the per-task pass
  rates in a report have confidence intervals too wide to support any claim.

`agentrel` is a small, framework-agnostic library that makes all of this explicit:
it decomposes how much of a score is the agent vs. luck, attaches honest
confidence intervals to every number, does paired model-vs-model comparison with
multiple-comparison control, and stamps results for reproducibility.

> **Keyless & offline.** `agentrel` never calls a model. It analyzes eval results
> you already have (or the built-in `simulate_agent_runs()` generator). Bring your
> own runs via the JSON/CSV or Inspect-log adapters.

## Install

```bash
git clone https://github.com/yongzhe2160cs/agent-eval-reliability
cd agent-eval-reliability
uv venv && uv pip install -e ".[dev]"
```

## Worked example

`examples/synthetic_runs.json` is a synthetic multi-run agent-eval log: 16 tasks,
10 runs each, partial-credit scores in `[0, 1]`. Load it and ask for a report:

```python
import agentrel as ar

runs = ar.from_json("examples/synthetic_runs.json", agent="demo-agent")
print(ar.reliability_report(runs, ks=(1, 2, 5)).summary())
```

```text
Reliability report — agent='demo-agent'
  design: 16 tasks x ~10.0 runs (160 runs)
  mean score:   0.6261 [0.4951, 0.7554] (95% CI, mean-score cluster-bootstrap)
  variance:     between-task=0.0577  within-task(luck)=0.0849
  ICC(1):       0.404  (luck-dominated; single runs are noisy)
  pass@1:       0.4688 [0.2938, 0.6438] (95% CI, pass@1 cluster-bootstrap)
  pass@2:       0.6181 [0.4194, 0.7833] (95% CI, pass@2 cluster-bootstrap)
  pass@5:       0.8090 [0.5816, 0.9187] (95% CI, pass@5 cluster-bootstrap)
  pass^1:       0.4688 [0.2938, 0.6438] (95% CI, pass^1 cluster-bootstrap)
  pass^2:       0.3194 [0.1556, 0.5223] (95% CI, pass^2 cluster-bootstrap)
  pass^5:       0.1947 [0.0506, 0.4137] (95% CI, pass^5 cluster-bootstrap)
  flakiness:    6/16 tasks have a pass-rate CI wider than 0.4 (too few runs to trust)
  provenance:   input=e1b47bf4082d5e13 numpy=2.4.6 agentrel=0.1.0 seed=0
```

How to read it:

- **ICC(1) = 0.40** — only ~40% of the score variance is the agent's consistent
  per-task skill; the other 60% is run-to-run luck. A *single* run on this suite
  is genuinely noisy, and the verdict line says so.
- **`pass@5` (0.81) vs `pass^5` (0.19)** — best-of-5 looks strong, but the agent
  almost never solves a task on all 5 attempts. That gap *is* the reliability
  story, and it is invisible if you only report `pass@k`.
- **Every number carries a 95% CI** computed by hierarchical bootstrap (resampling
  tasks *and* runs), because partial-credit multi-run scores are not binomial.
- **6 of 16 tasks are flagged flaky** — their pass-rate CI is wider than 0.4, i.e.
  too few runs to trust. Those are the first place to spend more compute.

Full runnable walkthrough — ingest → report → power → flakiness → paired
comparison → determinism check — is in [`examples/demo.py`](examples/demo.py):

```bash
uv run python examples/demo.py
```

## Comparing two agents

Paired comparison on a shared task set, with multiple-comparison control across
tasks:

```python
import agentrel as ar

a = ar.simulate_agent_runs(n_tasks=40, runs_per_task=10, base_skill=0.62, agent="candidate", seed=11)
b = ar.simulate_agent_runs(n_tasks=40, runs_per_task=10, base_skill=0.45, agent="baseline", seed=11)

cmp = ar.compare_agents(a, b)
print(cmp.paired.delta)            # 0.1214 [0.0788, 0.1642] (95% CI, paired-delta bootstrap)
print(cmp.paired.significant)      # True  (paired t-test on per-task means)
print(cmp.n_sig_holm, cmp.n_sig_bh)  # per-task discoveries after Holm / BH correction
```

The aggregate paired delta answers *"is A better overall?"* (respecting the
pairing — same tasks). The per-task tests answer *"on which tasks?"* — and are
corrected by both **Holm** (controls family-wise error) and **Benjamini-Hochberg**
(controls false discovery rate), so testing 40 tasks at once doesn't manufacture
false positives.

## What's in the box

| Area | API | What it gives you |
|------|-----|-------------------|
| **Data model** | `RunSet`, `TaskRun` | per-task, multi-run results with partial-credit scores |
| **Ingest** | `from_json`, `from_csv`, `from_records`, `from_inspect_log` | generic adapters + an Inspect-AI `.eval`-log stub |
| **Variance** | `variance_components`, `icc` | one-way random-effects decomposition; ICC(1) = "agent vs luck" |
| **Coverage** | `pass_at_k`, `pass_hat_k` | unbiased best-of-k *and* all-of-k, with bootstrap CIs |
| **Uncertainty** | `mean_score_ci` | cluster-bootstrap CI for the task-averaged mean score |
| **Power** | `min_runs_for_ci_width` | minimum runs/task for a target CI half-width |
| **Comparison** | `compare_agents`, `paired_delta`, `holm`, `benjamini_hochberg` | paired delta + per-task tests with multiplicity control |
| **Reproducibility** | `stamp`, `determinism_check`, `flakiness_report` | provenance, pipeline determinism, flaky-task flagging |
| **One call** | `reliability_report` | bundles all of the above into a readable summary |
| **Demo data** | `simulate_agent_runs` | synthetic stochastic agent runs (tune ICC via skill-spread vs luck) |

## Methods notes

- **ICC(1)** uses the standard one-way random-effects ANOVA estimator with the
  `n0` correction for unbalanced designs (different #runs per task). Verified
  against a hand-worked example in the tests.
- **`pass@k` / `pass^k`** use the unbiased combinatorial estimators
  (`pass@k = 1 - C(n-c,k)/C(n,k)`, the Chen et al. 2021 / HumanEval form;
  `pass^k = C(c,k)/C(n,k)`), verified against brute-force enumeration over all
  size-`k` subsets. CIs come from a hierarchical (task-then-run) bootstrap.
- **Paired comparison** uses a paired *t*-test and Wilcoxon on per-task mean
  scores plus a paired bootstrap CI; per-task Welch tests feed Holm and BH.
- Partial-credit, multi-run scores are **not binomial**, so aggregate CIs are
  bootstrap-based rather than Wald/Wilson. (Wilson *is* used for per-task binary
  pass-rate flakiness, where the binomial model applies.)

## Why simulation, and what a full empirical version adds

This toolkit is the *statistics layer*; it is deliberately model-free so it stays
keyless and reproducible in CI. `simulate_agent_runs()` stands in for real runs by
modeling the structure that makes agent evals hard (between-task skill spread vs
within-task luck), which is exactly what's needed to unit-test the estimators.

A full empirical study on top would add: real multi-epoch agent runs against a live
model + tool sandbox; a non-stub Inspect adapter reading `.eval` logs directly;
trajectory-level features (tool-call counts, tokens, wall-clock) joined to scores;
and a recommended-N report driven by observed within-task variance. None of that
changes the math here — it just feeds it real data.

## Development

```bash
uv run pytest          # 83 tests, verified against closed-form / brute-force references
uv run ruff check .    # lint
uv run ruff format .   # format
```

## License

MIT — see [LICENSE](LICENSE).

"""Paired comparison of two agent configurations on a shared task set.

Two complementary views:

* **Aggregate paired delta** — the per-task mean-score difference
  ``A - B``, averaged over the shared tasks, with a paired bootstrap CI and a
  paired *t*-test / Wilcoxon significance test. This answers "is A better than
  B overall?" while respecting the pairing (same tasks).
* **Per-task tests with multiplicity correction** — for each task, a
  two-sample test on its runs, then Holm (FWER) and Benjamini-Hochberg (FDR)
  correction across tasks. This answers "*on which tasks* do they differ?"
  without the false-positive blowup of testing many tasks at once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .datamodel import RunSet
from .metrics import Estimate

__all__ = [
    "PairedDelta",
    "TaskComparison",
    "AgentComparison",
    "paired_delta",
    "per_task_tests",
    "holm",
    "benjamini_hochberg",
    "compare_agents",
]


def _scores(runs: RunSet | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(runs, RunSet):
        return runs.scores_by_task()
    return {k: np.asarray(v, dtype=float) for k, v in runs.items()}


def _shared_tasks(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    shared = [t for t in a if t in b]
    if not shared:
        raise ValueError("agents share no common task ids")
    return shared


# --------------------------------------------------------------------------
# multiplicity corrections
# --------------------------------------------------------------------------
def holm(pvalues: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down. Returns (adjusted p-values, reject flags).

    Controls the family-wise error rate. Adjusted p-values are made monotone
    so they can be thresholded at ``alpha`` directly.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return [], []
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    reject = adj <= alpha
    return adj.tolist(), reject.tolist()


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg step-up. Returns (adjusted p-values, reject flags).

    Controls the false discovery rate. Adjusted p-values use the standard
    cumulative-min from the largest p downward.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return [], []
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    reject = adj <= alpha
    return adj.tolist(), reject.tolist()


# --------------------------------------------------------------------------
# aggregate paired delta
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedDelta:
    """Aggregate paired comparison of A vs B on shared tasks."""

    delta: Estimate  # mean per-task (A - B), bootstrap CI
    t_statistic: float
    t_pvalue: float
    wilcoxon_pvalue: float
    n_tasks: int

    @property
    def significant(self) -> bool:
        return self.t_pvalue < self.delta.alpha


def paired_delta(
    a: RunSet | dict[str, np.ndarray],
    b: RunSet | dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = None,
) -> PairedDelta:
    """Paired comparison on per-task mean scores: A - B."""
    sa, sb = _scores(a), _scores(b)
    tasks = _shared_tasks(sa, sb)
    da = np.array([sa[t].mean() for t in tasks])
    db = np.array([sb[t].mean() for t in tasks])
    diff = da - db
    point = float(diff.mean())

    rng = np.random.default_rng(seed)
    m = diff.size
    boot = np.array([diff[rng.integers(0, m, size=m)].mean() for _ in range(n_boot)])
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    est = Estimate(point, lo, hi, alpha=alpha, method="paired-delta bootstrap")

    if m >= 2 and np.std(diff) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_stat, t_p = stats.ttest_rel(da, db)
    else:
        t_stat, t_p = (float("nan"), float("nan"))
    try:
        if np.any(diff != 0):
            with warnings.catch_warnings():
                # near-constant differences trigger a benign precision warning
                warnings.simplefilter("ignore", RuntimeWarning)
                _, w_p = stats.wilcoxon(da, db)
        else:
            w_p = float("nan")
    except ValueError:
        w_p = float("nan")

    return PairedDelta(
        delta=est,
        t_statistic=float(t_stat),
        t_pvalue=float(t_p),
        wilcoxon_pvalue=float(w_p),
        n_tasks=len(tasks),
    )


# --------------------------------------------------------------------------
# per-task tests + multiplicity
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskComparison:
    task_id: str
    mean_a: float
    mean_b: float
    delta: float
    pvalue: float
    p_holm: float
    p_bh: float
    reject_holm: bool
    reject_bh: bool


@dataclass(frozen=True)
class AgentComparison:
    """Full result of :func:`compare_agents`."""

    agent_a: str | None
    agent_b: str | None
    paired: PairedDelta
    per_task: list[TaskComparison] = field(default_factory=list)
    alpha: float = 0.05

    @property
    def n_tasks(self) -> int:
        return len(self.per_task)

    @property
    def n_sig_holm(self) -> int:
        return sum(t.reject_holm for t in self.per_task)

    @property
    def n_sig_bh(self) -> int:
        return sum(t.reject_bh for t in self.per_task)


def per_task_tests(
    a: RunSet | dict[str, np.ndarray],
    b: RunSet | dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
) -> tuple[list[str], list[float], dict[str, tuple[float, float, float]]]:
    """Welch two-sample test per shared task. Returns (tasks, pvalues, stats).

    ``stats[task] = (mean_a, mean_b, delta)``. Tasks where either side has a
    single run, or both sides are constant, get ``p = 1.0`` (no evidence).
    """
    sa, sb = _scores(a), _scores(b)
    tasks = _shared_tasks(sa, sb)
    pvals: list[float] = []
    info: dict[str, tuple[float, float, float]] = {}
    for t in tasks:
        xa, xb = sa[t], sb[t]
        ma, mb = float(xa.mean()), float(xb.mean())
        info[t] = (ma, mb, ma - mb)
        if xa.size < 2 or xb.size < 2:
            pvals.append(1.0)
            continue
        if np.std(xa) == 0 and np.std(xb) == 0:
            pvals.append(1.0 if ma == mb else 0.0)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, p = stats.ttest_ind(xa, xb, equal_var=False)
        pvals.append(float(p) if np.isfinite(p) else 1.0)
    return tasks, pvals, info


def compare_agents(
    a: RunSet | dict[str, np.ndarray],
    b: RunSet | dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = None,
) -> AgentComparison:
    """End-to-end paired comparison of two agent configurations.

    Combines the aggregate paired delta with per-task Welch tests corrected by
    both Holm (FWER) and Benjamini-Hochberg (FDR).
    """
    paired = paired_delta(a, b, alpha=alpha, n_boot=n_boot, seed=seed)
    tasks, pvals, info = per_task_tests(a, b, alpha=alpha)
    p_holm, rej_holm = holm(pvals, alpha=alpha)
    p_bh, rej_bh = benjamini_hochberg(pvals, alpha=alpha)

    per_task = [
        TaskComparison(
            task_id=t,
            mean_a=info[t][0],
            mean_b=info[t][1],
            delta=info[t][2],
            pvalue=pvals[i],
            p_holm=p_holm[i],
            p_bh=p_bh[i],
            reject_holm=rej_holm[i],
            reject_bh=rej_bh[i],
        )
        for i, t in enumerate(tasks)
    ]

    agent_a = a.agent if isinstance(a, RunSet) else None
    agent_b = b.agent if isinstance(b, RunSet) else None
    return AgentComparison(
        agent_a=agent_a,
        agent_b=agent_b,
        paired=paired,
        per_task=per_task,
        alpha=alpha,
    )

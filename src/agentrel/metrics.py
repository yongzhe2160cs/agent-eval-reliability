"""Reliability metrics for stochastic agent evals.

Everything here treats the *multiple runs per task* explicitly. The recurring
theme: a point estimate from a single run (or a single aggregate number) hides
how much of the score is the agent vs. luck, and how wide the uncertainty is.

Implemented:

* :func:`variance_components` / :func:`icc` — one-way random-effects variance
  decomposition (between-task vs within-task) and ICC(1).
* :func:`pass_at_k` — unbiased pass@k (HumanEval/Codex estimator) with a
  cluster-bootstrap CI on the task-averaged value.
* :func:`pass_hat_k` — pass^k (probability *all* k attempts succeed), the
  reliability counterpart of pass@k.
* :func:`mean_score_ci` — cluster-bootstrap CI for the task-averaged mean
  score (partial credit is not binomial, so we bootstrap).
* :func:`min_runs_for_ci_width` — power helper: runs/task for a target CI
  half-width.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import stats

from .datamodel import RunSet

__all__ = [
    "VarianceComponents",
    "variance_components",
    "icc",
    "Estimate",
    "pass_at_k",
    "pass_hat_k",
    "mean_score_ci",
    "min_runs_for_ci_width",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _as_groups(runs: RunSet | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(runs, RunSet):
        return runs.scores_by_task()
    return {k: np.asarray(v, dtype=float) for k, v in runs.items()}


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a (1-alpha) confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    alpha: float = 0.05
    method: str = ""

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def half_width(self) -> float:
        return self.ci_width / 2.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        pct = int(round((1 - self.alpha) * 100))
        return (
            f"{self.value:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}] ({pct}% CI, {self.method})"
        )


# --------------------------------------------------------------------------
# variance decomposition / ICC
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VarianceComponents:
    """One-way random-effects variance decomposition.

    Model: ``score_ij = mu + tau_i + eps_ij`` where ``tau_i`` is the
    (random) task effect with variance ``between`` and ``eps_ij`` is the
    run-to-run noise with variance ``within``.
    """

    between: float
    within: float
    grand_mean: float
    n_tasks: int
    n_runs: int
    ms_between: float
    ms_within: float
    n0: float

    @property
    def total(self) -> float:
        return self.between + self.within

    @property
    def icc(self) -> float:
        """Fraction of variance attributable to the task (ICC(1))."""
        if self.total <= 0:
            return 0.0
        return self.between / self.total


def variance_components(runs: RunSet | dict[str, np.ndarray]) -> VarianceComponents:
    """One-way random-effects ANOVA decomposition over tasks.

    Handles the unbalanced case (different #runs per task) via the standard
    ``n0`` correction. Tasks with a single run contribute to the between-task
    sum of squares but not the within-task estimate.
    """
    groups = _as_groups(runs)
    # need at least one task with >1 run for a within estimate
    arrays = [g for g in groups.values() if g.size >= 1]
    if len(arrays) < 2:
        raise ValueError("need at least 2 tasks for variance decomposition")

    k = len(arrays)
    n_i = np.array([g.size for g in arrays], dtype=float)
    n_total = float(n_i.sum())
    group_means = np.array([g.mean() for g in arrays])
    grand_mean = float(np.concatenate(arrays).mean())

    ss_between = float(np.sum(n_i * (group_means - grand_mean) ** 2))
    ss_within = float(np.sum([np.sum((g - g.mean()) ** 2) for g in arrays]))

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # n0: average group size correction for unbalanced design
    n0 = (n_total - float(np.sum(n_i**2)) / n_total) / df_between if df_between > 0 else 0.0

    # n0 == 0 means every task has a single run -> components inseparable.
    between = max((ms_between - ms_within) / n0, 0.0) if n0 > 0 else 0.0
    within = ms_within

    return VarianceComponents(
        between=between,
        within=within,
        grand_mean=grand_mean,
        n_tasks=k,
        n_runs=int(n_total),
        ms_between=ms_between,
        ms_within=ms_within,
        n0=n0,
    )


def icc(runs: RunSet | dict[str, np.ndarray]) -> float:
    """Intraclass correlation ICC(1): share of variance that is the agent, not luck.

    ``1.0`` => runs of a task are perfectly consistent (all signal).
    ``0.0`` => a task's runs are as variable as the whole pool (all luck).
    """
    return variance_components(runs).icc


# --------------------------------------------------------------------------
# pass@k / pass^k
# --------------------------------------------------------------------------
def _pass_at_k_count(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one task: P(>=1 of k sampled runs succeeds).

    From Chen et al. 2021 (HumanEval): ``1 - C(n-c, k) / C(n, k)``.
    """
    if k > n:
        raise ValueError(f"k={k} exceeds available runs n={n} for a task")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _pass_hat_k_count(n: int, c: int, k: int) -> float:
    """Unbiased pass^k for one task: P(all k sampled runs succeed) = C(c,k)/C(n,k)."""
    if k > n:
        raise ValueError(f"k={k} exceeds available runs n={n} for a task")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def _aggregate_metric(
    succ: dict[str, np.ndarray],
    k: int,
    per_task_fn,
) -> tuple[float, dict[str, float]]:
    per_task: dict[str, float] = {}
    for tid, s in succ.items():
        n = int(s.size)
        c = int(np.count_nonzero(s))
        per_task[tid] = per_task_fn(n, c, k)
    agg = float(np.mean(list(per_task.values()))) if per_task else float("nan")
    return agg, per_task


def _cluster_bootstrap_metric(
    succ: dict[str, np.ndarray],
    k: int,
    per_task_fn,
    *,
    n_boot: int,
    alpha: float,
    seed: int | None,
) -> tuple[float, float]:
    rng = _rng(seed)
    tids = list(succ.keys())
    arrays = [succ[t] for t in tids]
    m = len(tids)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        task_idx = rng.integers(0, m, size=m)
        vals = []
        for ti in task_idx:
            s = arrays[ti]
            n = s.size
            # resample runs within the task (hierarchical bootstrap)
            rs = s[rng.integers(0, n, size=n)]
            c = int(np.count_nonzero(rs))
            vals.append(per_task_fn(n, c, k))
        boot[b] = np.mean(vals)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def pass_at_k(
    runs: RunSet | dict[str, np.ndarray],
    k: int,
    *,
    threshold: float = 1.0,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = None,
) -> Estimate:
    """Task-averaged unbiased pass@k with a hierarchical bootstrap CI.

    ``pass@k`` = probability that at least one of ``k`` sampled runs solves the
    task. Per-task values use the unbiased combinatorial estimator; the CI on
    the task average comes from resampling tasks *and* runs.
    """
    succ = _successes(runs, threshold)
    _check_k(succ, k)
    agg, _ = _aggregate_metric(succ, k, _pass_at_k_count)
    lo, hi = _cluster_bootstrap_metric(
        succ, k, _pass_at_k_count, n_boot=n_boot, alpha=alpha, seed=seed
    )
    return Estimate(agg, lo, hi, alpha=alpha, method=f"pass@{k} cluster-bootstrap")


def pass_hat_k(
    runs: RunSet | dict[str, np.ndarray],
    k: int,
    *,
    threshold: float = 1.0,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = None,
) -> Estimate:
    """Task-averaged unbiased pass^k (reliability) with a hierarchical bootstrap CI.

    ``pass^k`` = probability that *all* ``k`` sampled runs solve the task. This
    rewards consistency, where ``pass@k`` rewards best-of-k.
    """
    succ = _successes(runs, threshold)
    _check_k(succ, k)
    agg, _ = _aggregate_metric(succ, k, _pass_hat_k_count)
    lo, hi = _cluster_bootstrap_metric(
        succ, k, _pass_hat_k_count, n_boot=n_boot, alpha=alpha, seed=seed
    )
    return Estimate(agg, lo, hi, alpha=alpha, method=f"pass^{k} cluster-bootstrap")


def pass_at_k_table(
    runs: RunSet | dict[str, np.ndarray],
    k: int,
    *,
    threshold: float = 1.0,
) -> dict[str, float]:
    """Per-task unbiased pass@k point estimates."""
    succ = _successes(runs, threshold)
    _check_k(succ, k)
    _, per_task = _aggregate_metric(succ, k, _pass_at_k_count)
    return per_task


def _successes(runs: RunSet | dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    if isinstance(runs, RunSet):
        return runs.successes_by_task(threshold)
    return {k: (np.asarray(v, dtype=float) >= threshold) for k, v in runs.items()}


def _check_k(succ: dict[str, np.ndarray], k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    min_n = min((s.size for s in succ.values()), default=0)
    if k > min_n:
        raise ValueError(
            f"k={k} exceeds the minimum runs-per-task ({min_n}); "
            "every task needs at least k runs for an unbiased estimate"
        )


# --------------------------------------------------------------------------
# aggregate mean-score CI (partial credit -> bootstrap, not binomial)
# --------------------------------------------------------------------------
def mean_score_ci(
    runs: RunSet | dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = None,
) -> Estimate:
    """Cluster-bootstrap CI for the task-averaged mean score.

    The aggregate score is ``mean over tasks of (mean over runs)``; the CI
    resamples tasks and then runs within task, so it reflects both how few
    tasks and how few runs you have.
    """
    groups = _as_groups(runs)
    tids = list(groups.keys())
    arrays = [groups[t] for t in tids]
    if len(arrays) < 1:
        raise ValueError("need at least 1 task")
    point = float(np.mean([a.mean() for a in arrays]))

    rng = _rng(seed)
    m = len(arrays)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        task_idx = rng.integers(0, m, size=m)
        means = []
        for ti in task_idx:
            a = arrays[ti]
            rs = a[rng.integers(0, a.size, size=a.size)]
            means.append(rs.mean())
        boot[b] = np.mean(means)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return Estimate(point, lo, hi, alpha=alpha, method="mean-score cluster-bootstrap")


# --------------------------------------------------------------------------
# power helper
# --------------------------------------------------------------------------
def min_runs_for_ci_width(
    target_half_width: float,
    *,
    sd: float | None = None,
    p: float | None = None,
    alpha: float = 0.05,
) -> int:
    """Minimum runs per task to reach a target CI half-width.

    Provide ``sd`` (within-task standard deviation, partial credit) **or**
    ``p`` (a representative pass rate, binomial). Uses the normal approximation
    ``half_width = z * se`` with ``se = sd / sqrt(n)`` or ``sqrt(p(1-p)/n)``.
    """
    if target_half_width <= 0:
        raise ValueError("target_half_width must be > 0")
    z = float(stats.norm.ppf(1 - alpha / 2))
    if (sd is None) == (p is None):
        raise ValueError("provide exactly one of `sd` or `p`")
    if sd is not None:
        if sd < 0:
            raise ValueError("sd must be >= 0")
        var = sd**2
    else:
        if not (0.0 <= p <= 1.0):  # type: ignore[operator]
            raise ValueError("p must be in [0, 1]")
        var = p * (1 - p)  # type: ignore[operator]
    n = (z**2 * var) / (target_half_width**2)
    return max(1, int(np.ceil(n)))

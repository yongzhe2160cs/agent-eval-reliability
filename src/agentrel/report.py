"""High-level one-call API: :func:`reliability_report`.

Bundles the variance decomposition, aggregate CI, pass@k / pass^k, flakiness,
and provenance into a single dataclass with a readable ``summary()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .datamodel import RunSet
from .metrics import (
    Estimate,
    VarianceComponents,
    mean_score_ci,
    pass_at_k,
    pass_hat_k,
    variance_components,
)
from .repro import FlakinessReport, Provenance, flakiness_report, stamp

__all__ = ["ReliabilityReport", "reliability_report"]


@dataclass
class ReliabilityReport:
    """Everything you want to know before trusting an agent-eval number."""

    agent: str | None
    n_tasks: int
    n_runs: int
    mean_score: Estimate
    variance: VarianceComponents
    pass_at_k: dict[int, Estimate] = field(default_factory=dict)
    pass_hat_k: dict[int, Estimate] = field(default_factory=dict)
    flakiness: FlakinessReport | None = None
    provenance: Provenance | None = None

    @property
    def icc(self) -> float:
        return self.variance.icc

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines: list[str] = []
        lines.append(f"Reliability report — agent={self.agent!r}")
        lines.append(
            f"  design: {self.n_tasks} tasks x ~{self.n_runs / max(self.n_tasks, 1):.1f} "
            f"runs ({self.n_runs} runs)"
        )
        lines.append(f"  mean score:   {self.mean_score!r}")
        lines.append(
            f"  variance:     between-task={self.variance.between:.4f}  "
            f"within-task(luck)={self.variance.within:.4f}"
        )
        lines.append(f"  ICC(1):       {self.icc:.3f}  ({_icc_gloss(self.icc)})")
        for k in sorted(self.pass_at_k):
            lines.append(f"  pass@{k}:       {self.pass_at_k[k]!r}")
        for k in sorted(self.pass_hat_k):
            lines.append(f"  pass^{k}:       {self.pass_hat_k[k]!r}")
        if self.flakiness is not None:
            lines.append(
                f"  flakiness:    {self.flakiness.n_flaky}/{len(self.flakiness.tasks)} "
                f"tasks have a pass-rate CI wider than {self.flakiness.max_ci_width} "
                f"(too few runs to trust)"
            )
        if self.provenance is not None:
            p = self.provenance
            lines.append(
                f"  provenance:   input={p.input_hash} numpy={p.numpy} "
                f"agentrel={p.agentrel} seed={p.seed}"
            )
        return "\n".join(lines)


def _icc_gloss(icc: float) -> str:
    if icc >= 0.75:
        return "mostly agent skill; runs are consistent"
    if icc >= 0.5:
        return "moderate; meaningful run-to-run luck"
    if icc >= 0.25:
        return "luck-dominated; single runs are noisy"
    return "noise-dominated; single-run scores are unreliable"


def reliability_report(
    runs: RunSet,
    *,
    ks: tuple[int, ...] = (1, 2, 5),
    threshold: float = 1.0,
    alpha: float = 0.05,
    n_boot: int = 2000,
    max_ci_width: float = 0.4,
    seed: int | None = 0,
    with_provenance: bool = True,
) -> ReliabilityReport:
    """Compute a full reliability report for one agent's runs.

    ``ks`` controls which pass@k / pass^k values are computed; any ``k`` larger
    than the minimum runs-per-task is skipped (the unbiased estimator needs at
    least ``k`` runs on every task).
    """
    min_runs = min(runs.runs_per_task().values(), default=0)
    valid_ks = tuple(k for k in ks if k <= min_runs)

    mean = mean_score_ci(runs, alpha=alpha, n_boot=n_boot, seed=seed)
    var = variance_components(runs)
    pak = {
        k: pass_at_k(runs, k, threshold=threshold, alpha=alpha, n_boot=n_boot, seed=seed)
        for k in valid_ks
    }
    phk = {
        k: pass_hat_k(runs, k, threshold=threshold, alpha=alpha, n_boot=n_boot, seed=seed)
        for k in valid_ks
    }
    flaky = flakiness_report(runs, threshold=threshold, max_ci_width=max_ci_width, alpha=alpha)
    prov = stamp(runs, seed=seed) if with_provenance else None

    return ReliabilityReport(
        agent=runs.agent,
        n_tasks=runs.n_tasks,
        n_runs=runs.n_runs,
        mean_score=mean,
        variance=var,
        pass_at_k=pak,
        pass_hat_k=phk,
        flakiness=flaky,
        provenance=prov,
    )

"""Reproducibility harness: provenance stamping, determinism, flakiness.

These are the "is this number trustworthy?" checks that agent-eval reports
usually omit:

* :func:`stamp` captures library versions, a content hash of the runs, and the
  seed used, so a reported metric is tied to an exact input + environment.
* :func:`determinism_check` runs a callable twice and confirms identical output
  (catches hidden nondeterminism in your own analysis pipeline).
* :func:`flakiness_report` flags tasks whose pass-rate CI is too wide to draw
  any conclusion from — the tasks you should run more or distrust.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy

from .datamodel import RunSet

__all__ = [
    "Provenance",
    "stamp",
    "determinism_check",
    "TaskFlakiness",
    "FlakinessReport",
    "flakiness_report",
    "wilson_interval",
]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Provenance:
    """Environment + input fingerprint for a reported result."""

    python: str
    platform: str
    numpy: str
    scipy: str
    agentrel: str
    input_hash: str
    n_tasks: int
    n_runs: int
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "platform": self.platform,
            "numpy": self.numpy,
            "scipy": self.scipy,
            "agentrel": self.agentrel,
            "input_hash": self.input_hash,
            "n_tasks": self.n_tasks,
            "n_runs": self.n_runs,
            "seed": self.seed,
            **({"extra": self.extra} if self.extra else {}),
        }


def _hash_runs(runs: RunSet) -> str:
    """Order-independent content hash of the runs (task_id, score pairs)."""
    items = sorted((r.task_id, round(float(r.score), 12), r.run_id or "") for r in runs.runs)
    h = hashlib.sha256()
    for tid, score, rid in items:
        h.update(f"{tid}|{score:.12f}|{rid}\n".encode())
    return h.hexdigest()[:16]


def stamp(runs: RunSet, *, seed: int | None = None, **extra: Any) -> Provenance:
    """Capture a :class:`Provenance` record for ``runs``."""
    from . import __version__

    return Provenance(
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        agentrel=__version__,
        input_hash=_hash_runs(runs),
        n_tasks=runs.n_tasks,
        n_runs=runs.n_runs,
        seed=seed,
        extra=dict(extra),
    )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeterminismResult:
    deterministic: bool
    detail: str

    def __bool__(self) -> bool:
        return self.deterministic


def determinism_check(
    fn: Callable[[], Any],
    *,
    runs: int = 2,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> DeterminismResult:
    """Run ``fn`` ``runs`` times and check the outputs are equal.

    For numeric outputs, comparison uses ``np.allclose`` with the given
    tolerances; set both tolerances to 0 (default) to require bit-stable
    equality. Use this to confirm a seeded analysis pipeline is reproducible.
    """
    outputs = [fn() for _ in range(runs)]
    first = outputs[0]
    for i, out in enumerate(outputs[1:], start=1):
        ok, why = _approx_equal(first, out, rtol=rtol, atol=atol)
        if not ok:
            return DeterminismResult(False, f"run 0 vs run {i} differ: {why}")
    return DeterminismResult(True, f"{runs} runs identical")


def _approx_equal(a: Any, b: Any, *, rtol: float, atol: float) -> tuple[bool, str]:
    try:
        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)
        if arr_a.shape != arr_b.shape:
            return False, f"shape {arr_a.shape} vs {arr_b.shape}"
        if np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True):
            return True, ""
        return False, "numeric values differ"
    except (TypeError, ValueError):
        return (a == b, "" if a == b else f"{a!r} != {b!r}")


# --------------------------------------------------------------------------
# flakiness
# --------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    from scipy import stats

    z = float(stats.norm.ppf(1 - alpha / 2))
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class TaskFlakiness:
    task_id: str
    n_runs: int
    pass_rate: float
    ci_low: float
    ci_high: float
    flaky: bool

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


@dataclass(frozen=True)
class FlakinessReport:
    tasks: list[TaskFlakiness]
    max_ci_width: float
    threshold: float
    alpha: float

    @property
    def flaky_tasks(self) -> list[TaskFlakiness]:
        return [t for t in self.tasks if t.flaky]

    @property
    def n_flaky(self) -> int:
        return len(self.flaky_tasks)

    @property
    def frac_flaky(self) -> float:
        return self.n_flaky / len(self.tasks) if self.tasks else 0.0


def flakiness_report(
    runs: RunSet,
    *,
    threshold: float = 1.0,
    max_ci_width: float = 0.4,
    alpha: float = 0.05,
) -> FlakinessReport:
    """Flag tasks whose pass-rate CI is too wide to trust.

    For each task, computes the pass rate at ``threshold`` and its Wilson CI; a
    task is *flaky* when the CI width exceeds ``max_ci_width`` — i.e. you have
    too few runs to pin down its pass rate. These are the tasks dominating your
    run-to-run variance and the first place to spend more runs.
    """
    succ = runs.successes_by_task(threshold)
    out: list[TaskFlakiness] = []
    for tid, s in succ.items():
        n = int(s.size)
        c = int(np.count_nonzero(s))
        lo, hi = wilson_interval(c, n, alpha=alpha)
        width = hi - lo
        out.append(
            TaskFlakiness(
                task_id=tid,
                n_runs=n,
                pass_rate=(c / n if n else float("nan")),
                ci_low=lo,
                ci_high=hi,
                flaky=bool(width > max_ci_width),
            )
        )
    out.sort(key=lambda t: t.ci_width, reverse=True)
    return FlakinessReport(tasks=out, max_ci_width=max_ci_width, threshold=threshold, alpha=alpha)

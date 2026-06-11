"""Core data model for agent-eval reliability analysis.

The central object is a :class:`RunSet`: the results of running one agent
configuration over a set of tasks, with (crucially) *multiple runs per task*
because agents are stochastic. Each individual run is a :class:`TaskRun`.

Scores live in ``[0, 1]`` to support partial-credit / trajectory scoring, not
just binary pass/fail. A run "succeeds" when its score meets a configurable
threshold (default ``1.0`` == fully solved).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["TaskRun", "RunSet"]


@dataclass(frozen=True)
class TaskRun:
    """A single (stochastic) run of an agent on one task.

    Parameters
    ----------
    task_id:
        Identifier of the task / sample.
    score:
        Outcome in ``[0, 1]``. Use ``1.0``/``0.0`` for binary success, or a
        partial-credit / trajectory score in between.
    run_id:
        Optional identifier for the run (e.g. epoch index). Used only for
        bookkeeping; reliability math treats runs of a task as exchangeable.
    seed:
        Optional sampling seed recorded for reproducibility.
    agent:
        Optional label for the agent / model configuration.
    metadata:
        Free-form extra fields (model name, tool calls, tokens, ...).
    """

    task_id: str
    score: float
    run_id: str | None = None
    seed: int | None = None
    agent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score!r}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score!r}")

    def is_success(self, threshold: float = 1.0) -> bool:
        """Whether this run counts as a success at ``threshold``."""
        return self.score >= threshold


@dataclass
class RunSet:
    """A collection of :class:`TaskRun` for one agent configuration.

    This is an unbalanced design by default: tasks may have different numbers
    of runs. Reliability estimators below handle the unbalanced case.
    """

    runs: list[TaskRun] = field(default_factory=list)
    agent: str | None = None

    def __post_init__(self) -> None:
        if self.agent is not None:
            self.runs = [
                r if r.agent is not None else _with_agent(r, self.agent) for r in self.runs
            ]

    # -- construction -----------------------------------------------------
    def add(
        self,
        task_id: str,
        score: float,
        *,
        run_id: str | None = None,
        seed: int | None = None,
        agent: str | None = None,
        **metadata: Any,
    ) -> RunSet:
        """Append a run and return ``self`` (chainable)."""
        self.runs.append(
            TaskRun(
                task_id=str(task_id),
                score=float(score),
                run_id=run_id,
                seed=seed,
                agent=agent or self.agent,
                metadata=metadata,
            )
        )
        return self

    # -- views ------------------------------------------------------------
    @property
    def task_ids(self) -> list[str]:
        """Distinct task ids in first-seen order."""
        seen: dict[str, None] = {}
        for r in self.runs:
            seen.setdefault(r.task_id, None)
        return list(seen)

    @property
    def n_tasks(self) -> int:
        return len(self.task_ids)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def scores_by_task(self) -> dict[str, np.ndarray]:
        """Map each task id to its array of run scores."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in self.runs:
            buckets[r.task_id].append(r.score)
        return {tid: np.asarray(buckets[tid], dtype=float) for tid in self.task_ids}

    def successes_by_task(self, threshold: float = 1.0) -> dict[str, np.ndarray]:
        """Map each task id to a boolean array of per-run success."""
        return {tid: (scores >= threshold) for tid, scores in self.scores_by_task().items()}

    def runs_per_task(self) -> dict[str, int]:
        return {tid: len(s) for tid, s in self.scores_by_task().items()}

    def mean_score_by_task(self) -> dict[str, float]:
        return {tid: float(s.mean()) for tid, s in self.scores_by_task().items()}

    def filter_min_runs(self, min_runs: int) -> RunSet:
        """Return a new RunSet keeping only tasks with at least ``min_runs`` runs."""
        keep = {tid for tid, n in self.runs_per_task().items() if n >= min_runs}
        return RunSet([r for r in self.runs if r.task_id in keep], agent=self.agent)

    def __len__(self) -> int:
        return len(self.runs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RunSet(agent={self.agent!r}, n_tasks={self.n_tasks}, n_runs={self.n_runs})"


def _with_agent(run: TaskRun, agent: str) -> TaskRun:
    return TaskRun(
        task_id=run.task_id,
        score=run.score,
        run_id=run.run_id,
        seed=run.seed,
        agent=agent,
        metadata=run.metadata,
    )

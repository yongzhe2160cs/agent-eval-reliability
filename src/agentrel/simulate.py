"""Synthetic agent-run generator for demos and tests.

We have no model API here (keyless by design), so a generator stands in for
real runs. It models the structure that makes agent evals hard:

* each task has a latent *solve probability* (the agent's true skill on it),
* runs are Bernoulli draws around that latent skill (run-to-run luck),
* optional partial credit adds within-task score spread even on "fails".

Tuning ``skill_spread`` (between-task) vs ``luck`` (within-task) lets you dial
the ICC up and down, which is handy for testing the variance decomposition.
"""

from __future__ import annotations

import numpy as np

from .datamodel import RunSet

__all__ = ["simulate_agent_runs"]


def simulate_agent_runs(
    n_tasks: int = 30,
    runs_per_task: int = 8,
    *,
    base_skill: float = 0.55,
    skill_spread: float = 0.25,
    luck: float = 0.0,
    partial_credit: bool = True,
    agent: str = "sim-agent",
    seed: int | None = 0,
) -> RunSet:
    """Generate a synthetic :class:`RunSet`.

    Parameters
    ----------
    n_tasks, runs_per_task:
        Size of the design.
    base_skill:
        Mean latent solve probability across tasks.
    skill_spread:
        Std of latent per-task solve probability (between-task variance). Larger
        => higher ICC (more of the score is the task, less is luck).
    luck:
        Extra Bernoulli jitter added to each task's solve probability per run
        (within-task noise). Larger => lower ICC.
    partial_credit:
        If True, failed runs still get a partial trajectory score in ``[0, 1)``
        rather than exactly 0.
    agent:
        Label stamped on every run.
    seed:
        RNG seed (reproducible).
    """
    rng = np.random.default_rng(seed)
    skills = np.clip(rng.normal(base_skill, skill_spread, size=n_tasks), 0.02, 0.98)

    rs = RunSet(agent=agent)
    for t in range(n_tasks):
        tid = f"task_{t:03d}"
        for r in range(runs_per_task):
            p = skills[t]
            if luck > 0:
                p = float(np.clip(p + rng.normal(0, luck), 0.0, 1.0))
            solved = rng.random() < p
            if solved:
                score = 1.0
            elif partial_credit:
                # partial trajectory progress, biased low
                score = float(np.clip(rng.beta(2.0, 5.0), 0.0, 0.99))
            else:
                score = 0.0
            rs.add(tid, score, run_id=str(r), seed=int(rng.integers(0, 2**31)), agent=agent)
    return rs

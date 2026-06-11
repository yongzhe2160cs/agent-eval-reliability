"""agentrel — reliability & reproducibility statistics for stochastic agent evals.

Agent / tool-use evals are multi-turn, partial-credit, and *stochastic*: the
same agent on the same task scores differently run to run. A single-run number
hides how much of the score is skill vs luck and how wide the uncertainty is.
``agentrel`` makes those explicit.

Quick start
-----------
>>> from agentrel import simulate_agent_runs, reliability_report, compare_agents
>>> runs = simulate_agent_runs(n_tasks=30, runs_per_task=8, seed=0)
>>> print(reliability_report(runs).summary())          # doctest: +SKIP
"""

from __future__ import annotations

__version__ = "0.1.0"

from .adapters import from_csv, from_inspect_log, from_json, from_records
from .compare import (
    AgentComparison,
    PairedDelta,
    TaskComparison,
    benjamini_hochberg,
    compare_agents,
    holm,
    paired_delta,
    per_task_tests,
)
from .datamodel import RunSet, TaskRun
from .metrics import (
    Estimate,
    VarianceComponents,
    icc,
    mean_score_ci,
    min_runs_for_ci_width,
    pass_at_k,
    pass_hat_k,
    variance_components,
)
from .report import ReliabilityReport, reliability_report
from .repro import (
    FlakinessReport,
    Provenance,
    TaskFlakiness,
    determinism_check,
    flakiness_report,
    stamp,
    wilson_interval,
)
from .simulate import simulate_agent_runs

__all__ = [
    "__version__",
    # data model
    "TaskRun",
    "RunSet",
    # adapters
    "from_records",
    "from_json",
    "from_csv",
    "from_inspect_log",
    # metrics
    "Estimate",
    "VarianceComponents",
    "variance_components",
    "icc",
    "pass_at_k",
    "pass_hat_k",
    "mean_score_ci",
    "min_runs_for_ci_width",
    # compare
    "PairedDelta",
    "TaskComparison",
    "AgentComparison",
    "paired_delta",
    "per_task_tests",
    "holm",
    "benjamini_hochberg",
    "compare_agents",
    # repro
    "Provenance",
    "stamp",
    "determinism_check",
    "TaskFlakiness",
    "FlakinessReport",
    "flakiness_report",
    "wilson_interval",
    # report
    "ReliabilityReport",
    "reliability_report",
    # simulate
    "simulate_agent_runs",
]

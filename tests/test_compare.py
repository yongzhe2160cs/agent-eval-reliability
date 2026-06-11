"""Tests for paired comparison and multiplicity correction."""

import numpy as np
import pytest
from scipy import stats

from agentrel import (
    RunSet,
    benjamini_hochberg,
    compare_agents,
    holm,
    paired_delta,
    per_task_tests,
    simulate_agent_runs,
)


# ---- multiplicity corrections vs hand-computed references ----------------
def test_holm_reference():
    adj, reject = holm([0.01, 0.04, 0.03], alpha=0.05)
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert reject == [True, False, False]


def test_bh_reference():
    adj, reject = benjamini_hochberg([0.01, 0.04, 0.03], alpha=0.05)
    assert adj == pytest.approx([0.03, 0.04, 0.04])
    assert reject == [True, True, True]


def test_holm_at_least_as_conservative_as_bh():
    rng = np.random.default_rng(0)
    p = list(rng.uniform(0, 0.2, size=20))
    holm_adj, _ = holm(p)
    bh_adj, _ = benjamini_hochberg(p)
    assert all(h >= b - 1e-12 for h, b in zip(holm_adj, bh_adj, strict=True))


def test_corrections_handle_empty():
    assert holm([]) == ([], [])
    assert benjamini_hochberg([]) == ([], [])


def test_corrections_adjusted_pvalues_capped_at_one():
    adj, _ = holm([0.9, 0.95])
    assert all(a <= 1.0 for a in adj)


# ---- paired delta vs scipy ----------------------------------------------
def test_paired_delta_matches_scipy_ttest_rel():
    a = RunSet(agent="A")
    b = RunSet(agent="B")
    rng = np.random.default_rng(1)
    for t in range(15):
        for _ in range(6):
            a.add(f"t{t}", float(rng.random() < 0.7))
            b.add(f"t{t}", float(rng.random() < 0.5))
    pd = paired_delta(a, b, n_boot=500, seed=0)
    means_a = np.array([a.scores_by_task()[t].mean() for t in a.task_ids])
    means_b = np.array([b.scores_by_task()[t].mean() for t in b.task_ids])
    t_stat, t_p = stats.ttest_rel(means_a, means_b)
    assert pd.t_statistic == pytest.approx(t_stat)
    assert pd.t_pvalue == pytest.approx(t_p)
    assert pd.delta.value == pytest.approx(float((means_a - means_b).mean()))


def test_paired_delta_ci_contains_point():
    a = simulate_agent_runs(n_tasks=30, runs_per_task=8, base_skill=0.6, seed=2)
    b = simulate_agent_runs(n_tasks=30, runs_per_task=8, base_skill=0.6, seed=3)
    pd = paired_delta(a, b, n_boot=1000, seed=0)
    assert pd.delta.ci_low <= pd.delta.value <= pd.delta.ci_high


def test_identical_agents_no_significant_difference():
    runs = simulate_agent_runs(n_tasks=40, runs_per_task=8, seed=4)
    cmp = compare_agents(runs, runs, n_boot=500, seed=0)
    assert cmp.paired.delta.value == pytest.approx(0.0, abs=1e-9)
    assert cmp.n_sig_holm == 0
    assert cmp.n_sig_bh == 0


def test_better_agent_detected_and_correction_reduces_rejections():
    strong = simulate_agent_runs(
        n_tasks=40, runs_per_task=12, base_skill=0.8, skill_spread=0.1, partial_credit=False, seed=5
    )
    weak = simulate_agent_runs(
        n_tasks=40, runs_per_task=12, base_skill=0.3, skill_spread=0.1, partial_credit=False, seed=6
    )
    cmp = compare_agents(strong, weak, n_boot=500, seed=0)
    assert cmp.paired.delta.value > 0
    assert cmp.paired.significant
    # corrected discoveries cannot exceed the raw count at the same alpha
    raw_sig = sum(t.pvalue < cmp.alpha for t in cmp.per_task)
    assert cmp.n_sig_holm <= raw_sig
    assert cmp.n_sig_bh <= raw_sig
    assert cmp.n_sig_holm <= cmp.n_sig_bh


def test_per_task_tests_constant_groups():
    a = RunSet().add("t", 1.0).add("t", 1.0)
    b = RunSet().add("t", 1.0).add("t", 1.0)
    tasks, pvals, info = per_task_tests(a, b)
    assert pvals == [1.0]
    assert info["t"] == (1.0, 1.0, 0.0)


def test_no_shared_tasks_raises():
    a = RunSet().add("t1", 1.0).add("t1", 0.0)
    b = RunSet().add("t2", 1.0).add("t2", 0.0)
    with pytest.raises(ValueError):
        paired_delta(a, b)

"""Tests for the high-level reliability_report API."""

from agentrel import reliability_report, simulate_agent_runs


def test_reliability_report_fields():
    runs = simulate_agent_runs(n_tasks=25, runs_per_task=8, seed=0)
    rep = reliability_report(runs, ks=(1, 2, 5))
    assert rep.n_tasks == 25
    assert rep.n_runs == 200
    assert 0.0 <= rep.icc <= 1.0
    assert rep.mean_score.ci_low <= rep.mean_score.value <= rep.mean_score.ci_high
    assert set(rep.pass_at_k) == {1, 2, 5}
    assert set(rep.pass_hat_k) == {1, 2, 5}
    assert rep.flakiness is not None
    assert rep.provenance is not None


def test_reliability_report_skips_k_above_min_runs():
    runs = simulate_agent_runs(n_tasks=10, runs_per_task=4, seed=0)
    rep = reliability_report(runs, ks=(1, 2, 5, 10))
    # only k <= 4 retained
    assert set(rep.pass_at_k) == {1, 2}


def test_reliability_report_summary_is_string():
    runs = simulate_agent_runs(n_tasks=10, runs_per_task=6, seed=0)
    rep = reliability_report(runs)
    s = rep.summary()
    assert isinstance(s, str)
    assert "ICC" in s
    assert "pass@" in s


def test_reliability_report_without_provenance():
    runs = simulate_agent_runs(n_tasks=10, runs_per_task=6, seed=0)
    rep = reliability_report(runs, with_provenance=False)
    assert rep.provenance is None

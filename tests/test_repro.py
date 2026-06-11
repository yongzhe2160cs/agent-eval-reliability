"""Tests for the reproducibility harness."""

import pytest

from agentrel import (
    RunSet,
    determinism_check,
    flakiness_report,
    simulate_agent_runs,
    stamp,
    wilson_interval,
)


def test_stamp_fields_and_counts():
    runs = simulate_agent_runs(n_tasks=10, runs_per_task=5, seed=0)
    prov = stamp(runs, seed=42)
    assert prov.n_tasks == 10
    assert prov.n_runs == 50
    assert prov.seed == 42
    assert prov.agentrel
    assert len(prov.input_hash) == 16
    assert "input_hash" in prov.to_dict()


def test_stamp_hash_order_independent_but_value_sensitive():
    a = RunSet().add("t1", 1.0).add("t2", 0.0)
    b = RunSet().add("t2", 0.0).add("t1", 1.0)  # reordered
    assert stamp(a).input_hash == stamp(b).input_hash
    c = RunSet().add("t1", 1.0).add("t2", 0.5)  # different score
    assert stamp(a).input_hash != stamp(c).input_hash


def test_determinism_check_true():
    res = determinism_check(lambda: [1.0, 2.0, 3.0])
    assert res
    assert res.deterministic


def test_determinism_check_seeded_pipeline():
    def analysis():
        return simulate_agent_runs(n_tasks=5, runs_per_task=4, seed=123).mean_score_by_task()

    # seeded -> identical dict each call
    res = determinism_check(lambda: list(analysis().values()))
    assert res.deterministic


def test_determinism_check_false():
    state = {"n": 0}

    def nondeterministic():
        state["n"] += 1
        return [float(state["n"])]

    res = determinism_check(nondeterministic)
    assert not res.deterministic
    assert "differ" in res.detail


def test_wilson_interval_known_value():
    # 8/10 successes, 95% Wilson interval ~ [0.490, 0.943]
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3)
    assert hi == pytest.approx(0.943, abs=1e-3)
    assert lo <= 0.8 <= hi


def test_wilson_interval_edges():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi < 1.0


def test_flakiness_report_flags_wide_ci():
    # 2 runs/task => very wide CI => flaky; 200 runs/task => tight => not flaky.
    rs = RunSet()
    rs.add("noisy", 1.0).add("noisy", 0.0)
    for i in range(200):
        rs.add("solid", 1.0 if i % 2 else 0.0)
    rep = flakiness_report(rs, max_ci_width=0.4)
    flagged = {t.task_id: t.flaky for t in rep.tasks}
    assert flagged["noisy"] is True
    assert flagged["solid"] is False
    assert rep.n_flaky == 1
    # report is sorted widest-CI first
    assert rep.tasks[0].task_id == "noisy"


def test_flakiness_frac():
    runs = simulate_agent_runs(n_tasks=20, runs_per_task=3, seed=0)
    rep = flakiness_report(runs, max_ci_width=0.3)
    assert 0.0 <= rep.frac_flaky <= 1.0

"""Tests for reliability metrics, verified against closed-form / brute-force references."""

from itertools import combinations
from math import comb

import numpy as np
import pytest

from agentrel import (
    icc,
    mean_score_ci,
    min_runs_for_ci_width,
    pass_at_k,
    pass_hat_k,
    simulate_agent_runs,
    variance_components,
)
from agentrel.metrics import _pass_at_k_count, _pass_hat_k_count


# ---- variance decomposition / ICC ---------------------------------------
def test_variance_components_hand_example():
    # Balanced: 2 tasks x 3 runs. Worked by hand -> ICC = 0.5.
    groups = {"A": np.array([1.0, 0.0, 1.0]), "B": np.array([0.0, 0.0, 0.0])}
    vc = variance_components(groups)
    assert vc.n0 == pytest.approx(3.0)
    assert vc.ms_within == pytest.approx(1 / 6, abs=1e-9)
    assert vc.between == pytest.approx(1 / 6, abs=1e-9)
    assert vc.within == pytest.approx(1 / 6, abs=1e-9)
    assert vc.icc == pytest.approx(0.5, abs=1e-9)


def test_icc_bounds():
    runs = simulate_agent_runs(n_tasks=40, runs_per_task=10, seed=1)
    val = icc(runs)
    assert 0.0 <= val <= 1.0


def test_icc_high_when_no_luck_high_spread():
    # Wide between-task skill, no within-task jitter -> high ICC.
    hi = icc(
        simulate_agent_runs(
            n_tasks=60, runs_per_task=12, skill_spread=0.35, luck=0.0, partial_credit=False, seed=2
        )
    )
    # Narrow between-task skill, lots of luck -> low ICC.
    lo = icc(
        simulate_agent_runs(
            n_tasks=60,
            runs_per_task=12,
            base_skill=0.5,
            skill_spread=0.02,
            luck=0.3,
            partial_credit=False,
            seed=2,
        )
    )
    # Binary Bernoulli sampling itself caps ICC below 1 even with zero `luck`,
    # so we check the ordering and a generous gap rather than an absolute floor.
    assert hi > lo + 0.15
    assert hi > 0.3
    assert lo < 0.2


def test_variance_components_needs_two_tasks():
    with pytest.raises(ValueError):
        variance_components({"only": np.array([1.0, 0.0])})


# ---- pass@k / pass^k vs brute force -------------------------------------
def _brute_pass_at_k(successes, k):
    n = len(successes)
    subsets = list(combinations(range(n), k))
    hits = sum(1 for s in subsets if any(successes[i] for i in s))
    return hits / len(subsets)


def _brute_pass_hat_k(successes, k):
    n = len(successes)
    subsets = list(combinations(range(n), k))
    hits = sum(1 for s in subsets if all(successes[i] for i in s))
    return hits / len(subsets)


@pytest.mark.parametrize("n,c", [(8, 3), (10, 0), (10, 10), (6, 1), (7, 4)])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_pass_at_k_count_matches_bruteforce(n, c, k):
    succ = [True] * c + [False] * (n - c)
    assert _pass_at_k_count(n, c, k) == pytest.approx(_brute_pass_at_k(succ, k))


@pytest.mark.parametrize("n,c", [(8, 3), (10, 0), (10, 10), (6, 1), (7, 4)])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_pass_hat_k_count_matches_bruteforce(n, c, k):
    succ = [True] * c + [False] * (n - c)
    assert _pass_hat_k_count(n, c, k) == pytest.approx(_brute_pass_hat_k(succ, k))


def test_pass_at_1_equals_mean_success_rate():
    runs = simulate_agent_runs(n_tasks=25, runs_per_task=8, seed=3)
    succ = runs.successes_by_task()
    expected = np.mean([s.mean() for s in succ.values()])
    est = pass_at_k(runs, 1, n_boot=200, seed=0)
    assert est.value == pytest.approx(expected)


def test_pass_at_k_ge_pass_hat_k():
    runs = simulate_agent_runs(n_tasks=25, runs_per_task=8, seed=4)
    pak = pass_at_k(runs, 3, n_boot=200, seed=0).value
    phk = pass_hat_k(runs, 3, n_boot=200, seed=0).value
    assert pak >= phk


def test_pass_at_k_ci_contains_point_and_is_ordered():
    runs = simulate_agent_runs(n_tasks=30, runs_per_task=8, seed=5)
    est = pass_at_k(runs, 2, n_boot=1000, seed=0)
    assert est.ci_low <= est.value <= est.ci_high
    assert 0.0 <= est.ci_low <= est.ci_high <= 1.0


def test_pass_at_k_rejects_k_above_min_runs():
    runs = RunSet_with_uneven()
    with pytest.raises(ValueError):
        pass_at_k(runs, 3)


def RunSet_with_uneven():
    from agentrel import RunSet

    rs = RunSet()
    rs.add("t1", 1.0).add("t1", 0.0).add("t1", 1.0)
    rs.add("t2", 1.0).add("t2", 0.0)  # only 2 runs -> k=3 invalid
    return rs


# ---- mean score CI ------------------------------------------------------
def test_mean_score_ci_contains_point():
    runs = simulate_agent_runs(n_tasks=30, runs_per_task=8, seed=6)
    est = mean_score_ci(runs, n_boot=1000, seed=0)
    point = np.mean([s.mean() for s in runs.scores_by_task().values()])
    assert est.value == pytest.approx(point)
    assert est.ci_low <= est.value <= est.ci_high


def test_mean_score_ci_narrows_with_more_runs():
    few = mean_score_ci(
        simulate_agent_runs(n_tasks=15, runs_per_task=4, seed=7), n_boot=1500, seed=0
    )
    many = mean_score_ci(
        simulate_agent_runs(n_tasks=80, runs_per_task=20, seed=7), n_boot=1500, seed=0
    )
    assert many.ci_width < few.ci_width


# ---- power helper -------------------------------------------------------
def test_min_runs_for_ci_width_binomial_formula():
    # n = z^2 p(1-p) / h^2 ; p=0.5, h=0.1, z~1.96 -> ~96.04 -> 97
    n = min_runs_for_ci_width(0.1, p=0.5)
    assert n == 97


def test_min_runs_for_ci_width_sd_formula():
    # n = (z*sd/h)^2 ; sd=0.5, h=0.1 -> (1.96*5)^2 = 96.04 -> 97
    n = min_runs_for_ci_width(0.1, sd=0.5)
    assert n == 97


def test_min_runs_smaller_width_needs_more_runs():
    assert min_runs_for_ci_width(0.05, p=0.5) > min_runs_for_ci_width(0.1, p=0.5)


def test_min_runs_requires_exactly_one_of_sd_p():
    with pytest.raises(ValueError):
        min_runs_for_ci_width(0.1)
    with pytest.raises(ValueError):
        min_runs_for_ci_width(0.1, sd=0.5, p=0.5)


def test_combinatorial_sanity():
    # guard the imports actually used by reference math
    assert comb(5, 2) == 10

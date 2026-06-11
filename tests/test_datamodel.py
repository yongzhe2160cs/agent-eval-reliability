import numpy as np
import pytest

from agentrel import RunSet, TaskRun


def test_taskrun_validates_score_range():
    TaskRun("t", 0.0)
    TaskRun("t", 1.0)
    with pytest.raises(ValueError):
        TaskRun("t", 1.5)
    with pytest.raises(ValueError):
        TaskRun("t", -0.1)
    with pytest.raises(ValueError):
        TaskRun("t", float("nan"))


def test_is_success_threshold():
    assert TaskRun("t", 1.0).is_success()
    assert not TaskRun("t", 0.5).is_success()
    assert TaskRun("t", 0.5).is_success(threshold=0.5)


def test_runset_views():
    rs = RunSet(agent="a")
    rs.add("t1", 1.0).add("t1", 0.0).add("t2", 0.5)
    assert rs.task_ids == ["t1", "t2"]
    assert rs.n_tasks == 2
    assert rs.n_runs == 3
    sbt = rs.scores_by_task()
    np.testing.assert_allclose(sbt["t1"], [1.0, 0.0])
    np.testing.assert_allclose(sbt["t2"], [0.5])
    assert rs.runs_per_task() == {"t1": 2, "t2": 1}
    assert rs.mean_score_by_task()["t1"] == 0.5


def test_runset_stamps_agent_on_add():
    rs = RunSet(agent="default")
    rs.add("t1", 1.0)
    assert rs.runs[0].agent == "default"
    rs.add("t2", 1.0, agent="override")
    assert rs.runs[1].agent == "override"


def test_successes_by_task_threshold():
    rs = RunSet().add("t", 1.0).add("t", 0.7).add("t", 0.2)
    full = rs.successes_by_task()
    assert full["t"].tolist() == [True, False, False]
    partial = rs.successes_by_task(threshold=0.5)
    assert partial["t"].tolist() == [True, True, False]


def test_filter_min_runs():
    rs = RunSet().add("t1", 1.0).add("t1", 0.0).add("t2", 1.0)
    filtered = rs.filter_min_runs(2)
    assert filtered.task_ids == ["t1"]
    assert filtered.n_runs == 2

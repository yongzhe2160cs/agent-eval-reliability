"""Tests for ingest adapters."""

import json

import pytest

from agentrel import from_csv, from_inspect_log, from_json, from_records


def test_from_records_autodetect():
    recs = [
        {"task_id": "t1", "score": 1.0, "epoch": 0},
        {"task_id": "t1", "score": 0.0, "epoch": 1},
        {"task": "t2", "value": 0.5},
    ]
    rs = from_records(recs, agent="x")
    assert rs.n_tasks == 2
    assert rs.scores_by_task()["t1"].tolist() == [1.0, 0.0]
    assert rs.runs[0].run_id == "0"


def test_from_records_letter_grades():
    recs = [
        {"task_id": "t", "score": "C"},
        {"task_id": "t", "score": "I"},
        {"task_id": "t", "score": "P"},
    ]
    rs = from_records(recs)
    assert rs.scores_by_task()["t"].tolist() == [1.0, 0.0, 0.5]


def test_from_records_bool_and_int():
    rs = from_records([{"task_id": "t", "score": True}, {"task_id": "t", "score": 0}])
    assert rs.scores_by_task()["t"].tolist() == [1.0, 0.0]


def test_from_records_metadata_passthrough():
    rs = from_records([{"task_id": "t", "score": 1.0, "tokens": 123, "tool": "bash"}])
    assert rs.runs[0].metadata == {"tokens": 123, "tool": "bash"}


def test_from_records_missing_fields_raise():
    with pytest.raises(ValueError):
        from_records([{"score": 1.0}])
    with pytest.raises(ValueError):
        from_records([{"task_id": "t"}])


def test_from_json_list_and_object(tmp_path):
    p1 = tmp_path / "list.json"
    p1.write_text(json.dumps([{"task_id": "t", "score": 1.0}]))
    assert from_json(p1).n_runs == 1

    p2 = tmp_path / "obj.json"
    p2.write_text(json.dumps({"runs": [{"task_id": "t", "score": 0.0}]}))
    assert from_json(p2).n_runs == 1


def test_from_csv_roundtrip(tmp_path):
    p = tmp_path / "runs.csv"
    p.write_text("task_id,score,epoch\nt1,1.0,0\nt1,0.0,1\nt2,0.5,0\n")
    rs = from_csv(p)
    assert rs.n_tasks == 2
    assert rs.scores_by_task()["t1"].tolist() == [1.0, 0.0]


def test_from_inspect_log_stub(tmp_path):
    log = {
        "eval": {"model": "demo/model", "task": "swe"},
        "samples": [
            {"id": "s1", "epoch": 1, "score": {"value": "C"}},
            {"id": "s1", "epoch": 2, "score": {"value": "I"}},
            {"id": "s2", "epoch": 1, "score": {"value": 0.5}},
        ],
    }
    p = tmp_path / "log.json"
    p.write_text(json.dumps(log))
    rs = from_inspect_log(p)
    assert rs.agent == "demo/model"
    assert rs.n_tasks == 2
    assert rs.scores_by_task()["s1"].tolist() == [1.0, 0.0]
    assert rs.runs[0].run_id == "1"

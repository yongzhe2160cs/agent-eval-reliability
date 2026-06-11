"""Ingest adapters: turn eval result files into a :class:`RunSet`.

Framework-agnostic by design. The generic JSON/CSV adapters accept a flat list
of run records; the Inspect adapter is a *stub* that documents the expected
shape of an Inspect ``.eval`` log without requiring the dependency (we only
have the public log schema to go on, not real runs).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .datamodel import RunSet, TaskRun

__all__ = [
    "from_records",
    "from_json",
    "from_csv",
    "from_inspect_log",
]

# default field names looked up in each record (first match wins)
_TASK_KEYS = ("task_id", "task", "sample_id", "id")
_SCORE_KEYS = ("score", "value", "reward", "result")
_RUN_KEYS = ("run_id", "epoch", "run", "attempt")
_SEED_KEYS = ("seed", "sample_seed")
_AGENT_KEYS = ("agent", "model", "config", "solver")


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return None


def _coerce_score(raw: Any) -> float:
    """Map common scoring conventions onto ``[0, 1]``.

    Accepts floats already in range, booleans, and Inspect-style letter grades
    ``C`` (correct) / ``I`` (incorrect) / ``P`` (partial).
    """
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip().upper()
        mapping = {"C": 1.0, "I": 0.0, "P": 0.5, "CORRECT": 1.0, "INCORRECT": 0.0}
        if token in mapping:
            return mapping[token]
        return float(raw)
    raise ValueError(f"cannot interpret score value {raw!r}")


def from_records(
    records: list[dict[str, Any]],
    *,
    agent: str | None = None,
    task_key: str | None = None,
    score_key: str | None = None,
) -> RunSet:
    """Build a :class:`RunSet` from a list of dict records.

    Field names are auto-detected from common aliases; override with
    ``task_key`` / ``score_key`` when your schema differs.
    """
    rs = RunSet(agent=agent)
    for i, rec in enumerate(records):
        tid = rec.get(task_key) if task_key else _first(rec, _TASK_KEYS)
        if tid is None:
            raise ValueError(f"record {i} has no task id (looked for {_TASK_KEYS})")
        raw_score = rec.get(score_key) if score_key else _first(rec, _SCORE_KEYS)
        if raw_score is None:
            raise ValueError(f"record {i} has no score (looked for {_SCORE_KEYS})")
        known = set(_TASK_KEYS + _SCORE_KEYS + _RUN_KEYS + _SEED_KEYS + _AGENT_KEYS)
        meta = {k: v for k, v in rec.items() if k not in known}
        run_raw = _first(rec, _RUN_KEYS)
        seed_raw = _first(rec, _SEED_KEYS)
        rs.runs.append(
            TaskRun(
                task_id=str(tid),
                score=_coerce_score(raw_score),
                run_id=None if run_raw is None else str(run_raw),
                seed=None if seed_raw is None else int(seed_raw),
                agent=_first(rec, _AGENT_KEYS) or agent,
                metadata=meta,
            )
        )
    return rs


def from_json(path: str | Path, **kwargs: Any) -> RunSet:
    """Load runs from a JSON file.

    Accepts either a top-level list of records or an object with a ``"runs"``
    (or ``"results"``/``"samples"``) list.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        for key in ("runs", "results", "samples", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError("JSON object has no runs/results/samples list")
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of records or contain one")
    return from_records(data, **kwargs)


def from_csv(path: str | Path, **kwargs: Any) -> RunSet:
    """Load runs from a CSV file with a header row."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return from_records(rows, **kwargs)


def from_inspect_log(path: str | Path, *, agent: str | None = None) -> RunSet:
    """STUB adapter for Inspect AI ``.eval`` logs.

    Inspect stores, per sample, an ``epoch`` (the run index when ``--epochs``
    > 1) and a ``score`` whose ``value`` is typically a letter grade (``C`` /
    ``I`` / ``P``) or a float. We map ``epoch`` -> ``run_id`` and the score
    value -> ``[0, 1]``.

    This stub reads the *JSON-encoded* form of a log (e.g. produced by
    ``inspect log dump``) so it works without importing ``inspect_ai``. The
    expected shape::

        {"eval": {"model": "...", "task": "..."},
         "samples": [{"id": "...", "epoch": 1, "score": {"value": "C"}}, ...]}

    A full version would call ``inspect_ai.log.read_eval_log`` and iterate
    ``log.samples`` directly.
    """
    data = json.loads(Path(path).read_text())
    eval_meta = data.get("eval", {}) if isinstance(data, dict) else {}
    model = eval_meta.get("model")
    task = eval_meta.get("task")
    samples = data.get("samples", []) if isinstance(data, dict) else []
    rs = RunSet(agent=agent or model)
    for s in samples:
        tid = s.get("id") or s.get("sample_id") or task
        score_obj = s.get("score", {})
        raw = score_obj.get("value") if isinstance(score_obj, dict) else score_obj
        rs.runs.append(
            TaskRun(
                task_id=str(tid),
                score=_coerce_score(raw),
                run_id=None if s.get("epoch") is None else str(s["epoch"]),
                agent=agent or model,
                metadata={"model": model, "task": task},
            )
        )
    return rs

# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in BUG-004 packed-round diagnostic."""

from __future__ import annotations

import json

import torch

from vllm_gaudi.ops.deepseek_v41_bug004_profile import (
    ENV,
    SCHEMA,
    RoundProfile,
    begin_round,
)


def _cache_metric():
    return {"available": False, "reason": "test"}


def test_disabled_observer_keeps_default_path(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(tmp_path))
    assert begin_round(rank=0, tp_rank=0, pp_rank=0, device="cpu") is None
    assert not any(tmp_path.iterdir())


def test_enabled_observer_requires_evidence(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.delenv("DSV41_RUN_EVIDENCE", raising=False)
    try:
        begin_round(rank=0, tp_rank=0, pp_rank=0, device="cpu")
    except RuntimeError as error:
        assert "DSV41_RUN_EVIDENCE" in str(error)
    else:
        raise AssertionError("enabled diagnostics must require an evidence directory")


def test_round_record_has_stable_segment_and_op_attribution(tmp_path):
    profile = RoundProfile(
        tmp_path,
        rank=3,
        tp_rank=2,
        pp_rank=1,
        device="cpu",
        cache_metric=_cache_metric,
    )
    torch.ones(1)
    profile.mark("target.forward")
    torch.zeros(1)
    profile.mark("draft.control_prepare")
    torch.tensor([1])
    record = profile.close()

    assert record["schema"] == SCHEMA
    assert record["rank"] == 3 and record["tp_rank"] == 2 and record["pp_rank"] == 1
    assert record["status"] == "complete"
    assert record["op_count"] == sum(segment["ops"] or 0 for segment in record["segments"])
    assert all(segment["ops"] is None or segment["ops"] == sum(segment["op_counts"].values()) for segment in record["segments"])
    assert record["profiler"]["available"] is False
    assert record["profiler"]["reason"] == "device is not HPU"

    lines = (tmp_path / "bug004-rounds.rank3.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["schema"] == SCHEMA


def test_enabled_observer_writes_one_jsonl_line_per_round(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "true")
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(tmp_path))
    for _ in range(2):
        profile = begin_round(rank=0, tp_rank=0, pp_rank=0, device="cpu")
        profile.mark("target.forward")
        torch.ones(1)
        profile.close()
    lines = (tmp_path / "bug004-rounds.rank0.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["status"] == "complete" for line in lines)

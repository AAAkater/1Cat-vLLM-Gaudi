# SPDX-License-Identifier: Apache-2.0
"""Device accepted-prefix restoration without host accepted-count reads."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_target import DSparkTargetSnapshot, _RowSnapshot
from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import DSparkPackedTargetStages


def _snapshot():
    bank = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    indices = torch.tensor([2, 4, 6])
    row = _RowSnapshot(bank, indices, bank.index_select(0, indices).clone(),
                       (((0, 0), (0, 1)), ((0, 3), (1, 1)), ((1, 5), )))
    return bank, DSparkTargetSnapshot((row, ), row.values.numel() * row.values.element_size(), 2)


@pytest.mark.parametrize("counts", [[0, 0], [1, 2], [6, 6], [3, 0]])
def test_device_restore_matches_host_owner_semantics(counts):
    bank, snapshot = _snapshot()
    reference, host = _snapshot()
    snapshot.prepare_device_restore()
    bank.add_(1000)
    reference.add_(1000)
    snapshot.restore_device(torch.tensor(counts))
    host.restore(counts)
    assert torch.equal(bank, reference)


def test_malformed_device_count_cannot_authorize_tentative_rows():
    bank, snapshot = _snapshot()
    before = bank.clone()
    snapshot.prepare_device_restore()
    bank.add_(1000)
    snapshot.restore_device(torch.tensor([7, 1]))
    assert torch.equal(bank[[2, 4, 6]], before[[2, 4, 6]])
    assert torch.equal(bank[0], before[0] + 1000)


def test_device_restore_requires_prepared_ownership_and_integer_lane_counts():
    bank, snapshot = _snapshot()
    with pytest.raises(RuntimeError, match="prepared"):
        snapshot.restore_device(torch.tensor([1, 2]))
    snapshot.prepare_device_restore()
    with pytest.raises(ValueError):
        snapshot.restore_device(torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError):
        snapshot.restore_device(torch.tensor([1]))


def test_prepared_restore_fullgraph_reuses_count_changes(monkeypatch):
    # Dynamo's stream tracker must not acquire a visible-but-busy HPU in CPU tests.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    bank, snapshot = _snapshot()
    snapshot.prepare_device_restore()
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(snapshot.restore_device, backend=backend, fullgraph=True, dynamic=False)
    counts = torch.tensor([1, 2])
    bank.add_(1000)
    compiled(counts)
    first = bank.clone()
    bank.add_(1000)
    counts.zero_()
    compiled(counts)
    assert len(graphs) == 1
    assert torch.equal(first[2], torch.tensor([1004.0, 1005.0]))
    assert torch.equal(bank[[2, 4, 6]], snapshot.rows[0].values)


def test_compact_request_counts_map_to_snapshot_before_host_finish():
    bank, snapshot = _snapshot()
    snapshot.prepare_device_restore()
    bank.add_(1000)
    owner = object.__new__(DSparkPackedTargetStages)
    owner.batch_capacity = 11
    owner.device = torch.device("cpu")

    def forbidden_host_restore(counts):
        raise AssertionError("host finish restored an already device-finished snapshot")

    target = SimpleNamespace(payload={
        "source_lanes": (7, 10), "batch": SimpleNamespace(host_request_active=(True, True)),
        "ticket": SimpleNamespace(snapshot=snapshot, restore=forbidden_host_restore),
        "finished": False, "device_finished": False, "engram_owner": None,
    })
    counts = torch.zeros(11, dtype=torch.int64)
    counts[7] = 1
    counts[10] = 2
    owner.finish_device(target, counts)
    after_device = bank.clone()
    owner.finish(target, counts)
    assert target.payload["finished"] and target.payload["device_finished"]
    assert torch.equal(after_device, bank)

def _two_layer_snapshot():
    """Two layers with identical owner layouts plus one distinct page layout."""
    owners = (((0, 0), (0, 1)), ((0, 3), (1, 1)), ((1, 5), ))
    banks, rows = [], []
    for _layer in range(2):
        for _kind in range(2):
            bank = torch.arange(16, dtype=torch.float32).reshape(8, 2) + 100 * len(banks)
            indices = torch.tensor([2, 4, 6])
            banks.append(bank)
            rows.append(_RowSnapshot(bank, indices, bank.index_select(0, indices).clone(), owners))
    page = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    page_indices = torch.tensor([1, 3])
    banks.append(page)
    rows.append(_RowSnapshot(page, page_indices, page.index_select(0, page_indices).clone(),
                             (((0, 2), ), ((1, 0), (1, 4)))))
    return banks, DSparkTargetSnapshot(tuple(rows), 0, 2)


@pytest.mark.parametrize("counts", [[0, 0], [1, 1], [2, 0], [3, 5], [4, 6], [6, 1]])
def test_device_restore_matches_host_restore_for_shared_owner_layouts(counts):
    banks, snapshot = _two_layer_snapshot()
    reference, host = _two_layer_snapshot()
    snapshot.prepare_device_restore()
    for bank, expected in zip(banks, reference, strict=True):
        bank.add_(1000)
        expected.add_(1000)
    snapshot.restore_device(torch.tensor(counts))
    host.restore(counts)
    for bank, expected in zip(banks, reference, strict=True):
        assert torch.equal(bank, expected)


def test_identical_owner_layouts_share_one_device_tensor():
    _banks, snapshot = _two_layer_snapshot()
    snapshot.prepare_device_restore()
    tensors = {id(row.device_owners) for row in snapshot.rows}
    layouts = {(row.owners, row.indices.device) for row in snapshot.rows}
    assert len(tensors) == len(layouts) == 2
    assert snapshot.rows[0].device_owners is snapshot.rows[3].device_owners
    assert snapshot.rows[4].device_owners is not snapshot.rows[0].device_owners
    assert snapshot.rows[4].device_owners.shape == (2, 2, 2)
    assert snapshot.rows[4].device_owners.dtype == torch.int64


def _owner(capacity=2):
    owner = object.__new__(DSparkPackedTargetStages)
    owner.batch_capacity = capacity
    owner.device = torch.device("cpu")
    return owner


def _target(snapshot, *, lanes=(0, 1), active=(True, True)):
    restores = []

    def restore(counts):
        restores.append(tuple(counts))
        snapshot.restore(counts)

    batch = SimpleNamespace(host_request_active=active, batch_capacity=len(lanes))
    payload = {
        "source_lanes": lanes, "batch": batch, "ticket": SimpleNamespace(snapshot=snapshot, restore=restore),
        "finished": False, "device_finished": False, "engram_owner": None,
    }
    return SimpleNamespace(payload=payload), restores


@pytest.mark.parametrize("accepted", [[7, 1], [-1, 0], [1, 9]])
def test_invalid_counts_roll_back_on_device_and_fail_host_finish(accepted):
    bank, snapshot = _snapshot()
    before = bank.clone()
    snapshot.prepare_device_restore()
    bank.add_(1000)
    owner = _owner()
    target, restores = _target(snapshot)
    owner.finish_device(target, torch.tensor(accepted))
    # Safe device fallback: every snapshot row is rolled back.
    assert torch.equal(bank[[2, 4, 6]], before[[2, 4, 6]])
    with pytest.raises(RuntimeError, match="outside 0..6"):
        owner.finish(target, accepted)
    assert not target.payload["finished"]
    assert restores == []
    owner.abort(target)
    assert target.payload["finished"]
    assert restores == [(0, 0)]
    assert torch.equal(bank[[2, 4, 6]], before[[2, 4, 6]])


def test_device_finished_host_finish_rejects_inactive_lane_counts():
    bank, snapshot = _snapshot()
    snapshot.prepare_device_restore()
    owner = _owner(4)
    target, _restores = _target(snapshot, lanes=(1, 3), active=(True, False))
    counts = torch.tensor([0, 2, 0, 1])
    owner.finish_device(target, counts)
    with pytest.raises(RuntimeError, match="inactive lanes"):
        owner.finish(target, counts)
    with pytest.raises(ValueError, match="one accepted value per lane"):
        owner.finish(target, [0, 2, 0])
    assert not target.payload["finished"]


def test_device_finished_host_finish_completes_engram_with_valid_counts():
    _bank, snapshot = _snapshot()
    snapshot.prepare_device_restore()
    owner = _owner()
    target, restores = _target(snapshot)
    completed = []
    target.payload["engram_owner"] = (SimpleNamespace(complete_batch=lambda t, a: completed.append((t, a))), "tk")
    owner.finish_device(target, torch.tensor([3, 6]))
    owner.finish(target, torch.tensor([3, 6]))
    assert completed == [("tk", (3, 6))]
    assert restores == []
    assert target.payload["finished"]


def test_abort_after_device_finish_restores_snapshot_values():
    bank, snapshot = _snapshot()
    before = bank.clone()
    snapshot.prepare_device_restore()
    bank.add_(1000)
    owner = _owner()
    target, restores = _target(snapshot)
    owner.finish_device(target, torch.tensor([4, 6]))
    assert not torch.equal(bank[[2, 4, 6]], before[[2, 4, 6]])
    owner.abort(target)
    assert restores == [(0, 0)]
    assert target.payload["finished"]
    assert torch.equal(bank[[2, 4, 6]], before[[2, 4, 6]])
    assert torch.equal(bank[0], before[0] + 1000)
    owner.abort(target)
    assert restores == [(0, 0)]

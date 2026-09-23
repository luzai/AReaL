import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.infra.rpc import rtensor
from areal.trainer import rl_trainer


@pytest.fixture
def backend(monkeypatch):
    class Backend:
        def __init__(self):
            self.ids = []
            self.deleted = []
            self.fail_node = None

        def store(self, tensor):
            sid = f"eval-cleanup-{len(self.ids)}"
            self.ids.append(sid)
            rtensor.store(sid, tensor)
            return sid

        async def delete(self, addr, ids):
            self.deleted.append(addr)
            if addr == self.fail_node:
                raise RuntimeError("cleanup failed")
            for sid in ids:
                rtensor.remove(sid)

        def fetch(self, shards):
            pytest.fail("Cleanup must not fetch image tensors")

    backend = Backend()
    monkeypatch.setattr(rtensor, "get_backend", lambda: backend)
    yield backend
    for sid in backend.ids:
        rtensor.remove(sid)


def test_repeated_evaluation_releases_nested_image_shards(backend, monkeypatch):
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    baseline = rtensor.storage_stats()
    trainer = object.__new__(rl_trainer.PPOTrainer)
    trainer.actor = Mock()
    trainer.actor.is_data_parallel_head.return_value = True
    trainer.valid_dataloader = [[{}, {}, {}]]
    trainer.config = SimpleNamespace(eval_gconfig=SimpleNamespace(n_samples=12))
    trainer.eval_rollout = Mock()

    def wait(count, timeout):
        # The preceding result must be released before consuming the next one.
        assert rtensor.storage_stats() == baseline
        assert count == 1 and timeout is None
        return [
            rtensor.RTensor.remotize(
                {"multi_modal_input": [{"pixel_values": torch.ones(1024)}]},
                node_addr="eval-worker",
            )
        ]

    trainer.eval_rollout.wait.side_effect = wait
    for _ in range(20):
        trainer._evaluate_fn("workflow", {})
        assert rtensor.storage_stats() == baseline
    assert trainer.eval_rollout.submit.call_count == 60
    assert all(c.kwargs["is_eval"] for c in trainer.eval_rollout.submit.call_args_list)


@pytest.mark.parametrize(
    "result", [None, [None], [], [{"reward": 1.0}], [torch.ones(1)]]
)
def test_cleanup_accepts_empty_rejected_and_local_results(result, backend):
    asyncio.run(rl_trainer._clear_eval_result(result))
    assert backend.deleted == []


def test_cleanup_attempts_all_nodes_and_propagates_failure(backend):
    backend.fail_node = "failed"
    results = [
        rtensor.RTensor.remotize(torch.ones(1), node_addr=addr)
        for addr in ("failed", "healthy")
    ]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(rl_trainer._clear_eval_result(results))
    assert set(backend.deleted) == {"failed", "healthy"}
    assert results[1].shard.shard_id not in rtensor._storage

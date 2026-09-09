from __future__ import annotations

from types import SimpleNamespace

import pytest

from areal.trainer.rl_trainer import PPOTrainer


class _FakeEngine:
    # A TrainController reports a controller-local world size of one.  Its
    # allocation is the source of truth for distributed RPC alignment.
    data_parallel_world_size = 1
    parallel_strategy = SimpleNamespace(dp_size=4)

    def __init__(self):
        self.calls = []

    def compute_logp(self, batch):
        values = [row["id"] * 10 for row in batch]
        self.calls.append([row["id"] for row in batch])
        return values


def test_logp_rpc_chunking_preserves_order(monkeypatch):
    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    trainer = PPOTrainer.__new__(PPOTrainer)
    engine = _FakeEngine()
    batch = [{"id": index} for index in range(48)]

    outputs = trainer._compute_logp_in_rpc_chunks(engine, batch, role="actor_test")

    assert len(engine.calls) == 12
    assert all(len(call) == 4 for call in engine.calls)
    assert [item for call in engine.calls for item in call] == list(range(48))
    assert outputs == [index * 10 for index in range(48)]


@pytest.mark.parametrize("chunk_size", [3, 6])
def test_logp_rpc_chunking_rejects_non_dp_aligned_chunks(monkeypatch, chunk_size):
    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", str(chunk_size))
    trainer = PPOTrainer.__new__(PPOTrainer)

    with pytest.raises(ValueError, match="multiple of the engine DP size"):
        trainer._compute_logp_in_rpc_chunks(
            _FakeEngine(), [{"id": index} for index in range(48)], role="actor_test"
        )

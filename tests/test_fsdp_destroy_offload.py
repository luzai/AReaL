"""CPU lifetime/order tests of the real destroy method, not FSDP kernel mocks."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from areal.api.cli_args import TrainEngineConfig
from areal.engine import fsdp_engine


@pytest.fixture
def engine(tmp_path):
    """Use a real local config constructor and tiny CPU-owned model/optimizer."""
    LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    ).save_pretrained(tmp_path)
    result = fsdp_engine.FSDPEngine(
        TrainEngineConfig(path=str(tmp_path), optimizer=None, backend="fsdp:d1p1t1")
    )
    result.model = torch.nn.Linear(2, 2)
    result.optimizer = torch.optim.SGD(result.model.parameters(), lr=0.1)
    result._initialized = True
    assert not torch.distributed.is_initialized()
    return result


def boundaries(monkeypatch, engine, *, failure=None):
    """Replace only TMS/platform side effects; retain actual object deletion/gc."""
    events = []
    refs = {
        name: weakref.ref(
            getattr(engine, name), lambda _, name=name: events.append(name)
        )
        for name in ("optimizer", "model")
    }

    def resume():
        events.append("resume")
        assert engine.is_offload is True
        assert engine.initialized
        assert refs["model"]() is not None and refs["optimizer"]() is not None
        if failure == "resume":
            raise RuntimeError("resume failure")

    def synchronize():
        events.append("synchronize")
        assert engine.is_offload is False
        assert engine.initialized
        assert refs["model"]() is not None and refs["optimizer"]() is not None
        if failure == "synchronize":
            raise RuntimeError("synchronize failure")

    def empty_cache():
        events.append("empty_cache")
        assert not engine.initialized
        assert refs["model"]() is None and refs["optimizer"]() is None

    monkeypatch.setattr(
        fsdp_engine, "torch_memory_saver", SimpleNamespace(resume=resume)
    )
    # No clear_memory/onload method is supplied: accidentally adding one fails.
    monkeypatch.setattr(
        fsdp_engine,
        "current_platform",
        SimpleNamespace(synchronize=synchronize, empty_cache=empty_cache),
    )
    return events, refs


def test_destroy_offloaded_resumes_and_synchronizes_before_free(engine, monkeypatch):
    """Paused storage is restored before the first model/optimizer deletion."""
    # Arrange
    engine.is_offload = True
    events, refs = boundaries(monkeypatch, engine)
    # Act
    engine.destroy()
    # Assert
    assert events == ["resume", "synchronize", "optimizer", "model", "empty_cache"]
    assert all(ref() is None for ref in refs.values())
    assert not engine.is_offload and not engine.initialized
    assert engine._offload_depth == 0


def test_destroy_not_offloaded_preserves_original_path(engine, monkeypatch):
    """A normal engine performs no new TMS call or synchronization."""
    # Arrange
    events, _ = boundaries(monkeypatch, engine)
    # Act
    engine.destroy()
    # Assert
    assert events == ["optimizer", "model", "empty_cache"]


def test_destroy_repeated_does_not_resume_again(engine, monkeypatch):
    """Repeated normal teardown keeps the existing idempotent deletion path."""
    # Arrange
    engine.is_offload = True
    events, _ = boundaries(monkeypatch, engine)
    # Act
    engine.destroy()
    engine.destroy()
    # Assert
    assert events == [
        "resume",
        "synchronize",
        "optimizer",
        "model",
        "empty_cache",
        "empty_cache",
    ]
    assert not engine.own_global_group


@pytest.mark.parametrize("depth", [-1, 1, 2])
@pytest.mark.parametrize("offloaded", [False, True])
def test_destroy_nonzero_context_depth_rejects_before_any_mutation(
    engine, monkeypatch, depth, offloaded
):
    """Even an onloaded engine inside a context must not be destroyed."""
    # Arrange
    engine._offload_depth = depth
    engine.is_offload = offloaded
    events, refs = boundaries(monkeypatch, engine)
    # Act / Assert
    with pytest.raises(RuntimeError, match="nonzero offload context depth"):
        engine.destroy()
    assert events == []
    assert engine.initialized and engine.is_offload is offloaded
    assert engine._offload_depth == depth
    assert all(ref() is not None for ref in refs.values())


def test_destroy_resume_failure_keeps_paused_objects_and_state(engine, monkeypatch):
    """A failed resume cannot fall through into freeing paused allocations."""
    # Arrange
    engine.is_offload = True
    events, refs = boundaries(monkeypatch, engine, failure="resume")
    # Act / Assert
    with pytest.raises(RuntimeError, match="resume failure"):
        engine.destroy()
    assert events == ["resume"]
    assert engine.initialized and engine.is_offload
    assert all(ref() is not None for ref in refs.values())


def test_destroy_sync_failure_records_resumed_state_but_does_not_free(
    engine, monkeypatch
):
    """After successful resume, sync failure propagates before object deletion."""
    # Arrange
    engine.is_offload = True
    events, refs = boundaries(monkeypatch, engine, failure="synchronize")
    # Act / Assert
    with pytest.raises(RuntimeError, match="synchronize failure"):
        engine.destroy()
    assert events == ["resume", "synchronize"]
    assert engine.initialized and not engine.is_offload
    assert all(ref() is not None for ref in refs.values())

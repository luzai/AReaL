from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.api import SaveLoadMeta, WeightUpdateMeta
from areal.api.cli_args import PPOConfig, SchedulingStrategy
from areal.trainer import rl_trainer


@pytest.fixture
def weight_sync(monkeypatch, tmp_path):
    trainer = object.__new__(rl_trainer.PPOTrainer)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(experiment_name="experiment", trial_name="trial")
    )
    trainer.tokenizer = object()
    trainer.processor = object()
    trainer.actor = Mock()
    trainer.rollout = Mock()
    trainer.rollout.get_version.return_value = 7
    trainer._offload_model = Mock()
    trainer._onload_model = Mock()
    ready = Mock()
    monkeypatch.setattr(rl_trainer.name_resolve, "add", ready)
    events = []
    for label, method in (
        ("save", trainer.actor.save),
        ("actor_offload", trainer._offload_model),
        ("rollout_onload", trainer.rollout.onload),
        ("ready", ready),
        ("reload", trainer.rollout.update_weights_from_disk),
        ("rollout_offload", trainer.rollout.offload),
        ("actor_onload", trainer._onload_model),
    ):
        method.side_effect = lambda *args, label=label, **kwargs: events.append(label)
    meta = WeightUpdateMeta(type="disk", path=str(tmp_path / "weights"), version=8)
    return trainer, meta, events, ready


def test_colocated_sync_success_releases_memory_before_each_wake(weight_sync):
    """Save and reload while only one model owns the GPU allocations."""
    trainer, meta, events, ready = weight_sync

    trainer._update_colocated_vllm_weights(meta)

    assert events == [
        "save",
        "actor_offload",
        "rollout_onload",
        "ready",
        "reload",
        "rollout_offload",
        "actor_onload",
    ]
    trainer.actor.save.assert_called_once_with(
        SaveLoadMeta(
            path=meta.path,
            weight_format="hf",
            with_optim=False,
            tokenizer=trainer.tokenizer,
            processor=trainer.processor,
        )
    )
    trainer._offload_model.assert_called_once_with(
        trainer.actor, role="actor_weight_sync"
    )
    trainer.rollout.onload.assert_called_once_with()
    trainer.rollout.update_weights_from_disk.assert_called_once_with(meta)
    assert ready.call_args.args[0] == rl_trainer.names.update_weights_from_disk(
        "experiment", "trial", 7
    )
    assert float(ready.call_args.args[1]) > 0
    assert ready.call_args.kwargs == {"keepalive_ttl": 120}
    trainer.actor.set_version.assert_not_called()
    trainer.rollout.set_version.assert_not_called()
    trainer.rollout.resume.assert_not_called()
    trainer.rollout.continue_generation.assert_not_called()


def test_colocated_sync_async_reload_is_awaited(weight_sync):
    """An async reload completes before sleeping the inference engine."""
    trainer, meta, events, _ = weight_sync
    trainer.rollout.update_weights_from_disk = AsyncMock(
        side_effect=lambda *_: events.append("async_reload")
    )

    trainer._update_colocated_vllm_weights(meta)

    trainer.rollout.update_weights_from_disk.assert_awaited_once_with(meta)
    assert events[-3:] == ["async_reload", "rollout_offload", "actor_onload"]


@pytest.mark.parametrize("failure", ["onload", "update_weights_from_disk"])
def test_colocated_sync_reload_failure_sleeps_before_restoring_actor(
    weight_sync, failure
):
    """A failed wake or reload still releases inference memory before actor wake."""
    trainer, meta, events, _ = weight_sync
    getattr(trainer.rollout, failure).side_effect = RuntimeError("reload failed")

    with pytest.raises(RuntimeError, match="reload failed"):
        trainer._update_colocated_vllm_weights(meta)

    assert events[-2:] == ["rollout_offload", "actor_onload"]
    trainer.rollout.set_version.assert_not_called()


def test_colocated_sync_sleep_failure_does_not_restore_actor(weight_sync):
    """Never restore actor allocations if inference memory may still be resident."""
    trainer, meta, _, _ = weight_sync
    trainer.rollout.offload.side_effect = RuntimeError("sleep failed")

    with pytest.raises(RuntimeError, match="sleep failed"):
        trainer._update_colocated_vllm_weights(meta)

    trainer._onload_model.assert_not_called()


def test_colocated_sync_save_failure_does_not_wake_rollout(weight_sync):
    """A failed checkpoint must not publish readiness or change memory ownership."""
    trainer, meta, _, ready = weight_sync
    trainer.actor.save.side_effect = RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        trainer._update_colocated_vllm_weights(meta)

    trainer._offload_model.assert_not_called()
    trainer._onload_model.assert_not_called()
    trainer.rollout.onload.assert_not_called()
    trainer.rollout.offload.assert_not_called()
    ready.assert_not_called()


@pytest.mark.parametrize(
    "changes", [{"type": "xccl"}, {"path": None}, {"use_lora": True}]
)
def test_colocated_sync_unsupported_weights_fail_before_save(weight_sync, changes):
    """Reject unsupported transfers without touching actor or inference state."""
    trainer, meta, events, _ = weight_sync
    for key, value in changes.items():
        setattr(meta, key, value)

    with pytest.raises(ValueError, match="full disk weights"):
        trainer._update_colocated_vllm_weights(meta)

    assert events == []


@pytest.mark.parametrize("release_before_rollout", [False, True])
def test_initial_offload_releases_training_models_once(
    weight_sync, release_before_rollout
):
    """Early release survives rollout initialization without duplicate offloads."""
    trainer, _, _, _ = weight_sync
    trainer._initial_train_offload_done = False
    trainer._initial_rollout_offload_done = False
    trainer._should_offload_rollout = True
    trainer._should_offload_actor = True
    trainer._should_offload_ref = True
    trainer._should_offload_critic = False
    trainer._should_offload_teacher = False
    trainer.ref = Mock()
    events = []
    trainer._offload_model.side_effect = lambda model, role: events.append(role)
    trainer._offload_rollout = Mock(side_effect=lambda: events.append("rollout"))

    if release_before_rollout:
        trainer._offload_train_models_initially()
        assert events == ["ref", "actor"]
    trainer._apply_initial_offload_policy()

    expected = (
        ["ref", "actor", "rollout"]
        if release_before_rollout
        else ["rollout", "ref", "actor"]
    )
    assert events == expected
    assert trainer._initial_train_offload_done is True
    assert trainer._offload_model.call_count == 2
    trainer._offload_rollout.assert_called_once_with()


@pytest.mark.parametrize("connect_fails", [False, True])
def test_colocated_connection_sleeps_rollout_and_cleans_up_actor(
    weight_sync, connect_fails
):
    """CUDA control RPC runs with actor awake and inference already asleep."""
    trainer, meta, events, _ = weight_sync
    trainer.weight_update_meta = meta
    trainer._initial_rollout_offload_done = False
    trainer._initial_train_offload_done = True
    trainer._should_offload_rollout = True
    trainer._offload_rollout = Mock(
        side_effect=lambda: events.append("rollout_offload")
    )

    def connect(*args):
        events.append("connect")
        assert trainer._initial_rollout_offload_done is True
        if connect_fails:
            raise RuntimeError("connect failed")

    trainer.actor.connect_engine.side_effect = connect
    if connect_fails:
        with pytest.raises(RuntimeError, match="connect failed"):
            trainer._connect_colocated_vllm()
    else:
        trainer._connect_colocated_vllm()
        trainer._apply_initial_offload_policy()

    assert events == ["rollout_offload", "actor_onload", "connect", "actor_offload"]
    trainer._offload_rollout.assert_called_once_with()
    trainer._onload_model.assert_called_once_with(trainer.actor, role="actor_connect")
    trainer.actor.connect_engine.assert_called_once_with(trainer.rollout, meta)
    trainer._offload_model.assert_called_once_with(trainer.actor, role="actor_connect")


def test_colocated_connection_sleep_failure_does_not_wake_actor(weight_sync):
    """A failed initial inference sleep prevents CUDA actor RPC from starting."""
    trainer, meta, _, _ = weight_sync
    trainer.weight_update_meta = meta
    trainer._initial_rollout_offload_done = False
    trainer._offload_rollout = Mock(side_effect=RuntimeError("sleep failed"))

    with pytest.raises(RuntimeError, match="sleep failed"):
        trainer._connect_colocated_vllm()

    assert trainer._initial_rollout_offload_done is False
    trainer._onload_model.assert_not_called()
    trainer.actor.connect_engine.assert_not_called()


@pytest.fixture
def colocated_config(monkeypatch, tmp_path):
    trainer = object.__new__(rl_trainer.PPOTrainer)
    trainer.config = PPOConfig(experiment_name="experiment", trial_name="trial")
    trainer.config.enable_offload = True
    trainer.config.actor.weight_update_mode = "disk"
    trainer.config.actor._version = "v1"
    trainer.config.rollout._version = "v1"
    trainer.config.rollout.scheduling_strategy = SchedulingStrategy(
        type="colocation", target="actor"
    )
    trainer.config.vllm.enable_sleep_mode = True
    trainer.config.recover.mode = "auto"
    trainer.config.recover.experiment_name = "experiment"
    trainer.config.recover.trial_name = "trial"
    trainer.config.recover.fileroot = str(tmp_path)
    trainer.actor_alloc = SimpleNamespace(backend="fsdp")
    trainer.rollout_alloc = SimpleNamespace(backend="vllm")
    trainer._colocated_vllm = True
    trainer._should_offload_rollout = True
    trainer._should_offload_actor = True
    trainer._should_offload_ref = False
    trainer._should_offload_critic = False
    trainer._should_offload_teacher = False
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    return trainer


def test_colocated_config_fresh_run_accepts_optimizer_saving(colocated_config):
    """Automatic recovery is allowed in a fresh directory with optimizer saving."""
    assert colocated_config.config.recover.no_save_optim is False
    colocated_config._validate_cfg()


def test_colocated_config_sleep_disabled_rejects_before_initialization(
    colocated_config,
):
    """Colocated inference must support releasing its GPU allocations."""
    colocated_config.config.vllm.enable_sleep_mode = False
    with pytest.raises(ValueError, match="enable_sleep_mode=True"):
        colocated_config._validate_cfg()


@pytest.mark.parametrize("mode", ["auto", "on", "off", "disabled"])
def test_colocated_config_existing_recover_info_requires_disabled_resume(
    colocated_config, mode
):
    """Only enabled recovery rejects an actual recover_info directory."""
    recover = colocated_config.config.recover
    recover.mode = mode
    recovery_dir = Path(
        rl_trainer.RecoverHandler.recover_info_path(
            recover.experiment_name, recover.trial_name, recover.fileroot
        )
    )
    recovery_dir.mkdir(parents=True)

    if mode in ("auto", "on"):
        with pytest.raises(ValueError, match="checkpoint resume is not supported"):
            colocated_config._validate_cfg()
    else:
        colocated_config._validate_cfg()


@pytest.mark.parametrize("unsupported", ["megatron", "v2", "lora", "multi_controller"])
def test_colocated_config_unsupported_engine_rejects(
    colocated_config, monkeypatch, unsupported
):
    """Fail unsupported backend/controller/adapter combinations before loading."""
    if unsupported == "megatron":
        colocated_config.actor_alloc.backend = "megatron"
    elif unsupported == "v2":
        colocated_config.config.actor._version = "v2"
        colocated_config.config.rollout._version = "v2"
    elif unsupported == "lora":
        colocated_config.config.actor.use_lora = True
    else:
        monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: False)

    with pytest.raises(ValueError, match="v1 single-controller FSDP"):
        colocated_config._validate_cfg()


def test_separated_config_keeps_existing_backend_and_xccl_support(colocated_config):
    """New colocated restrictions do not constrain the separated inference path."""
    trainer = colocated_config
    trainer._colocated_vllm = False
    trainer._should_offload_actor = False
    trainer._should_offload_rollout = False
    trainer.config.enable_offload = False
    trainer.config.rollout.scheduling_strategy = SchedulingStrategy(type="separation")
    trainer.actor_alloc.backend = "megatron"
    trainer.config.actor.weight_update_mode = "xccl"
    trainer.config.actor._version = "v2"
    trainer.config.rollout._version = "v2"
    trainer.config.vllm.enable_sleep_mode = False

    trainer._validate_cfg()

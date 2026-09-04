from unittest.mock import Mock

import pytest

from areal.trainer.rl_trainer import PPOTrainer, _parse_update_gate_global_steps


def test_update_gate_steps_support_staged_iteration_release():
    assert _parse_update_gate_global_steps("17,18,19", "16") == {17, 18, 19}


def test_update_gate_steps_keep_legacy_single_gate_compatibility():
    assert _parse_update_gate_global_steps(None, "17") == {17}


@pytest.mark.parametrize("value", [",", "-1", "17,bad"])
def test_update_gate_steps_reject_invalid_values(value):
    with pytest.raises(ValueError, match="invalid managed update gate steps"):
        _parse_update_gate_global_steps(value, None)


def test_clear_consumed_rollout_batch_fans_out_to_all_consumers():
    trainer = object.__new__(PPOTrainer)
    trainer.actor = Mock()
    trainer.critic = Mock()
    trainer.ref = Mock()
    rollout_batch = [{"sample": object()}]

    trainer._clear_consumed_rollout_batch(rollout_batch)

    trainer.actor.clear_batches.assert_called_once_with(rollout_batch)
    trainer.critic.clear_batches.assert_called_once_with(rollout_batch)
    trainer.ref.clear_batches.assert_called_once_with(rollout_batch)


def test_clear_consumed_rollout_batch_handles_optional_consumers():
    trainer = object.__new__(PPOTrainer)
    trainer.actor = Mock()
    trainer.critic = None
    trainer.ref = None
    rollout_batch = [{"sample": object()}]

    trainer._clear_consumed_rollout_batch(rollout_batch)

    trainer.actor.clear_batches.assert_called_once_with(rollout_batch)


def test_reset_actor_memory_before_ppo_rebuilds_offloaded_actor_allocator():
    trainer = object.__new__(PPOTrainer)
    trainer.actor = Mock()
    trainer._should_offload_actor = True
    trainer._offload_model = Mock()
    trainer._onload_model = Mock()

    trainer._reset_actor_memory_before_ppo(did_recompute_logp=True)

    trainer._offload_model.assert_called_once_with(
        trainer.actor, role="actor_pre_ppo_reset"
    )
    trainer._onload_model.assert_called_once_with(
        trainer.actor, role="actor_pre_ppo_reset"
    )
    trainer.actor.get_device_stats.return_value.log.assert_called_once_with(
        "actor memory reset before ppo"
    )


def test_reset_actor_memory_before_ppo_skips_without_recompute_or_offload():
    trainer = object.__new__(PPOTrainer)
    trainer.actor = Mock()
    trainer._offload_model = Mock()
    trainer._onload_model = Mock()

    trainer._should_offload_actor = False
    trainer._reset_actor_memory_before_ppo(did_recompute_logp=True)
    trainer._should_offload_actor = True
    trainer._reset_actor_memory_before_ppo(did_recompute_logp=False)

    trainer._offload_model.assert_not_called()
    trainer._onload_model.assert_not_called()
    trainer.actor.get_device_stats.assert_not_called()

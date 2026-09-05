"""Regress non-finite config bounds crossing the real JSON RPC boundary.

These CPU tests validate transport and PPO advantage semantics, not distributed
training or acceptance of non-finite task rewards by a rollout workflow.
"""

import math
from typing import Any

import orjson
import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.actor import PPOActor


def _json_rpc_roundtrip(value: Any) -> Any:
    """Include the wire codec; a Python-only round-trip misses null coercion."""
    return deserialize_value(orjson.loads(orjson.dumps(serialize_value(value))))


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_float_json_roundtrip_preserves_value(value: float):
    """Infinity and NaN remain floats rather than becoming JSON null."""
    assert orjson.loads(orjson.dumps(value)) is None

    serialized = serialize_value(value)
    assert serialized == {"type": "nonfinite_float", "value": str(value)}
    restored = _json_rpc_roundtrip(value)

    assert isinstance(restored, float)
    if math.isnan(value):
        assert math.isnan(restored)
    else:
        assert restored == value


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 42, 0.0, -0.0, 20.0, 1.25, "inf", "-inf", "nan", ""],
)
def test_ordinary_primitive_json_roundtrip_keeps_type_and_value(value: Any):
    """Finite values, None and strings do not acquire special marker semantics."""
    assert serialize_value(value) == value

    restored = _json_rpc_roundtrip(value)

    assert type(restored) is type(value)
    assert restored == value
    if isinstance(value, float):
        assert math.copysign(1.0, restored) == math.copysign(1.0, value)


def test_nested_nonfinite_json_roundtrip_preserves_collection_values():
    """Nested dicts, lists and tuple-to-list conversion retain every bound."""
    original = {
        "bounds": [math.inf, {"lower": -math.inf, "missing": None}],
        "extra": (math.nan, 20.0, "inf"),
    }

    restored = _json_rpc_roundtrip(original)

    assert restored["bounds"] == [math.inf, {"lower": -math.inf, "missing": None}]
    assert isinstance(restored["extra"], list)
    assert math.isnan(restored["extra"][0])
    assert restored["extra"][1:] == [20.0, "inf"]


@pytest.mark.parametrize("reward_clip", [math.inf, -math.inf, math.nan, 20.0])
def test_ppo_config_json_roundtrip_preserves_float_field(reward_clip: float):
    """Dataclass transport retains field state, without validating PPO bounds."""
    config = PPOActorConfig(backend="fsdp:d1", reward_clip=reward_clip)

    restored = _json_rpc_roundtrip({"args": [config]})["args"][0]

    assert isinstance(restored, PPOActorConfig)
    assert isinstance(restored.reward_clip, float)
    if math.isnan(reward_clip):
        assert math.isnan(restored.reward_clip)
    else:
        assert restored.reward_clip == reward_clip
    assert restored.backend == config.backend
    assert restored.reward_bias == config.reward_bias
    assert restored.reward_scaling == config.reward_scaling
    assert restored.gradient_checkpointing == config.gradient_checkpointing


@pytest.mark.parametrize(
    "value", [None, True, 20, "20.0", "Infinity", "-Infinity", "NaN", "", [], {}]
)
def test_nonfinite_marker_invalid_value_is_rejected(value: Any):
    """Malformed marker values cannot silently become a different float."""
    wire = orjson.dumps({"type": "nonfinite_float", "value": value})

    with pytest.raises(ValueError, match="non.?finite"):
        deserialize_value(orjson.loads(wire))


def test_nonfinite_marker_missing_value_is_rejected():
    """A missing marker value is an invalid transport payload, not None."""
    wire = orjson.dumps({"type": "nonfinite_float"})

    with pytest.raises(ValueError, match="non.?finite"):
        deserialize_value(orjson.loads(wire))


def test_nonfinite_tensor_json_roundtrip_does_not_sanitize_invalid_rewards():
    """Transport preserves invalid tensor values for existing caller validation."""
    rewards = torch.tensor([math.inf, -math.inf, math.nan, 31.0], dtype=torch.float32)

    restored = _json_rpc_roundtrip({"rewards": rewards})["rewards"]

    torch.testing.assert_close(restored, rewards, rtol=0.0, atol=0.0, equal_nan=True)
    torch.testing.assert_close(
        torch.isfinite(restored),
        torch.tensor([False, False, False, True]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("reward_clip", "expected"),
    [(math.inf, [31.0, -47.0, 20.0, 7.0]), (20.0, [20.0, -20.0, 20.0, 7.0])],
    ids=["c1-unbounded", "c2-finite-20"],
)
def test_ppo_advantages_after_json_rpc_preserve_stage_clipping(
    reward_clip: float, expected: list[float]
):
    """Real CPU PPO advantages keep C1 rewards above 20 and clip C2 to 20."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        reward_clip=reward_clip,
        reward_norm=None,
        adv_norm=None,
        reward_bias=0.0,
        reward_scaling=1.0,
        kl_ctl=0.0,
        use_decoupled_loss=True,
        prox_logp_method="recompute",
    )
    raw_rewards = torch.tensor([31.0, -47.0, 20.0, 7.0], dtype=torch.float32)
    data = {
        "input_ids": torch.ones(4, 3, dtype=torch.long),
        "attention_mask": torch.ones(4, 3, dtype=torch.bool),
        # Sample-token mask; the actor rolls it to causal prediction positions.
        "loss_mask": torch.tensor([[0, 0, 1]] * 4, dtype=torch.bool),
        "logprobs": torch.zeros(4, 3, dtype=torch.float32),
        "ref_logp": torch.zeros(4, 3, dtype=torch.float32),
        "rewards": raw_rewards.clone(),
    }
    received = _json_rpc_roundtrip({"config": config, "data": [data]})
    # This compute-only API never accesses the engine. No FSDP/DTensor mock,
    # model construction, distributed initialization or GPU is needed.
    actor = PPOActor(received["config"], engine=None)

    results = actor.compute_advantages(received["data"])

    assert len(results) == 1
    result = results[0]
    expected_rewards = torch.tensor(expected, dtype=torch.float32)
    expected_total = torch.zeros(4, 3, dtype=torch.float32)
    expected_total[:, 1] = expected_rewards
    torch.testing.assert_close(
        result["tot_rewards"], expected_total, rtol=0.0, atol=0.0
    )
    for key in ("advantages", "returns"):
        torch.testing.assert_close(
            result[key][:, 1], expected_rewards, rtol=0.0, atol=0.0
        )
        assert torch.isfinite(result[key]).all()
    torch.testing.assert_close(result["rewards"], raw_rewards, rtol=0.0, atol=0.0)

"""CPU regressions for PPO dispatch of grouped C1 and singleton C2 batches.

These exercise real tensor/RTensor metadata and local advantage/concat/padding
code, without a model, FSDP mock, distributed process group, or RPC server.
"""

import math
from types import SimpleNamespace
from typing import Any

import orjson
import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.infra.controller.train_controller import TrainController
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.actor import PPOActor, _pad_ppo_batch_to_size
from areal.utils.data import batched_call, concat_batch


def _controller(dp_size: int = 4) -> TrainController:
    controller = object.__new__(TrainController)
    controller.train_alloc = SimpleNamespace(parallel=SimpleNamespace(dp_size=dp_size))
    return controller


def _group(rows: int, length: int = 3) -> dict[str, torch.Tensor]:
    loss_mask = torch.zeros(rows, length, dtype=torch.bool)
    loss_mask[:, -1] = True
    return {
        "input_ids": torch.ones(rows, length, dtype=torch.long),
        "attention_mask": torch.ones(rows, length, dtype=torch.bool),
        "loss_mask": loss_mask,
        "logprobs": torch.zeros(rows, length, dtype=torch.float32),
        "rewards": torch.ones(rows, dtype=torch.float32),
    }


def _remote_shape(group: dict[str, torch.Tensor], index: int) -> dict[str, Any]:
    """Use real metadata-only RTensors and the actual JSON wire codec."""
    remote = {
        key: RTensor(
            shard=TensorShardInfo(
                shard_id=f"fixture-{index}-{key}", node_addr="unused"
            ),
            data=value.to("meta"),
        )
        for key, value in group.items()
    }
    return deserialize_value(orjson.loads(orjson.dumps(serialize_value(remote))))


def test_c1_advantages_keep_group_rows_for_ppo_dispatch_and_worker_concat():
    """Regress smoke08: four 12 x 512 groups mean 6144 rows/rank, not one."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        reward_clip=math.inf,
        reward_norm=None,
        adv_norm=None,
        reward_bias=0.0,
        reward_scaling=1.0,
        kl_ctl=0.0,
        use_decoupled_loss=True,
        prox_logp_method="recompute",
    )
    actor = PPOActor(config, engine=None)
    groups = [_group(12 * 512) for _ in range(4)]

    # C1 does not request repartition_for_ppo; real compute preserves groups.
    advantages = actor.compute_advantages(groups)
    assert len(advantages) == 4
    assert [item["attention_mask"].shape[0] for item in advantages] == [6144] * 4
    dp_args, dp_kwargs, indices = _controller()._prepare_ppo_dispatch(advantages)

    assert indices == [[0], [1], [2], [3]]
    assert dp_kwargs["_ppo_target_batch_size"] == [6144] * 4
    for rank, rank_groups in enumerate(dp_args[0]):
        # This is the same concat-and-call boundary used by PPOActor.ppo_update.
        padded = batched_call(
            lambda data: _pad_ppo_batch_to_size(
                data, dp_kwargs["_ppo_target_batch_size"][rank]
            ),
            rank_groups,
            unpack=False,
        )
        assert padded["attention_mask"].shape == (6144, 3)
        assert padded["loss_mask"].count_nonzero().item() == 6144
        assert torch.isfinite(padded["advantages"]).all()


@pytest.mark.parametrize("representation", ["tensor", "rtensor-json", "mixed"])
@pytest.mark.parametrize(
    ("shapes", "expected_local_rows"),
    [
        ([(6144, 3)] * 4, [6144] * 4),
        ([(9, 3), (7, 3), (5, 3), (3, 3), (2, 3)], [9, 7, 5, 5]),
        ([(4, 3)] * 8, [8] * 4),
        ([(2, 20), (8, 3), (7, 3), (6, 3), (3, 2)], [2, 8, 7, 9]),
        ([(1, 3)] * 9, [3, 2, 2, 2]),
    ],
    ids=[
        "smoke08",
        "uneven-groups",
        "sum-multiple-groups",
        "tokens-not-rows",
        "singleton",
    ],
)
def test_ppo_target_sums_actual_rows_per_rank_without_remote_fetch(
    shapes: list[tuple[int, int]],
    expected_local_rows: list[int],
    representation: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Count leading mask dimensions, not groups, tokens, or largest single group."""
    groups = [_group(rows, length) for rows, length in shapes]
    remote_indices = {
        i
        for i in range(len(groups))
        if representation == "rtensor-json" or (representation == "mixed" and i % 2)
    }
    items = [
        _remote_shape(group, i) if i in remote_indices else group
        for i, group in enumerate(groups)
    ]

    def no_fetch(*args, **kwargs):
        pytest.fail("PPO controller row counting must not fetch remote payloads")

    monkeypatch.setattr(RTensor, "to_local", no_fetch)
    monkeypatch.setattr("areal.infra.rpc.rtensor.get_backend", no_fetch)
    for i in remote_indices:
        assert items[i]["attention_mask"].data.is_meta
        assert tuple(items[i]["attention_mask"].data.shape) == shapes[i]

    dp_args, dp_kwargs, indices = _controller()._prepare_ppo_dispatch(items)

    assert sorted(index for rank in indices for index in rank) == list(
        range(len(items))
    )
    local_rows = [sum(shapes[index][0] for index in rank) for rank in indices]
    assert local_rows == expected_local_rows
    target = max(expected_local_rows)
    assert dp_kwargs["_ppo_target_batch_size"] == [target] * 4
    assert [len(rank) for rank in dp_args[0]] == [len(rank) for rank in indices]

    # Compare against the actual worker-local concat on original CPU payloads.
    for rank, rank_indices in enumerate(indices):
        local, _ = concat_batch([groups[index] for index in rank_indices])
        assert local["attention_mask"].shape[0] == local_rows[rank]
        before_loss = local["loss_mask"].sum().item()
        padded = _pad_ppo_batch_to_size(local, target)
        assert padded["attention_mask"].shape[0] == target
        assert padded["loss_mask"].sum().item() == before_loss
        assert padded["loss_mask"][local_rows[rank] :].count_nonzero().item() == 0
        torch.testing.assert_close(
            padded["input_ids"][: local_rows[rank]],
            local["input_ids"],
            rtol=0.0,
            atol=0.0,
        )


def test_grouped_ppo_keyword_dispatch_uses_rows_and_preserves_scalar_kwargs():
    """The same row-count contract holds when the batch is a keyword argument."""
    groups = [_group(5), _group(3), _group(2), _group(1)]

    dp_args, dp_kwargs, indices = _controller()._prepare_ppo_dispatch(
        data=groups, flag=True
    )

    assert dp_args == []
    assert indices == [[0], [1], [2], [3]]
    assert dp_kwargs["_ppo_target_batch_size"] == [5] * 4
    assert dp_kwargs["flag"] == [True] * 4
    assert [rank[0]["attention_mask"].shape[0] for rank in dp_kwargs["data"]] == [
        5,
        3,
        2,
        1,
    ]


def test_c2_6378_singleton_rows_keep_1595_synchronized_target():
    """Fixing C1 groups must retain the prior non-divisible C2 DP4 guarantee."""
    rows = [{"attention_mask": torch.ones(1, 2, dtype=torch.bool)} for _ in range(6378)]

    dp_args, dp_kwargs, indices = _controller()._prepare_ppo_dispatch(rows)

    assert sorted(map(len, indices)) == [1594, 1594, 1595, 1595]
    assert dp_kwargs["_ppo_target_batch_size"] == [1595] * 4
    assert sum(len(rank) for rank in dp_args[0]) == 6378
    assert sorted(index for rank in indices for index in rank) == list(range(6378))


@pytest.mark.parametrize("remote", [False, True], ids=["tensor", "rtensor"])
@pytest.mark.parametrize("shape", [(), (3,), (0, 3), (1, 2, 3)])
def test_grouped_ppo_dispatch_rejects_invalid_attention_mask_shapes(
    shape: tuple[int, ...], remote: bool
):
    """Scalar, 1-D, empty-row and 3-D metadata cannot define a valid row count."""
    malformed = {"attention_mask": torch.ones(shape, dtype=torch.bool)}
    if remote:
        malformed = _remote_shape(malformed, 0)
    groups = [malformed] + [_group(1) for _ in range(3)]

    with pytest.raises(ValueError, match="non-empty 2D attention_mask"):
        _controller()._prepare_ppo_dispatch(groups)


@pytest.mark.parametrize("mask", [None, "not-a-tensor"])
def test_grouped_ppo_dispatch_rejects_missing_or_non_tensor_mask(mask: Any):
    """Another tensor in the item must not silently substitute for the mask."""
    malformed = {"input_ids": torch.ones(1, 3, dtype=torch.long)}
    if mask is not None:
        malformed["attention_mask"] = mask
    groups = [malformed] + [_group(1) for _ in range(3)]

    with pytest.raises(ValueError, match="tensor attention_mask per item"):
        _controller()._prepare_ppo_dispatch(groups)

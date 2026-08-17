"""Focused regressions for post-advantage PPO option repartitioning."""

from types import SimpleNamespace

import pytest
import torch

from areal.infra.controller.train_controller import (
    TrainController,
    _dispatch_ppo_tensors,
    _merge_tensors,
)
from areal.trainer.ppo.actor import (
    _flatten_ppo_advantage_groups,
    _pad_ppo_batch_to_size,
    _split_ppo_batch_rows,
)
from areal.utils.seqpack import ffd_allocate
from areal.v2.training_service.data_proxy.dispatcher import DispatchRequest


def _gate3_sequence_lengths() -> list[list[int]]:
    """Return a deterministic fixture matching the failed Gate 3 loads."""

    return [
        [558] * 1043 + [373] * 807 + [374] * 236 + [299] * 141,
        [558] * 995 + [362] * 390 + [363] * 605 + [299] * 108,
        [558] * 836 + [358] * 605 + [359] * 231 + [299] * 84,
        [558] * 186 + [402] * 114 + [403] * 72 + [299] * 108,
    ]


def test_split_ppo_batch_rows_preserves_fields_and_multimodal_entries():
    """Every PPO field stays attached to its trimmed singleton sequence."""

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    token_values = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    token_values = token_values * attention_mask
    vector_values = token_values.unsqueeze(-1).repeat(1, 1, 2)
    data = {
        "input_ids": token_values.long(),
        "attention_mask": attention_mask,
        "loss_mask": attention_mask.clone(),
        "advantages": token_values + attention_mask,
        "logprobs": token_values / 10,
        "prox_logp": token_values / 20,
        "ref_logp": token_values / 30,
        "episode_loss_weights": token_values / 40,
        "pacman_allowed_token_ids": vector_values,
        "rewards": torch.tensor([1.0, 2.0, 3.0]),
        "versions": torch.tensor([7, 7, 7]),
        "option_ids": torch.tensor([100, 101, 102]),
        "begin_of_trajectory": torch.tensor([1, 0, 0]),
        "multi_modal_input": [
            {"sample": 100},
            {"sample": 101},
            {"sample": 102},
        ],
        "shared": {"contract": "option_return_raw_v1"},
    }

    rows = _split_ppo_batch_rows(data)

    assert len(rows) == 3
    assert [row["attention_mask"].shape[1] for row in rows] == [5, 3, 4]
    for row in rows:
        for value in row.values():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == 1
    assert [row["multi_modal_input"][0]["sample"] for row in rows] == [
        100,
        101,
        102,
    ]
    assert rows[0]["multi_modal_input"] is not data["multi_modal_input"]
    assert [int(row["option_ids"].item()) for row in rows] == [100, 101, 102]

    for key in (
        "attention_mask",
        "loss_mask",
        "advantages",
        "logprobs",
        "prox_logp",
        "ref_logp",
        "episode_loss_weights",
        "pacman_allowed_token_ids",
        "rewards",
        "versions",
        "begin_of_trajectory",
    ):
        actual = sum(row[key].sum() for row in rows)
        expected = data[key].sum()
        if expected.dtype.is_floating_point:
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        else:
            assert torch.equal(actual, expected)


def test_split_ppo_batch_rows_rejects_noncontiguous_attention_mask():
    """The splitter fails closed instead of cropping a non-right-padded row."""

    data = {
        "attention_mask": torch.tensor([[1, 0, 1]], dtype=torch.bool),
        "input_ids": torch.tensor([[1, 0, 2]]),
    }

    with pytest.raises(ValueError, match="right-padded"):
        _split_ppo_batch_rows(data)


def test_nested_worker_results_keep_merge_contract_before_flattening():
    """Variable inner row counts do not violate the controller merge contract."""

    group0 = [{"option_ids": torch.tensor([0])}, {"option_ids": torch.tensor([1])}]
    group1 = [{"option_ids": torch.tensor([2])}]
    group2 = [
        {"option_ids": torch.tensor([3])},
        {"option_ids": torch.tensor([4])},
        {"option_ids": torch.tensor([5])},
    ]

    merged = _merge_tensors([[group2, group0], [group1]], [[2, 0], [1]])
    flattened = _flatten_ppo_advantage_groups(merged)

    assert [int(row["option_ids"].item()) for row in flattened] == list(range(6))


def test_ppo_batch_padding_clones_valid_input_with_zero_loss_mass():
    """Padding preserves forward schema but cannot affect PPO loss or inputs."""

    data = {
        "input_ids": torch.tensor([[11, 12, 13], [21, 22, 0]]),
        "attention_mask": torch.tensor(
            [[1, 1, 1], [1, 1, 0]], dtype=torch.bool
        ),
        "loss_mask": torch.tensor(
            [[1, 1, 1], [1, 1, 0]], dtype=torch.bool
        ),
        "episode_loss_weights": torch.tensor(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]
        ),
        "pacman_allowed_token_ids": torch.arange(24).reshape(2, 3, 4),
        "option_ids": torch.tensor([100, 101]),
        "multi_modal_input": [
            {"pixel_values": torch.tensor([1.0])},
            {"pixel_values": torch.tensor([2.0])},
        ],
    }
    original = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }

    padded = _pad_ppo_batch_to_size(data, 3)

    assert padded["attention_mask"].shape[0] == 3
    torch.testing.assert_close(padded["input_ids"][2], data["input_ids"][1])
    torch.testing.assert_close(
        padded["attention_mask"][2], data["attention_mask"][1]
    )
    torch.testing.assert_close(
        padded["pacman_allowed_token_ids"][2],
        data["pacman_allowed_token_ids"][1],
    )
    assert padded["loss_mask"][2].count_nonzero() == 0
    assert padded["episode_loss_weights"][2].count_nonzero() == 0
    assert padded["multi_modal_input"][2] is not data["multi_modal_input"][1]
    torch.testing.assert_close(
        padded["multi_modal_input"][2]["pixel_values"],
        data["multi_modal_input"][1]["pixel_values"],
    )
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(data[key], value)


def test_dp4_nondivisible_ppo_rows_are_feasible_after_local_padding():
    """Regress the 6378-row, 1594<1595 synchronized seqpack failure."""

    sequence_count = 6378
    sequence_length = 600
    capacity = 1024
    items = [
        {
            "attention_mask": torch.ones(1, sequence_length, dtype=torch.bool),
            "option_ids": torch.tensor([option_id]),
        }
        for option_id in range(sequence_count)
    ]

    splits, group_indices = _dispatch_ppo_tensors(items, dp_size=4)
    local_counts = [len(split) for split in splits]
    assert sorted(local_counts) == [1594, 1594, 1595, 1595]
    assert sorted(index for shard in group_indices for index in shard) == list(
        range(sequence_count)
    )

    target_count = max(local_counts)
    real_option_ids: list[int] = []
    total_real_loss_mass = 0
    total_padded_loss_mass = 0
    for split in splits:
        local_count = len(split)
        data = {
            "attention_mask": torch.ones(
                local_count, sequence_length, dtype=torch.bool
            ),
            "loss_mask": torch.ones(
                local_count, sequence_length, dtype=torch.bool
            ),
            "episode_loss_weights": torch.ones(
                local_count, sequence_length, dtype=torch.float32
            ),
            "option_ids": torch.tensor(
                [int(item["option_ids"].item()) for item in split]
            ),
        }
        total_real_loss_mass += int(data["loss_mask"].sum().item())
        padded = _pad_ppo_batch_to_size(data, target_count)
        total_padded_loss_mass += int(padded["loss_mask"].sum().item())

        assert padded["attention_mask"].shape[0] == target_count
        lengths = padded["attention_mask"].sum(-1).tolist()
        assert len(ffd_allocate(lengths, capacity, min_groups=target_count)) == (
            target_count
        )
        assert padded["loss_mask"][local_count:].count_nonzero() == 0
        assert (
            padded["episode_loss_weights"][local_count:].count_nonzero() == 0
        )
        real_option_ids.extend(padded["option_ids"][:local_count].tolist())

    assert sorted(real_option_ids) == list(range(sequence_count))
    assert total_padded_loss_mass == total_real_loss_mass


def test_controller_propagates_one_ppo_padding_target_to_every_dp_rank():
    """The controller supplies the target without a worker-side collective."""

    controller = object.__new__(TrainController)
    controller.train_alloc = SimpleNamespace(
        parallel=SimpleNamespace(dp_size=4)
    )
    rows = [
        {
            "attention_mask": torch.ones(1, 600, dtype=torch.bool),
            "option_ids": torch.tensor([option_id]),
        }
        for option_id in range(6378)
    ]

    dp_args, dp_kwargs, group_indices = controller._prepare_ppo_dispatch(rows)

    assert group_indices is not None
    assert sorted(map(len, group_indices)) == [1594, 1594, 1595, 1595]
    assert [len(rank_rows) for rank_rows in dp_args[0]] == [
        len(indices) for indices in group_indices
    ]
    assert dp_kwargs["_ppo_target_batch_size"] == [1595] * 4
    assert all(len(rank_rows) <= 1595 for rank_rows in dp_args[0])


def test_controller_rejects_user_supplied_ppo_padding_target():
    """The internal padding target cannot be overridden at the RPC boundary."""

    controller = object.__new__(TrainController)
    controller.train_alloc = SimpleNamespace(
        parallel=SimpleNamespace(dp_size=2)
    )
    rows = [
        {"attention_mask": torch.ones(1, 2, dtype=torch.bool)} for _ in range(2)
    ]

    with pytest.raises(ValueError, match="reserved"):
        controller._prepare_ppo_dispatch(
            rows,
            _ppo_target_batch_size=1,
        )


def test_gate3_repartition_reproduces_failure_and_balances_dp4():
    """The real Gate 3 shape fails before repartition and is feasible after it."""

    grouped_lengths = _gate3_sequence_lengths()
    expected_group_sizes = [2227, 2098, 1756, 480]
    expected_group_tokens = [1_013_428, 948_297, 791_123, 210_924]
    expected_initial_mbs = [1090, 1031, 864, 222]

    assert [len(lengths) for lengths in grouped_lengths] == expected_group_sizes
    assert [sum(lengths) for lengths in grouped_lengths] == expected_group_tokens
    assert [
        len(ffd_allocate(lengths, capacity=1024, min_groups=1))
        for lengths in grouped_lengths
    ] == expected_initial_mbs
    with pytest.raises(
        RuntimeError,
        match="Number of values 480 is smaller than min_groups 1090",
    ):
        ffd_allocate(grouped_lengths[-1], capacity=1024, min_groups=1090)

    items = []
    option_id = 0
    for lengths in grouped_lengths:
        for length in lengths:
            items.append(
                {
                    "attention_mask": torch.ones(1, length, dtype=torch.bool),
                    "option_ids": torch.tensor([option_id]),
                }
            )
            option_id += 1

    splits, _ = _dispatch_ppo_tensors(items, dp_size=4)
    rank_lengths = [
        [int(item["attention_mask"].sum().item()) for item in split]
        for split in splits
    ]

    assert [(len(lengths), sum(lengths)) for lengths in rank_lengths] == [
        (1640, 740_879),
        (1640, 740_878),
        (1640, 740_863),
        (1641, 741_152),
    ]
    initial_counts = [
        len(ffd_allocate(lengths, capacity=1024, min_groups=1))
        for lengths in rank_lengths
    ]
    assert initial_counts == [802, 802, 802, 802]
    synchronized_count = max(initial_counts)
    # allocate_balanced_mbs_synced returns immediately when every rank's first
    # allocation already has the same count; no forced second allocation occurs.
    assert initial_counts == [synchronized_count] * 4
    assert all(len(lengths) >= synchronized_count for lengths in rank_lengths)

    dispatched_ids = [
        int(item["option_ids"].item()) for split in splits for item in split
    ]
    assert sorted(dispatched_ids) == list(range(6561))
    assert sum(sum(lengths) for lengths in rank_lengths) == 2_963_772


def test_v2_ppo_dispatch_allows_nondivisible_sequence_count():
    """The v2 PPO endpoint opts into the same uneven dispatch semantics."""

    topology = SimpleNamespace(dp_size=4)
    dispatcher = SimpleNamespace(_topology=topology)
    request = DispatchRequest(
        dispatcher,
        "/ppo/actor/update",
        uneven_ppo=True,
    )
    items = [
        {
            "attention_mask": torch.ones(1, length, dtype=torch.bool),
            "option_ids": torch.tensor([index]),
        }
        for index, length in enumerate([9, 8, 7, 6, 5])
    ]

    dp_args, _, group_indices = request._partition_inputs(
        [items], {}, group_size=1, uneven_ppo=True
    )

    assert sorted(index for group in group_indices for index in group) == list(range(5))
    assert sorted(len(split) for split in dp_args[0]) == [1, 1, 1, 2]
    with pytest.raises(ValueError, match="divisible by K"):
        request._partition_inputs([items], {}, group_size=1, uneven_ppo=False)

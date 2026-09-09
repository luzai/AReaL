from __future__ import annotations

from unittest import mock

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import reorder_and_pad_outputs
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils.data import (
    DP_VALID_TOKEN_MASK,
    MicroBatchList,
    split_padded_tensor_dict_into_mb_list,
)


def _fake_uneven_gathers(output, value, group=None):
    del group
    values = [4, 4, 4, 3] if value == 3 else [4, 4, 4, 4]
    output[:] = values


def test_uneven_dp_batch_adds_zero_loss_multimodal_row() -> None:
    pixel_values = torch.arange(2, dtype=torch.float32)
    original = {
        "input_ids": torch.arange(12, dtype=torch.long).view(3, 4),
        "attention_mask": torch.ones((3, 4), dtype=torch.bool),
        "loss_mask": torch.ones((3, 4), dtype=torch.bool),
        "pacman_action_mask_bits": torch.arange(12, dtype=torch.uint8).view(3, 4),
        "multi_modal_input": [
            {"pixel_values": pixel_values + index} for index in range(3)
        ],
    }

    with (
        mock.patch("areal.utils.data.dist.is_initialized", return_value=True),
        mock.patch("areal.utils.data.dist.get_world_size", return_value=4),
        mock.patch("areal.utils.data.dist.get_rank", return_value=3),
        mock.patch(
            "areal.utils.data.dist.all_gather_object",
            side_effect=_fake_uneven_gathers,
        ),
    ):
        result = split_padded_tensor_dict_into_mb_list(
            original,
            MicroBatchSpec(max_tokens_per_mb=4),
            group=object(),
            pad_uneven_dp_sequences=True,
        )

    assert result.valid_batch_size == 3
    assert result.padded_batch_size == 4
    assert result.dp_padding_source_indices == [2]
    assert len(result.mbs) == 4
    assert result.data["input_ids"].shape == (4, 4)
    torch.testing.assert_close(result.data["input_ids"][3], original["input_ids"][2])
    torch.testing.assert_close(
        result.data["pacman_action_mask_bits"][3],
        original["pacman_action_mask_bits"][2],
    )
    assert result.data["loss_mask"][3].count_nonzero() == 0
    assert result.data[DP_VALID_TOKEN_MASK][3].count_nonzero() == 0
    assert len(result.data["multi_modal_input"]) == 4
    assert result.data["multi_modal_input"][3] is not original["multi_modal_input"][2]
    assert (
        result.data["multi_modal_input"][3]["pixel_values"]
        is original["multi_modal_input"][2]["pixel_values"]
    )
    assert original["input_ids"].shape == (3, 4)
    assert len(original["multi_modal_input"]) == 3


def test_nested_padding_preserves_existing_invalid_tokens() -> None:
    data = {
        "input_ids": torch.arange(12, dtype=torch.long).view(3, 4),
        "attention_mask": torch.ones((3, 4), dtype=torch.bool),
        "loss_mask": torch.ones((3, 4), dtype=torch.bool),
        DP_VALID_TOKEN_MASK: torch.tensor(
            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool
        ),
    }
    with (
        mock.patch("areal.utils.data.dist.is_initialized", return_value=True),
        mock.patch("areal.utils.data.dist.get_world_size", return_value=4),
        mock.patch("areal.utils.data.dist.get_rank", return_value=3),
        mock.patch(
            "areal.utils.data.dist.all_gather_object",
            side_effect=_fake_uneven_gathers,
        ),
    ):
        result = split_padded_tensor_dict_into_mb_list(
            data,
            MicroBatchSpec(max_tokens_per_mb=4),
            group=object(),
            pad_uneven_dp_sequences=True,
        )
    assert result.data[DP_VALID_TOKEN_MASK][2:].count_nonzero() == 0
    denominator = infer_token_denominator(
        result.data, fallback=result.data["loss_mask"]
    )
    torch.testing.assert_close(denominator, result.data[DP_VALID_TOKEN_MASK])


def test_forward_outputs_drop_appended_dp_rows_after_inverse_reorder() -> None:
    mb_list = MicroBatchList(
        data={},
        mb_spec=MicroBatchSpec(),
        mbs=[],
        group_lens=[],
        forward_indices=[0, 1, 2, 3],
        backward_indices=[0, 1, 2, 3],
        valid_batch_size=3,
        padded_batch_size=4,
        dp_padding_source_indices=[2],
    )
    outputs = [
        torch.tensor([10.0, 11.0]),
        torch.tensor([20.0, 21.0, 22.0]),
        torch.tensor([30.0]),
        torch.tensor([99.0]),
    ]
    result = reorder_and_pad_outputs(outputs, [2, 3, 1], mb_list)
    expected = torch.tensor([[10.0, 11.0, 0.0], [20.0, 21.0, 22.0], [30.0, 0.0, 0.0]])
    torch.testing.assert_close(result, expected)


def test_microbatch_to_preserves_dp_padding_metadata() -> None:
    mb_list = MicroBatchList(
        data={"x": torch.ones(1)},
        mb_spec=MicroBatchSpec(),
        mbs=[{"x": torch.ones(1)}],
        group_lens=[1],
        valid_batch_size=3,
        padded_batch_size=4,
        dp_padding_source_indices=[2],
    )
    moved = mb_list.to(dtype=torch.float64)
    assert moved.valid_batch_size == 3
    assert moved.padded_batch_size == 4
    assert moved.dp_padding_source_indices == [2]
    assert moved.data["x"].dtype == torch.float64


def test_padding_is_disabled_by_default_for_other_algorithms() -> None:
    data = {
        "input_ids": torch.arange(12, dtype=torch.long).view(3, 4),
        "attention_mask": torch.ones((3, 4), dtype=torch.bool),
    }
    result = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(max_tokens_per_mb=4)
    )
    assert result.valid_batch_size == 3
    assert result.padded_batch_size == 3
    assert DP_VALID_TOKEN_MASK not in result.data

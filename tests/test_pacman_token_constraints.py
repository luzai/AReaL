from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils.data import (
    MicroBatchItem,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)


def make_engine() -> FSDPEngine:
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.parallel_helper = SimpleNamespace(tp_size=1)
    engine.config = SimpleNamespace(temperature=1.0)
    engine.tokenizer = SimpleNamespace(
        encode=lambda token, **_: [[1], [2], [3], [4]]["UDLR".index(token)]
    )
    return engine


def test_variable_option_token_constraint_matches_manual_logsumexp() -> None:
    engine = make_engine()
    logits = torch.tensor(
        [
            [0.0, 0.1, 0.2, 1.0, 0.4, 2.0],
            [0.0, 0.1, 0.2, 0.3, 3.0, 0.5],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([3, 4, 0])
    initial = torch.tensor([-9.0, -8.0, -7.0])
    entropy = torch.tensor([9.0, 8.0, 7.0])
    # Stored on sampled tokens. IDs use +1 encoding; zero is inactive/padding.
    allowed = torch.tensor(
        [
            [0, 0],
            [3 + 1, 5 + 1],
            [4 + 1, 0],
        ]
    )

    actual, actual_entropy = engine._apply_pacman_action_mask(
        logits, labels, None, allowed, initial, entropy
    )

    expected_first = logits[0, 3] - torch.logsumexp(logits[0, [3, 5]], dim=0)
    torch.testing.assert_close(actual[0], expected_first)
    torch.testing.assert_close(actual[1], torch.tensor(0.0))
    torch.testing.assert_close(actual[2], initial[2])
    probabilities = torch.softmax(logits[0, [3, 5]], dim=0)
    expected_entropy = -(probabilities * probabilities.log()).sum()
    torch.testing.assert_close(actual_entropy[0], expected_entropy)
    torch.testing.assert_close(actual_entropy[1], torch.tensor(0.0))
    torch.testing.assert_close(actual_entropy[2], entropy[2])

    (actual.sum() + actual_entropy.sum()).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 3] != 0
    assert logits.grad[0, 5] != 0
    assert torch.count_nonzero(logits.grad[0, [0, 1, 2, 4]]) == 0


@pytest.mark.parametrize(
    ("allowed", "match"),
    [
        (torch.tensor([[0, 0], [4, 4]]), "duplicate"),
        (torch.tensor([[0, 0], [99, 0]]), "out-of-vocabulary"),
        (torch.tensor([[0, 0], [5, 0]]), "absent"),
    ],
)
def test_variable_option_token_constraint_rejects_invalid_support(
    allowed: torch.Tensor, match: str
) -> None:
    engine = make_engine()
    logits = torch.zeros((2, 8))
    labels = torch.tensor([3, 0])
    with pytest.raises(RuntimeError, match=match):
        engine._apply_pacman_action_mask(
            logits,
            labels,
            None,
            allowed,
            torch.zeros(2),
        )


def test_legacy_atomic_action_bits_remain_supported() -> None:
    engine = make_engine()
    logits = torch.zeros((2, 8))
    logits[0, 1] = 1.0
    logits[0, 4] = 2.0
    labels = torch.tensor([1, 0])
    result, _ = engine._apply_pacman_action_mask(
        logits,
        labels,
        torch.tensor([0, 0b1001], dtype=torch.uint8),
        None,
        torch.zeros(2),
    )
    expected = logits[0, 1] - torch.logsumexp(logits[0, [1, 4]], dim=0)
    torch.testing.assert_close(result[0], expected)


def test_option_token_constraint_rejects_tensor_parallelism() -> None:
    engine = make_engine()
    engine.parallel_helper.tp_size = 2
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        engine._apply_pacman_action_mask(
            torch.zeros((2, 8)),
            torch.tensor([3, 0]),
            None,
            torch.tensor([[0, 0], [4, 0]]),
            torch.zeros(2),
        )


def test_option_token_constraint_rejects_sequence_parallelism() -> None:
    engine = make_engine()
    engine.parallel_helper.sp_size = 2
    engine.is_vision_model = False
    item = MicroBatchItem(
        orig_mb={
            "input_ids": torch.tensor([[1, 2]]),
            "pacman_allowed_token_ids": torch.tensor([[[0], [4]]]),
        },
        padded_mb={"input_ids": torch.tensor([[1, 2]])},
        padding_length=0,
        old_cu_seqlens=None,
        padded_to_length=2,
    )
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        engine._prepare_mb_inputs(item)


def test_token_constraint_trailing_dimension_survives_microbatch_pipeline() -> None:
    data = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "pacman_allowed_token_ids": torch.tensor(
            [
                [[0, 0], [4, 6], [5, 0]],
                [[0, 0], [7, 8], [0, 0]],
            ],
            dtype=torch.long,
        ),
    }
    mbs = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(n_mbs=1, max_tokens_per_mb=16),
    )
    assert mbs.mbs[0]["pacman_allowed_token_ids"].shape == (2, 3, 2)
    mbs.mbs[0] = pack_tensor_dict(mbs.mbs[0])
    assert mbs.mbs[0]["pacman_allowed_token_ids"].shape == (5, 2)
    pad_mb_list(mbs)
    assert mbs.padded_mbs[0]["pacman_allowed_token_ids"].shape[-1] == 2
    unsqueeze_mb_list(mbs)
    assert mbs.padded_mbs[0]["pacman_allowed_token_ids"].ndim == 3

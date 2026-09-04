from types import SimpleNamespace

import pytest
import torch

from areal.trainer.rl_trainer import _compute_logp_in_rpc_chunks


def test_logp_rpc_chunks_preserve_order_and_dp_alignment(monkeypatch) -> None:
    calls = []

    class Engine:
        def compute_logp(self, batch):
            calls.append([item["id"] for item in batch])
            return [f"logp-{item['id']}" for item in batch]

    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    batch = [{"id": index} for index in range(12)]
    result = _compute_logp_in_rpc_chunks(Engine(), batch, role="ref", dp_size=4)

    assert calls == [list(range(4)), list(range(4, 8)), list(range(8, 12))]
    assert result == [f"logp-{index}" for index in range(12)]


def test_logp_rpc_chunks_split_and_regroup_nested_samples(monkeypatch) -> None:
    calls = []

    class Engine:
        def compute_logp(self, batch):
            calls.append([int(item["id"].item()) for item in batch])
            return [
                item["id"].float().expand(-1, item["attention_mask"].shape[-1])
                for item in batch
            ]

    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    batch = [
        {
            "id": torch.arange(0, 12).view(12, 1),
            "attention_mask": torch.ones(12, 3, dtype=torch.bool),
            "multi_modal_input": [{"sample": i} for i in range(12)],
        },
        {
            "id": torch.arange(12, 24).view(12, 1),
            "attention_mask": torch.ones(12, 5, dtype=torch.bool),
            "multi_modal_input": [{"sample": i} for i in range(12, 24)],
        },
    ]
    result = _compute_logp_in_rpc_chunks(Engine(), batch, role="ref", dp_size=4)

    assert calls == [list(range(i, i + 4)) for i in range(0, 24, 4)]
    assert len(result) == 2
    assert result[0][:, 0].tolist() == list(range(12))
    assert result[0].shape == (12, 3)
    assert result[1][:, 0].tolist() == list(range(12, 24))
    assert result[1].shape == (12, 5)


def test_logp_rpc_chunks_restore_per_sample_compaction_width(monkeypatch) -> None:
    calls = []

    class Engine:
        def compute_logp(self, batch):
            calls.append([int(item["id"].item()) for item in batch])
            # Reproduce RTensor's per-trajectory compaction on the RPC path.
            return [
                item["id"]
                .float()
                .expand(-1, int(item["attention_mask"].sum(-1).max().item()))
                for item in batch
            ]

    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    batch = [
        {
            "id": torch.arange(4).view(4, 1),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                ],
                dtype=torch.bool,
            ),
        }
    ]

    result = _compute_logp_in_rpc_chunks(Engine(), batch, role="ref", dp_size=4)

    assert calls == [[0, 1, 2, 3]]
    assert result[0].shape == (4, 5)
    assert result[0].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 3.0, 0.0],
    ]


def test_logp_rpc_chunks_reject_result_with_wrong_compacted_width(
    monkeypatch,
) -> None:
    class Engine:
        def compute_logp(self, batch):
            return [torch.zeros(1, 4) for _ in batch]

    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    batch = [
        {
            "id": torch.arange(4).view(4, 1),
            "attention_mask": torch.ones(4, 3, dtype=torch.bool),
        }
    ]

    with pytest.raises(RuntimeError, match="expected compacted input width 3"):
        _compute_logp_in_rpc_chunks(Engine(), batch, role="actor", dp_size=4)


def test_logp_rpc_chunks_restore_compaction_across_group_boundary(
    monkeypatch,
) -> None:
    calls = []

    class Engine:
        def compute_logp(self, batch):
            calls.append([int(item["id"].item()) for item in batch])
            return [
                item["id"]
                .float()
                .expand(-1, int(item["attention_mask"].sum(-1).max().item()))
                for item in batch
            ]

    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    batch = [
        {
            "id": torch.arange(5).view(5, 1),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                ],
                dtype=torch.bool,
            ),
        },
        {
            "id": torch.arange(5, 8).view(3, 1),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0],
                ],
                dtype=torch.bool,
            ),
        },
    ]

    result = _compute_logp_in_rpc_chunks(Engine(), batch, role="ref", dp_size=4)

    assert calls == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [value.shape for value in result] == [(5, 5), (3, 7)]
    assert result[0][:, 0].tolist() == list(range(5))
    assert result[1][:, 0].tolist() == list(range(5, 8))
    assert result[0][0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert result[0][1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert result[1][0].tolist() == [5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0]


def test_logp_rpc_chunks_reject_different_width_result_swap(monkeypatch) -> None:
    class Engine:
        def compute_logp(self, batch):
            result = [
                item["id"]
                .float()
                .expand(-1, int(item["attention_mask"].sum(-1).max().item()))
                for item in batch
            ]
            result[0], result[1] = result[1], result[0]
            return result

    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    batch = [
        {
            "id": torch.arange(4).view(4, 1),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                ],
                dtype=torch.bool,
            ),
        }
    ]

    with pytest.raises(RuntimeError, match="expected compacted input width 2"):
        _compute_logp_in_rpc_chunks(Engine(), batch, role="ref", dp_size=4)


def test_logp_rpc_chunks_pad_final_dp_collective_and_discard_padding(
    monkeypatch,
) -> None:
    calls = []

    class Engine:
        def compute_logp(self, batch):
            calls.append([int(item["id"].item()) for item in batch])
            return [
                item["id"].float().expand(-1, item["attention_mask"].shape[-1])
                for item in batch
            ]

    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "4")
    batch = [
        {
            "id": torch.arange(0, 5).view(5, 1),
            "attention_mask": torch.ones(5, 3, dtype=torch.bool),
        },
        {
            "id": torch.arange(5, 10).view(5, 1),
            "attention_mask": torch.ones(5, 3, dtype=torch.bool),
        },
    ]

    result = _compute_logp_in_rpc_chunks(Engine(), batch, role="actor", dp_size=4)

    assert calls == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 9, 9],
    ]
    assert [value[:, 0].tolist() for value in result] == [
        list(range(5)),
        list(range(5, 10)),
    ]
    assert [value.shape for value in result] == [(5, 3), (5, 3)]


@pytest.mark.parametrize("value", ["0", "3", "6", "bad"])
def test_logp_rpc_chunks_reject_invalid_dp_shape(monkeypatch, value) -> None:
    monkeypatch.setenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", value)
    with pytest.raises(ValueError):
        _compute_logp_in_rpc_chunks(
            SimpleNamespace(compute_logp=lambda batch: batch),
            [{"id": index} for index in range(8)],
            role="actor",
            dp_size=4,
        )


def test_logp_rpc_chunks_are_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", raising=False)
    engine = SimpleNamespace(compute_logp=lambda batch: ["one"] * len(batch))
    assert _compute_logp_in_rpc_chunks(
        engine, [{"id": 0}], role="actor", dp_size=4
    ) == ["one"]

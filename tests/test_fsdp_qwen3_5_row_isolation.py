"""CPU packing regressions; no model, optimizer, or CUDA is instantiated."""

import copy
import multiprocessing
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine, _qwen3_5_row_isolated_mb_spec
from areal.utils.data import split_padded_tensor_dict_into_mb_list


def _batch(lengths):
    width = max(lengths)
    data = {
        "input_ids": torch.zeros(len(lengths), width, dtype=torch.long),
        "attention_mask": torch.zeros(len(lengths), width, dtype=torch.bool),
        "loss_mask": torch.zeros(len(lengths), width, dtype=torch.bool),
        "pacman_action_mask_bits": torch.zeros(len(lengths), width, dtype=torch.long),
        "episode_loss_weights": torch.zeros(len(lengths), width),
        "multi_modal_input": [],
    }
    for row, length in enumerate(lengths):
        data["input_ids"][row, :length] = row + 1
        data["attention_mask"][row, :length] = True
        data["loss_mask"][row, length - 1] = True
        data["pacman_action_mask_bits"][row, length - 1] = 1 << (row % 4)
        data["episode_loss_weights"][row, length - 1] = 0.25
        data["multi_modal_input"].append(
            {
                "pixel_values": torch.full((2, 4), float(row)),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            }
        )
    return data


@pytest.fixture
def cpu_group(tmp_path):
    """Use a real single-rank CPU group for the existing packing collectives."""
    if not dist.is_gloo_available():
        pytest.skip("CPU Gloo is unavailable")
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "group").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def _engine(group, model_type="qwen3_5", **spec_kwargs):
    # Exercise the real preparation method, not FSDP/DTensor internals.
    engine = object.__new__(FSDPEngine)
    engine._initialized = True
    engine._cpu_group = group
    engine.config = SimpleNamespace(
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024, **spec_kwargs),
        pad_to_maximum=False,
    )
    engine.model_config = SimpleNamespace(model_type=model_type)
    engine.parallel_helper = SimpleNamespace(sp_size=1)
    engine.enable_tree_training = False
    engine.logger = SimpleNamespace(info=lambda _: None)
    return engine


@pytest.mark.parametrize("algorithm", ["ffd", "kk"])
@pytest.mark.parametrize("rows", [1, 4, 32])
def test_qwen35_preparation_keeps_one_row_and_matching_payloads(
    cpu_group, algorithm, rows
):
    """Keep every token, image, action mask, and loss weight in its own forward."""
    lengths = [478 + 2 * (i % 17) for i in range(rows)]
    data = _batch(lengths)
    before = copy.deepcopy(data)
    engine = _engine(cpu_group, packing_algorithm=algorithm)

    result = engine._prepare_mb_list(data)

    assert len(result.mbs) == rows
    assert result.group_lens == lengths
    assert result.forward_indices == result.backward_indices == list(range(rows))
    assert engine.config.mb_spec.n_mbs == 1
    assert (
        result.mb_spec.max_tokens_per_mb
        == engine.config.mb_spec.max_tokens_per_mb
        == 1024
    )
    assert result.mb_spec.packing_algorithm == algorithm
    for row, (original, padded) in enumerate(
        zip(result.mbs, result.padded_mbs, strict=True)
    ):
        torch.testing.assert_close(
            original["cu_seqlens"],
            torch.tensor([0, lengths[row]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        for key in (
            "input_ids",
            "loss_mask",
            "pacman_action_mask_bits",
            "episode_loss_weights",
        ):
            torch.testing.assert_close(
                original[key], before[key][row, : lengths[row]], rtol=0, atol=0
            )
            torch.testing.assert_close(data[key], before[key], rtol=0, atol=0)
        torch.testing.assert_close(
            padded["pixel_values"],
            before["multi_modal_input"][row]["pixel_values"],
            rtol=0,
            atol=0,
        )
        assert padded["position_ids"].shape == padded["input_ids"].shape
    assert sum(mb["loss_mask"].sum().item() for mb in result.mbs) == rows
    assert (
        sum(mb["episode_loss_weights"].sum().item() for mb in result.mbs) == rows * 0.25
    )


def test_non_qwen35_keeps_existing_two_row_packing(cpu_group):
    """Do not change the generic FSDP packing policy."""
    engine = _engine(cpu_group, model_type="qwen3")
    result = engine._prepare_mb_list(_batch([494] * 4))
    assert len(result.mbs) == 2
    assert result.group_lens == [988, 988]
    assert result.mb_spec.n_mbs == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"granularity": 2}, "granularity"),
        ({"n_mbs": 5}, "n_mbs"),
        ({"n_mbs": 0}, "n_mbs"),
        ({"n_mbs_divisor": 3}, "n_mbs_divisor"),
        ({"n_mbs_divisor": 0}, "n_mbs_divisor"),
    ],
)
def test_invalid_spec_constraints_fail_closed(cpu_group, kwargs, match):
    """Do not silently discard existing grouping or minimum-count constraints."""
    engine = _engine(cpu_group, **kwargs)
    with pytest.raises(ValueError, match=match):
        engine._prepare_mb_list(_batch([494] * 4))


@pytest.mark.parametrize("tree_training,sp_size", [(True, 1), (False, 2)])
def test_unsupported_qwen35_paths_do_not_bypass_isolation(
    cpu_group, tree_training, sp_size
):
    """Tree and SP must reject before entering their old packed path."""
    engine = _engine(cpu_group)
    engine.enable_tree_training = tree_training
    engine.parallel_helper.sp_size = sp_size
    with pytest.raises(ValueError, match="non-tree training with SP=1"):
        engine._prepare_mb_list(_batch([494] * 4))


def test_token_limit_is_preserved_not_lowered_to_512(cpu_group):
    """A valid 1024-token row passes; 1025 tokens still fail, without truncation."""
    engine = _engine(cpu_group, n_mbs=2, n_mbs_divisor=2)
    result = engine._prepare_mb_list(_batch([478, 1024]))
    assert result.group_lens == [478, 1024]
    assert result.mb_spec.n_mbs_divisor == 2
    with pytest.raises(ValueError, match="row exceeds max_tokens_per_mb=1024"):
        engine._prepare_mb_list(_batch([478, 1025]))


@pytest.mark.parametrize("shape", [(0, 4), (2, 0), (2, 3, 4)])
def test_empty_or_malformed_batch_fails_closed(cpu_group, shape):
    """Reject invalid row layouts before entering synchronized packing."""
    with pytest.raises(ValueError, match="non-empty 2D"):
        _qwen3_5_row_isolated_mb_spec(
            torch.zeros(shape),
            MicroBatchSpec(max_tokens_per_mb=1024),
            seq_lens=None,
            group=cpu_group,
        )


def _distributed_worker(rank, init_method, scenario, results):
    """Real Gloo ranks must both accept or both reject before packing."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        rows = 3 if scenario == "unequal_rows" and rank == 1 else 2
        lengths = [494] * rows
        if scenario == "one_rank_overlong" and rank == 1:
            lengths[-1] = 1025
        spec = MicroBatchSpec(
            max_tokens_per_mb=1024,
            granularity=2 if scenario == "one_rank_invalid_spec" and rank == 1 else 1,
        )
        data = _batch(lengths)
        try:
            isolated = _qwen3_5_row_isolated_mb_spec(
                data["attention_mask"], spec, seq_lens=lengths, group=dist.group.WORLD
            )
            batches = split_padded_tensor_dict_into_mb_list(
                data, isolated, _seq_lens=lengths
            )
            assert all(mb["attention_mask"].shape[0] == 1 for mb in batches.mbs)
            results.put((rank, "ok"))
        except ValueError as error:
            results.put((rank, str(error)))
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo is unavailable")
@pytest.mark.parametrize(
    "scenario",
    ["equal_rows", "unequal_rows", "one_rank_invalid_spec", "one_rank_overlong"],
)
def test_distributed_row_guard_agrees_on_every_rank(tmp_path, scenario):
    """Bound two real CPU processes, including malformed input on only one rank."""
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    init_method = (tmp_path / "group").as_uri()
    processes = [
        context.Process(
            target=_distributed_worker, args=(rank, init_method, scenario, results)
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        reports = sorted(results.get(timeout=45) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert reports[0][1] == reports[1][1]
        if scenario == "equal_rows":
            assert reports[0][1] == "ok"
        else:
            assert "Qwen3.5 row isolation" in reports[0][1]
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()


@pytest.mark.parametrize("granularity", [1, 2])
def test_splitter_reuses_planned_lengths_without_tensor_sum_or_cpu(
    monkeypatch, granularity
):
    """Reuse exact CPU planning metadata without a second tensor reduction/copy."""
    assert not dist.is_initialized()
    lengths = [478, 490, 502, 510]
    data = _batch(lengths)
    spec = MicroBatchSpec(max_tokens_per_mb=1024, granularity=granularity)
    expected = split_padded_tensor_dict_into_mb_list(data, spec)

    def unexpected_sync(*args, **kwargs):
        raise AssertionError("length planning must not repeat tensor sum/cpu")

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "sum", unexpected_sync)
        scoped.setattr(torch.Tensor, "cpu", unexpected_sync)
        if granularity == 1:
            isolated = _qwen3_5_row_isolated_mb_spec(
                data["attention_mask"], spec, seq_lens=lengths, group=None
            )
            assert isolated.n_mbs == len(lengths)
        actual = split_padded_tensor_dict_into_mb_list(data, spec, _seq_lens=lengths)

    assert actual.forward_indices == expected.forward_indices
    assert actual.backward_indices == expected.backward_indices
    assert actual.group_lens == expected.group_lens
    assert lengths == [478, 490, 502, 510]
    for actual_mb, expected_mb in zip(actual.mbs, expected.mbs, strict=True):
        for key, value in actual_mb.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected_mb[key], rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [[478], [0, 494], [478, 1025], [478, 494.0]])
def test_invalid_planning_metadata_fails_closed(cpu_group, lengths):
    """The internal metadata interface must not silently truncate or invent rows."""
    data = _batch([478, 494])
    spec = MicroBatchSpec(max_tokens_per_mb=1024)
    with pytest.raises(ValueError, match="sequence lengths|sequence-length"):
        _qwen3_5_row_isolated_mb_spec(
            data["attention_mask"], spec, seq_lens=lengths, group=cpu_group
        )
    with pytest.raises(ValueError, match="sequence-length"):
        split_padded_tensor_dict_into_mb_list(data, spec, _seq_lens=lengths)

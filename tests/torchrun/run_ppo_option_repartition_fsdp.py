"""DP4 contract smoke test for uneven PPO dispatch and a real FSDP update.

This worker is launched with ``torchrun --nproc_per_node=4``. Nine equal-length
post-advantage PPO rows are token-balanced into per-rank counts
``[3, 2, 2, 2]``. ``FSDPPPOActor.ppo_update`` must pad the shorter ranks to
three valid-forward, zero-loss rows before synchronized packing and update.

Set ``PPO_TEST_REWARD_CONTRACT`` to either ``option_return_raw_v1`` or
``episode_return_group_v1``.  The latter also verifies that post-advantage
episode loss weights survive dispatch, micro-batch packing, distributed loss
normalization, and the optimizer step.
"""

import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3Config

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, PPOActorConfig
from areal.engine import FSDPPPOActor
from areal.infra.controller.train_controller import _dispatch_ppo_tensors
from areal.infra.platforms import current_platform

SUPPORTED_CONTRACTS = {
    "option_return_raw_v1",
    "episode_return_group_v1",
}


def _setup_distributed() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        backend="nccl",
        init_method=(
            f"tcp://{os.environ.get('MASTER_ADDR', 'localhost')}:"
            f"{os.environ.get('MASTER_PORT', '29500')}"
        ),
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(int(os.environ["LOCAL_RANK"]))


def _create_tiny_model_dir(tokenizer_source: str) -> str:
    """Create one tiny local Qwen3 config while reusing an installed tokenizer."""

    rank = dist.get_rank()
    model_dir = (
        tempfile.mkdtemp(prefix="areal_ppo_repartition_fsdp_") if rank == 0 else None
    )
    model_dirs = [model_dir]
    dist.broadcast_object_list(model_dirs, src=0)
    model_dir = model_dirs[0]
    assert isinstance(model_dir, str)

    if rank == 0:
        config = Qwen3Config(
            vocab_size=2048,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
        config.save_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(model_dir)
    dist.barrier()
    return model_dir


def main() -> None:
    _setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"expected DP4, got world_size={world_size}"

    reward_contract = os.environ.get("PPO_TEST_REWARD_CONTRACT", "option_return_raw_v1")
    assert reward_contract in SUPPORTED_CONTRACTS, reward_contract

    tokenizer_source = os.environ.get(
        "AREAL_TEST_TOKENIZER_PATH",
        "/home/h100-repro/maapacman-level1/model/Qwen3.5-9B",
    )
    assert Path(tokenizer_source).is_dir(), tokenizer_source

    model_dir = _create_tiny_model_dir(tokenizer_source)
    engine = None
    try:
        torch.manual_seed(20260813)
        sequence_length = 16
        rows = []
        for option_id in range(9):
            row = {
                "option_id": torch.tensor([option_id]),
                "input_ids": torch.full(
                    (1, sequence_length),
                    32 + option_id,
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(
                    (1, sequence_length),
                    dtype=torch.bool,
                ),
                "loss_mask": torch.ones(
                    (1, sequence_length),
                    dtype=torch.bool,
                ),
                "advantages": torch.ones(
                    (1, sequence_length),
                    dtype=torch.float32,
                ),
                "logprobs": torch.full(
                    (1, sequence_length),
                    -math.log(2048),
                    dtype=torch.float32,
                ),
                "prox_logp": torch.full(
                    (1, sequence_length),
                    -math.log(2048),
                    dtype=torch.float32,
                ),
                "rewards": torch.tensor([1.0], dtype=torch.float32),
                "kl_rewards": torch.zeros(
                    (1, sequence_length),
                    dtype=torch.float32,
                ),
                "tot_rewards": torch.ones(
                    (1, sequence_length),
                    dtype=torch.float32,
                ),
            }
            if reward_contract == "episode_return_group_v1":
                row["episode_loss_weights"] = torch.full(
                    (1, sequence_length),
                    1.0 / sequence_length,
                    dtype=torch.float32,
                )
            rows.append(row)
        splits, indices = _dispatch_ppo_tensors(rows, dp_size=world_size)
        assert [len(split) for split in splits] == [3, 2, 2, 2]
        dispatched_indices = sorted(index for shard in indices for index in shard)
        assert dispatched_indices == list(range(9))
        local_rows = splits[rank]
        if reward_contract == "option_return_raw_v1":
            assert all("episode_loss_weights" not in row for row in local_rows)
        else:
            assert all("episode_loss_weights" in row for row in local_rows)
            for row in local_rows:
                torch.testing.assert_close(
                    row["episode_loss_weights"].sum(),
                    torch.tensor(1.0),
                )

        real_sequence_count = len(local_rows)
        config = PPOActorConfig(
            experiment_name="ppo-contract-repartition-fsdp-test",
            trial_name=reward_contract,
            path=model_dir,
            backend=f"fsdp:d{world_size}p1t1",
            dtype="bfloat16",
            optimizer_dtype="float32",
            init_from_scratch=True,
            gradient_checkpointing=False,
            # Keep this smoke test independent of host-specific cuDNN SDPA
            # execution-plan availability; attention kernels are not under test.
            attn_impl="eager",
            mb_spec=MicroBatchSpec(max_tokens_per_mb=16),
            optimizer=OptimizerConfig(
                type="adam",
                lr=1e-4,
                lr_scheduler_type="constant",
            ),
            ppo_n_minibatches=1,
            kl_ctl=0.0,
            max_new_tokens=sequence_length,
        )
        allocation = ModelAllocation.from_str(config.backend)
        engine = FSDPPPOActor(config)
        engine.create_process_group(parallel_strategy=allocation.parallel)
        engine.initialize(
            addr=None,
            ft_spec=FinetuneSpec(
                total_train_epochs=1,
                dataset_size=9,
                train_batch_size=9,
            ),
        )
        engine.train()
        recorded_updates: list[tuple[int, dict]] = []
        original_train_batch = engine.train_batch

        def _record_train_batch(input_, *args, **kwargs):
            if isinstance(input_, dict):
                sequence_count = int(input_["attention_mask"].shape[0])
            else:
                sequence_count = sum(
                    int(item["attention_mask"].shape[0]) for item in input_
                )
            update_stats = original_train_batch(input_, *args, **kwargs)
            recorded_updates.append((sequence_count, update_stats))
            return update_stats

        engine.train_batch = _record_train_batch
        delayed_rank_seconds = float(
            os.environ.get("PPO_TEST_DELAYED_RANK_SECONDS", "0")
        )
        if rank == 3 and delayed_rank_seconds > 0:
            time.sleep(delayed_rank_seconds)
        engine.ppo_update(local_rows, _ppo_target_batch_size=3)

        assert len(recorded_updates) == 1, recorded_updates
        padded_sequence_count, stats = recorded_updates[0]

        assert padded_sequence_count == 3, (rank, padded_sequence_count)
        assert stats["num_micro_batches"] == 3, (rank, stats)
        assert stats["update_successful"] == 1.0, (rank, stats)
        assert math.isfinite(stats["grad_norm"]), (rank, stats)
        assert engine.optimizer is not None and engine.optimizer.state

        summaries = [None] * world_size
        dist.all_gather_object(
            summaries,
            {
                "rank": rank,
                "reward_contract": reward_contract,
                "real_sequence_count": real_sequence_count,
                "padded_sequence_count": padded_sequence_count,
                "option_ids": [int(row["option_id"].item()) for row in local_rows],
                "num_micro_batches": stats["num_micro_batches"],
                "update_successful": stats["update_successful"],
                "grad_norm": stats["grad_norm"],
            },
        )
        if rank == 0:
            print(
                f"PPO_CONTRACT_REPARTITION_FSDP_OK "
                f"contract={reward_contract} {summaries}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
            if rank == 0:
                resolved = Path(model_dir).resolve()
                tmp_root = Path(tempfile.gettempdir()).resolve()
                assert resolved.parent == tmp_root
                assert resolved.name.startswith("areal_ppo_repartition_fsdp_")
                shutil.rmtree(resolved)
            dist.barrier()
        if engine is not None:
            engine.destroy()
        elif dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

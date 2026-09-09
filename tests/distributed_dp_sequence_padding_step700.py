from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import reorder_and_pad_outputs
from areal.utils.data import DP_VALID_TOKEN_MASK, split_padded_tensor_dict_into_mb_list


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"expected four ranks, got {world_size}")

    valid_batch_size = [4, 4, 4, 3][rank]
    input_ids = (
        torch.arange(valid_batch_size * 4, dtype=torch.long).view(valid_batch_size, 4)
        + rank * 100
    )
    data = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "loss_mask": torch.ones_like(input_ids, dtype=torch.bool),
    }
    mb_list = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(max_tokens_per_mb=4),
        group=dist.group.WORLD,
        pad_uneven_dp_sequences=True,
    )
    if len(mb_list) != 4 or mb_list.padded_batch_size != 4:
        raise RuntimeError(
            f"rank {rank} did not receive four synchronized microbatches"
        )
    if mb_list.valid_batch_size != valid_batch_size:
        raise RuntimeError(f"rank {rank} valid size changed")
    if rank == 3:
        if mb_list.data["loss_mask"][3].count_nonzero() != 0:
            raise RuntimeError("rank 3 dummy has nonzero loss")
        if mb_list.data[DP_VALID_TOKEN_MASK][3].count_nonzero() != 0:
            raise RuntimeError("rank 3 dummy has nonzero stats weight")

    outputs = [mb["input_ids"].flatten().float() for mb in mb_list.mbs]
    reordered = reorder_and_pad_outputs(outputs, [4] * valid_batch_size, mb_list)
    torch.testing.assert_close(reordered, input_ids.float())

    torch.manual_seed(1234)
    model = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
    for mb in mb_list.mbs:
        values = mb["input_ids"].reshape(-1, 1).float()
        weights = mb["loss_mask"].reshape(-1, 1).float()
        loss = (model(values) * weights).sum() / 64.0
        loss.backward()
    gradient = model.module.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"rank {rank} has invalid gradient")
    gathered = [torch.empty_like(gradient) for _ in range(world_size)]
    dist.all_gather(gathered, gradient)
    for peer_gradient in gathered:
        torch.testing.assert_close(peer_gradient, gradient)

    payload = {
        "rank": rank,
        "valid_batch_size": valid_batch_size,
        "padded_batch_size": mb_list.padded_batch_size,
        "dummy_sequences": 4 - valid_batch_size,
        "gradient": gradient.detach().cpu().tolist(),
    }
    print("DP_SEQUENCE_PADDING_DISTRIBUTED_TEST " + json.dumps(payload, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

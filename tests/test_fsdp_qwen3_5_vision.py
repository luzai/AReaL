from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from areal.engine.fsdp_utils import apply_fsdp2


class Qwen3_5VisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(16, 4)
        self.block = nn.Linear(4, 4)


class SyntheticQwen3_5Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.visual = Qwen3_5VisionModel()
        self.model.embed_tokens = nn.Embedding(16, 4)


def test_apply_fsdp2_keeps_qwen3_5_vision_pos_embed_with_parent() -> None:
    model = SyntheticQwen3_5Model()
    wrapped = []

    with patch(
        "areal.engine.fsdp_utils.fully_shard",
        side_effect=lambda module, **_: wrapped.append(module),
    ):
        apply_fsdp2(
            model,
            {},
            SimpleNamespace(transformer_layer_cls_to_wrap=["Linear"]),
        )

    assert model.model.visual.pos_embed not in wrapped
    assert model.model.embed_tokens in wrapped
    assert model.model.visual.block in wrapped
    assert wrapped[-1] is model


def test_vision_block_preserves_fp32_angles_without_changing_global_policy():
    """Only vision block inputs bypass FSDP's recursive BF16 cast."""

    class Qwen3_5VisionBlock(nn.Linear):
        pass

    model = SyntheticQwen3_5Model()
    model.model.visual.block = Qwen3_5VisionBlock(4, 4)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    wrapped = {}
    with patch(
        "areal.engine.fsdp_utils.fully_shard",
        side_effect=lambda module, **kwargs: wrapped.update({module: kwargs}),
    ):
        apply_fsdp2(
            model,
            {"mp_policy": policy},
            SimpleNamespace(transformer_layer_cls_to_wrap=["Qwen3_5VisionBlock"]),
        )
    selected = wrapped[model.model.visual.block]["mp_policy"]
    assert selected.cast_forward_inputs is False
    assert selected.param_dtype == torch.bfloat16
    assert selected.reduce_dtype == torch.float32
    assert wrapped[model]["mp_policy"] is policy
    assert policy.cast_forward_inputs is True


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FSDP2")
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_real_vision_fsdp_bf16_forward_backward(tmp_path, checkpointing, offload):
    """Exercise actual FSDP2, BF16 gradients, checkpointing and CPU offload."""
    import copy

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'process-group'}", rank=0, world_size=1
    )
    try:
        cfg = Qwen3_5VisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=64,
            num_position_embeddings=16,
        )
        cfg._attn_implementation = "sdpa"
        cfg.tie_word_embeddings = False
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = Qwen3_5VisionModel(cfg).cuda().train()
        finally:
            torch.set_default_dtype(previous)
        reference = copy.deepcopy(model)
        if checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        kwargs = dict(
            mesh=init_device_mesh("cuda", (1,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        if offload:
            kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=False)
        apply_fsdp2(
            model,
            kwargs,
            SimpleNamespace(transformer_layer_cls_to_wrap=["Qwen3_5VisionBlock"]),
        )
        pixels = torch.randn(16, 12, device="cuda", dtype=torch.bfloat16)
        grid = torch.tensor([[1, 4, 4]], device="cuda")
        expected = reference(pixels, grid_thw=grid).pooler_output
        actual = model(pixels, grid_thw=grid).pooler_output
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        for (_, parameter), (_, ref_parameter) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            gradient = parameter.grad.to_local().cpu()
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(
                gradient, ref_parameter.grad.cpu(), rtol=0, atol=0
            )
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()

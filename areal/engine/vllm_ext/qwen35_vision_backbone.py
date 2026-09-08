# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 vision parity; retain FP32 rotary frequency/angle arithmetic."""

from types import SimpleNamespace
from typing import Any

import torch

from areal.engine.vllm_ext.qwen35_vision_pos import install_transformers_vision_pos

IMPLEMENTATION = "transformers-qwen35-vision-backbone-v1"


def apply_vision_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Use HF's FP32 multiply/add order, returning the input compute dtype."""
    xf = x.float()
    first, second = xf.chunk(2, dim=-1)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2).float()
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2).float()
    return (xf * cos + torch.cat((-second, first), dim=-1) * sin).to(x.dtype)


def install_transformers_vision_backbone(visual: Any) -> dict[str, str]:
    """Adapt a vLLM instance without changing parameters or cached model outputs.

    Only TORCH_SDPA is accepted until other encoder backends pass parity tests.
    Decoder attention and AReaL weight synchronization are untouched.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionModel,
        Qwen3_5VisionRotaryEmbedding,
    )

    if getattr(visual, "_areal_vision_backbone_impl", None) == IMPLEMENTATION:
        return visual._areal_vision_pos_metadata
    if str(visual.attn_backend).split(".")[-1] != "TORCH_SDPA":
        raise RuntimeError(
            "Qwen3.5 vision parity requires mm_encoder_attn_backend=TORCH_SDPA"
        )
    if not hasattr(visual.patch_embed.proj, "enable_linear"):
        raise RuntimeError("Unsupported vLLM vision patch embedding")
    if not visual.blocks:
        raise RuntimeError("Unsupported empty vision backbone")
    head_dim = visual.blocks[0].attn.hidden_size_per_attention_head
    for block in visual.blocks:
        if not hasattr(block.attn, "apply_rotary_emb"):
            raise RuntimeError("Unsupported vLLM vision rotary interface")

    metadata = dict(install_transformers_vision_pos(visual))
    # Keep this outside the module's buffers: a later module.to(BF16) must not
    # silently round the immutable frequency table. Rebuild nothing per request.
    # Match HF's normal GPU model-loading path, including where FP32 pow is
    # evaluated. CPU and GPU pow can differ by an ULP that BF16 later amplifies.
    with torch.device(visual.pos_embed.weight.device):
        rotary = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

    def rot_pos_emb(grid_thw) -> tuple[torch.Tensor, torch.Tensor]:
        device = visual.pos_embed.weight.device
        rotary.to(device=device)  # No dtype conversion; inv_freq remains FP32.
        proxy = SimpleNamespace(
            spatial_merge_size=visual.spatial_merge_size, rotary_pos_emb=rotary
        )
        # HF expects a tensor and converts it to a Python list. Keep grid on CPU
        # to avoid an unnecessary device synchronization for vLLM's list input.
        grid = torch.as_tensor(grid_thw, dtype=torch.long, device="cpu")
        frequencies = Qwen3_5VisionModel.rot_pos_emb(proxy, grid)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = angles.cos(), angles.sin()
        return cos[..., : head_dim // 2], sin[..., : head_dim // 2]

    visual.patch_embed.proj.enable_linear = False
    visual.rot_pos_emb = rot_pos_emb
    for block in visual.blocks:
        block.attn.apply_rotary_emb.forward = apply_vision_rotary
    metadata.update(
        implementation=IMPLEMENTATION,
        interpolation_implementation=metadata["implementation"],
        rotary_dtype="float32",
        patch_embedding="conv3d",
        encoder_attention="TORCH_SDPA",
        rotary_frequency_device="model_device",
    )
    visual._areal_vision_backbone_impl = IMPLEMENTATION
    visual._areal_vision_pos_metadata = metadata
    return metadata

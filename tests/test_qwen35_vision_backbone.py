# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from areal.engine.vllm_ext.qwen35_vision_backbone import (
    apply_vision_rotary,
    install_transformers_vision_backbone,
)

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[1, 6, 4], [2, 4, 4]]])
def test_backbone_rotary_matches_hf_with_gradients(dtype, grid):
    """Keep angles FP32 and match HF rotation, including gradient flow."""
    head_dim = 16
    visual = SimpleNamespace(
        attn_backend="TORCH_SDPA",
        patch_embed=SimpleNamespace(proj=SimpleNamespace(enable_linear=True)),
        pos_embed=torch.nn.Embedding(64, 16, dtype=dtype),
        num_grid_per_side=8,
        spatial_merge_size=2,
        blocks=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    hidden_size_per_attention_head=head_dim,
                    apply_rotary_emb=SimpleNamespace(),
                )
            )
        ],
    )
    metadata = install_transformers_vision_backbone(visual)
    assert install_transformers_vision_backbone(visual) == metadata
    assert not visual.patch_embed.proj.enable_linear
    cos, sin = visual.rot_pos_emb(grid)
    assert cos.dtype == sin.dtype == torch.float32
    proxy = SimpleNamespace(
        spatial_merge_size=2,
        rotary_pos_emb=hf.Qwen3_5VisionRotaryEmbedding(head_dim // 2),
    )
    freq = hf.Qwen3_5VisionModel.rot_pos_emb(proxy, torch.tensor(grid))
    angles = torch.cat((freq, freq), dim=-1)
    torch.testing.assert_close(torch.cat((cos, cos), -1), angles.cos(), rtol=0, atol=0)
    torch.testing.assert_close(torch.cat((sin, sin), -1), angles.sin(), rtol=0, atol=0)
    qk = torch.randn(2, len(cos), 2, head_dim, dtype=dtype, requires_grad=True)
    expected = torch.stack(
        hf.apply_rotary_pos_emb_vision(qk[0], qk[1], angles.cos(), angles.sin())
    )
    actual = apply_vision_rotary(qk, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(qk.grad).all()


def test_backbone_rejects_unverified_backend():
    """Do not silently claim parity for an unvalidated attention backend."""
    with pytest.raises(RuntimeError, match="TORCH_SDPA"):
        install_transformers_vision_backbone(SimpleNamespace(attn_backend="FLASH_ATTN"))

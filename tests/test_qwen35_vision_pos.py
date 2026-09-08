# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from areal.engine.vllm_ext.qwen35_vision_pos import (
    IMPLEMENTATION,
    install_transformers_vision_pos,
)

Qwen3_5VisionModel = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5"
).Qwen3_5VisionModel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[1, 6, 4], [2, 4, 4]]])
def test_interpolation_matches_transformers_and_live_weights(dtype, grid):
    """Match the actual HF reference exactly, including after weight changes."""
    torch.manual_seed(7)
    visual = SimpleNamespace(
        pos_embed=torch.nn.Embedding(64, 16, dtype=dtype),
        num_grid_per_side=8,
        spatial_merge_size=2,
    )
    metadata = install_transformers_vision_pos(visual)
    assert metadata["implementation"] == IMPLEMENTATION
    assert len(metadata["reference_sha256"]) == 64
    assert install_transformers_vision_pos(visual) == metadata
    proxy = SimpleNamespace(
        pos_embed=visual.pos_embed,
        num_grid_per_side=8,
        config=SimpleNamespace(spatial_merge_size=2),
    )
    for delta in (0.0, 0.25):
        with torch.no_grad():
            visual.pos_embed.weight.add_(delta)
        expected = Qwen3_5VisionModel.fast_pos_embed_interpolate(
            proxy, torch.tensor(grid)
        )
        actual = visual.fast_pos_embed_interpolate(grid)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_incompatible_vision_fails_explicitly():
    """Do not silently enable a partial adapter on an unsupported structure."""
    with pytest.raises(RuntimeError, match="Incompatible"):
        install_transformers_vision_pos(SimpleNamespace())

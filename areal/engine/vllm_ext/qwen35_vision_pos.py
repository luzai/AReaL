# SPDX-License-Identifier: Apache-2.0
"""Opt-in Transformers-compatible Qwen3.5 vision position interpolation."""

import hashlib
import inspect
from types import SimpleNamespace

import torch

IMPLEMENTATION = "transformers-qwen35-vision-pos-v1"


def install_transformers_vision_pos(visual) -> dict[str, str]:
    """Adapt one vision instance; keep live weights and the HF arithmetic order.

    No output caching: weights may change during RL. Imports stay lazy so the
    default vLLM path does not import or modify this compatibility implementation.
    """
    import transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    if getattr(visual, "_areal_vision_pos_impl", None) == IMPLEMENTATION:
        return visual._areal_vision_pos_metadata
    reference = Qwen3_5VisionModel.fast_pos_embed_interpolate
    for name in ("pos_embed", "num_grid_per_side", "spatial_merge_size"):
        if not hasattr(visual, name):
            raise RuntimeError(f"Incompatible Qwen3.5 vision implementation: {name}")

    def interpolate(grid_thw):
        # Resolve the live embedding on every call, including after weight reload.
        proxy = SimpleNamespace(
            pos_embed=visual.pos_embed,
            num_grid_per_side=visual.num_grid_per_side,
            config=SimpleNamespace(spatial_merge_size=visual.spatial_merge_size),
        )
        grid = torch.as_tensor(
            grid_thw, dtype=torch.long, device=visual.pos_embed.weight.device
        )
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
            raise ValueError("grid_thw must have shape [images, 3]")
        return reference(proxy, grid)

    metadata = {
        "implementation": IMPLEMENTATION,
        "transformers_version": transformers.__version__,
        "reference_sha256": hashlib.sha256(
            inspect.getsource(reference).encode()
        ).hexdigest(),
    }
    visual.fast_pos_embed_interpolate = interpolate
    visual._areal_vision_pos_impl = IMPLEMENTATION
    visual._areal_vision_pos_metadata = metadata
    return metadata

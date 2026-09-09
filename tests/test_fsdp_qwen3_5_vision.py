from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

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

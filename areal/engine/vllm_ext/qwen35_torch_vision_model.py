# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen3.5 model adapter selected by the opt-in worker extension."""

from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from areal.engine.vllm_ext.qwen35_vision_backbone import (
    install_transformers_vision_backbone,
)

logger = init_logger("Qwen35TorchVisionPos")


class Qwen35TorchVisionModel(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.model_config.hf_config.model_type != "qwen3_5":
            raise RuntimeError(
                "Torch vision-position adapter supports dense Qwen3.5 only"
            )
        mm_config = vllm_config.model_config.multimodal_config
        if mm_config is None:
            raise RuntimeError("Vision compatibility requires multimodal configuration")
        backend = mm_config.mm_encoder_attn_backend
        if backend not in (None, AttentionBackendEnum.TORCH_SDPA):
            raise RuntimeError(
                "Vision compatibility requires TORCH_SDPA encoder attention"
            )
        mm_config.mm_encoder_attn_backend = AttentionBackendEnum.TORCH_SDPA
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        metadata = install_transformers_vision_backbone(self.visual)
        logger.info("Enabled vision-position compatibility: %s", metadata)

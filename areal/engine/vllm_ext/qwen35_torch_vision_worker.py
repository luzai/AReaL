# SPDX-License-Identifier: Apache-2.0
"""Explicit worker-extension entry point; importing opts this worker into HF parity.

Use only for dense Qwen3.5. The ordinary VLLMWorkerExtension remains unchanged.
Registration occurs before worker model loading/profile, not during requests.
"""

from vllm import ModelRegistry

from areal.engine.vllm_ext.vllm_worker_extension import VLLMWorkerExtension

ModelRegistry.register_model(
    "Qwen3_5ForConditionalGeneration",
    "areal.engine.vllm_ext.qwen35_torch_vision_model:Qwen35TorchVisionModel",
)


class Qwen35TorchVisionWorkerExtension(VLLMWorkerExtension):
    """Retain AReaL weight synchronization while selecting vision compatibility."""

    def areal_vision_pos_status(self):
        """Read-only worker-side confirmation of the actually loaded adapter."""
        model = self.model_runner.model
        metadata = getattr(model.visual, "_areal_vision_pos_metadata", None)
        if metadata is None:
            raise RuntimeError("Requested vision-position adapter was not installed")
        return {"model_class": type(model).__name__, **metadata}

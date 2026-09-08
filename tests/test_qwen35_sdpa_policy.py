"""Process-local backend policy tests; no model, CUDA context or FSDP mocks."""

import pytest
import torch

from areal.engine.fsdp_engine import _configure_qwen3_5_vision_sdpa


@pytest.fixture
def sdpa_flags():
    cuda = torch.backends.cuda
    names = ("flash", "math", "mem_efficient", "cudnn")
    previous = {name: getattr(cuda, f"{name}_sdp_enabled")() for name in names}
    try:
        yield previous
    finally:
        for name, enabled in previous.items():
            getattr(cuda, f"enable_{name}_sdp")(enabled)


def test_dense_qwen35_vision_disables_only_cudnn_and_is_idempotent(sdpa_flags):
    cuda = torch.backends.cuda
    cuda.enable_cudnn_sdp(True)
    for _ in range(2):
        assert _configure_qwen3_5_vision_sdpa(
            "qwen3_5", is_vision_model=True, device_type="cuda"
        )
        assert not cuda.cudnn_sdp_enabled()
        for name in ("flash", "math", "mem_efficient"):
            assert getattr(cuda, f"{name}_sdp_enabled")() == sdpa_flags[name]


@pytest.mark.parametrize(
    ("model_type", "is_vision_model", "device_type"),
    [
        ("llama", False, "cuda"),
        ("qwen3_vl", True, "cuda"),
        ("qwen3_5_moe", True, "cuda"),
        ("qwen3_5", False, "cuda"),
        ("qwen3_5", True, "cpu"),
        ("qwen3_5", True, "npu"),
    ],
)
def test_other_engines_preserve_backend_flags(
    sdpa_flags, model_type, is_vision_model, device_type
):
    cuda = torch.backends.cuda
    cuda.enable_cudnn_sdp(True)
    assert not _configure_qwen3_5_vision_sdpa(
        model_type, is_vision_model=is_vision_model, device_type=device_type
    )
    assert cuda.cudnn_sdp_enabled()
    for name in ("flash", "math", "mem_efficient"):
        assert getattr(cuda, f"{name}_sdp_enabled")() == sdpa_flags[name]

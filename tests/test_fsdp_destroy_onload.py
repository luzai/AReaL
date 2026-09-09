from __future__ import annotations

import ast
from pathlib import Path

_ADJACENT_SOURCE = Path(__file__).with_name("fsdp_engine.py")
_REPO_SOURCE = (
    Path(__file__).resolve().parents[1] / "areal" / "engine" / "fsdp_engine.py"
)
SOURCE = _REPO_SOURCE if _REPO_SOURCE.exists() else _ADJACENT_SOURCE


def _load_destroy_method():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPEngine"
    )
    destroy = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "destroy"
    )
    harness = ast.ClassDef(
        name="Harness",
        bases=[],
        keywords=[],
        body=[destroy],
        decorator_list=[],
    )
    ast.fix_missing_locations(harness)
    events: list[str] = []

    class _GC:
        @staticmethod
        def collect():
            events.append("gc")

    class _Platform:
        @staticmethod
        def empty_cache():
            events.append("empty_cache")

    class _Dist:
        @staticmethod
        def is_initialized():
            return False

    namespace = {"gc": _GC, "current_platform": _Platform, "dist": _Dist}
    exec(
        compile(ast.Module(body=[harness], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["Harness"], events


def test_destroy_onloads_before_releasing_an_offloaded_engine():
    harness_cls, events = _load_destroy_method()
    engine = harness_cls()
    engine.is_offload = True
    engine._initialized = True
    engine.own_global_group = False
    engine._per_layer_optim_wrapper = None
    engine.model = object()
    engine.optimizer = object()

    class _Logger:
        @staticmethod
        def info(message):
            events.append(message)

    engine.logger = _Logger()

    def _onload():
        events.append("onload")
        engine.is_offload = False

    engine.onload = _onload
    engine.destroy()

    assert events.index("onload") < events.index("gc")
    assert events.count("onload") == 1
    assert events[:3] == [
        "TMS_DESTROY_ONLOAD_AUDIT phase=before_onload",
        "onload",
        "TMS_DESTROY_ONLOAD_AUDIT phase=after_onload",
    ]
    assert not hasattr(engine, "model")
    assert not hasattr(engine, "optimizer")
    assert engine._initialized is False
    assert engine.is_offload is False


def test_destroy_does_not_onload_an_active_engine():
    harness_cls, events = _load_destroy_method()
    engine = harness_cls()
    engine.is_offload = False
    engine._initialized = True
    engine.own_global_group = False
    engine._per_layer_optim_wrapper = None
    engine.model = object()
    engine.optimizer = object()
    engine.logger = object()
    engine.onload = lambda: events.append("onload")

    engine.destroy()

    assert "onload" not in events
    assert not hasattr(engine, "model")
    assert not hasattr(engine, "optimizer")

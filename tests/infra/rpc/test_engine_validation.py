import pytest
from flask import Flask

from areal.infra.rpc.guard import engine_blueprint
from areal.infra.rpc.guard.app import GuardState
from areal.infra.rpc.guard.engine_blueprint import (
    _should_stage_ppo_payload_on_cpu,
    _should_store_rpc_result_on_cpu,
    engine_bp,
)


@pytest.fixture
def client():
    app = Flask(__name__)
    # The blueprint calls get_state(), which looks for this config key
    state = GuardState()
    app.config["guard_state"] = state
    app.register_blueprint(engine_bp)

    with app.test_client() as client:
        yield client


def test_create_engine_empty_string(client):
    """Ensure empty strings are rejected (Functional parity with old manual check)."""
    resp = client.post("/create_engine", json={"engine": "", "engine_name": "test"})
    assert resp.status_code == 400
    # Pydantic errors are returned in the 'error' key per your route logic
    assert "error" in resp.get_json()


def test_create_engine_missing_fields(client):
    """Ensure missing required fields are caught by Pydantic."""
    resp = client.post(
        "/create_engine", json={"engine_name": "test"}
    )  # missing 'engine'
    assert resp.status_code == 400


def test_call_engine_missing_method(client):
    """Ensure missing method is rejected."""
    resp = client.post("/call", json={"engine_name": "actor/0"})
    assert resp.status_code == 400


def test_set_env_invalid_json(client):
    """Ensure malformed JSON or invalid types are rejected."""
    # Sending a string where an object is expected for 'env'
    resp = client.post("/set_env", json={"env": "not-a-dict"})
    assert resp.status_code == 400


def test_pure_dp_ppo_payload_is_staged_on_cpu(monkeypatch):
    group = object()
    monkeypatch.setattr(engine_blueprint.dist, "get_world_size", lambda value: 1)

    assert _should_stage_ppo_payload_on_cpu("ppo_update", True, group)
    assert _should_stage_ppo_payload_on_cpu("safety_probe_backward", True, group)
    assert not _should_stage_ppo_payload_on_cpu("compute_logp", True, group)
    assert not _should_stage_ppo_payload_on_cpu("ppo_update", False, group)


def test_model_parallel_ppo_payload_still_broadcasts(monkeypatch):
    group = object()
    monkeypatch.setattr(engine_blueprint.dist, "get_world_size", lambda value: 2)

    assert not _should_stage_ppo_payload_on_cpu("ppo_update", True, group)


def test_only_dp_head_advantages_are_cpu_backed():
    assert _should_store_rpc_result_on_cpu(
        "compute_advantages",
        is_train_engine=True,
        is_initialized=True,
        is_data_parallel_head=True,
    )
    assert not _should_store_rpc_result_on_cpu(
        "compute_advantages",
        is_train_engine=True,
        is_initialized=True,
        is_data_parallel_head=False,
    )
    assert not _should_store_rpc_result_on_cpu(
        "compute_logp",
        is_train_engine=True,
        is_initialized=True,
        is_data_parallel_head=True,
    )

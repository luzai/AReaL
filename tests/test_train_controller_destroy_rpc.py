from __future__ import annotations

import asyncio
from types import SimpleNamespace

from areal.infra.controller.train_controller import TrainController


class _FakeScheduler:
    def __init__(self):
        self.engine_calls = []
        self.deleted = []

    async def async_call_engine(self, **kwargs):
        await asyncio.sleep(0)
        self.engine_calls.append(kwargs)

    def delete_workers(self, *, role, reverse_order=False):
        self.deleted.append((role, reverse_order))


def test_destroy_disables_payload_broadcast_for_offloaded_engines():
    scheduler = _FakeScheduler()
    controller = TrainController.__new__(TrainController)
    controller.scheduler = scheduler
    controller.workers = [SimpleNamespace(id="actor/0"), SimpleNamespace(id="actor/1")]
    controller.workers_is_dp_head = [True, True]
    controller._worker_role = "actor"
    controller._own_process_group = False

    controller.destroy()

    assert scheduler.engine_calls == [
        {
            "worker_id": "actor/0",
            "method": "destroy",
            "engine_name": "actor/0",
            "rpc_meta": {"broadcast": False},
        },
        {
            "worker_id": "actor/1",
            "method": "destroy",
            "engine_name": "actor/1",
            "rpc_meta": {"broadcast": False},
        },
    ]
    assert scheduler.deleted == [("actor", True)]
    assert controller.workers == []

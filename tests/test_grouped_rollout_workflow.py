from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from areal.infra.remote_inf_engine import GroupedRolloutWorkflow


def test_grouped_rollout_rejects_partial_semantic_group() -> None:
    child = MagicMock()
    child.arun_episode = AsyncMock(side_effect=[{"ok": 1}, None, {"ok": 3}])
    grouped = GroupedRolloutWorkflow(child, group_size=3, logger=MagicMock())

    result = asyncio.run(grouped.arun_episode(engine=MagicMock(), data={}))

    assert result is None
    assert child.arun_episode.await_count == 3

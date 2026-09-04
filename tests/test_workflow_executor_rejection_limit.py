import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.infra.workflow_executor import BatchTaskDispatcher


def make_dispatcher(
    arrivals: list[list[object | None]], *, queue_size: int = 1
) -> SimpleNamespace:
    queued = deque(arrivals)
    logger = Mock()
    wait_results = Mock(side_effect=lambda **_kwargs: queued.popleft())
    return SimpleNamespace(
        _input_cv=threading.Condition(),
        _pending_inputs=deque(),
        staleness_manager=SimpleNamespace(get_pending_limit=lambda: queue_size),
        runner=SimpleNamespace(
            max_queue_size=queue_size,
            get_input_queue_size=lambda: 0,
        ),
        enable_tracing=False,
        submit_task_input=lambda _item: None,
        wait_results=wait_results,
        logger=logger,
    )


@pytest.mark.parametrize("dynamic_bs", [False, True])
def test_rejection_limit_zero_fails_closed(
    monkeypatch: pytest.MonkeyPatch, dynamic_bs: bool
) -> None:
    monkeypatch.setenv("MAAPACMAN_MAX_REJECTED_GROUPS", "0")
    dispatcher = make_dispatcher([[None]])

    with pytest.raises(RuntimeError, match="rejection limit exceeded"):
        BatchTaskDispatcher.active_submit_and_wait(
            dispatcher,
            iter(()),
            batch_size=1,
            dynamic_bs=dynamic_bs,
        )

    message = dispatcher.logger.error.call_args.args[0]
    assert "status=rejected_limit_exceeded" in message
    assert "attempts=1 accepted=0 rejected=1" in message
    assert f"dynamic_bs={int(dynamic_bs)} batch_size=1" in message


def test_rejection_limit_checks_one_arrival_at_a_time_for_larger_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAAPACMAN_MAX_REJECTED_GROUPS", "0")
    dispatcher = make_dispatcher([[None]], queue_size=4)

    with pytest.raises(RuntimeError, match="rejection limit exceeded"):
        BatchTaskDispatcher.active_submit_and_wait(
            dispatcher,
            iter(()),
            batch_size=4,
            dynamic_bs=False,
        )

    dispatcher.wait_results.assert_called_once_with(count=1, timeout=1)
    message = dispatcher.logger.error.call_args.args[0]
    assert "attempts=1 accepted=0 rejected=1" in message
    assert "max_rejected_groups=0 dynamic_bs=0 batch_size=4" in message


def test_rejection_limit_counts_rejections_when_dynamic_bs_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAAPACMAN_MAX_REJECTED_GROUPS", "1")
    accepted = object()
    dispatcher = make_dispatcher([[None], [accepted]])

    results = BatchTaskDispatcher.active_submit_and_wait(
        dispatcher,
        iter(()),
        batch_size=1,
        dynamic_bs=False,
    )

    assert results == [accepted]
    message = dispatcher.logger.info.call_args.args[0]
    assert "status=complete" in message
    assert "attempts=2 accepted=1 rejected=1" in message
    assert "max_rejected_groups=1 dynamic_bs=0 batch_size=1" in message


@pytest.mark.parametrize("value", ["", "-1", "not-an-integer"])
def test_rejection_limit_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MAAPACMAN_MAX_REJECTED_GROUPS", value)
    dispatcher = make_dispatcher([])

    with pytest.raises(ValueError, match="must be a non-negative integer"):
        BatchTaskDispatcher.active_submit_and_wait(
            dispatcher,
            iter(()),
            batch_size=1,
        )

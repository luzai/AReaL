from types import SimpleNamespace

from areal.api.cli_args import EvaluatorConfig
from areal.utils.evaluator import Evaluator


def test_evaluator_runs_once_before_training_then_at_epoch_boundaries():
    evaluator = Evaluator(
        EvaluatorConfig(freq_epochs=1, eval_before_train=True),
        SimpleNamespace(steps_per_epoch=2),
    )
    calls = []

    evaluator.evaluate_before_train(lambda: calls.append("before"))
    evaluator.evaluate(lambda: calls.append("epoch0"), epoch=0, step=0, global_step=0)
    evaluator.evaluate(lambda: calls.append("epoch0"), epoch=0, step=1, global_step=1)
    evaluator.evaluate(lambda: calls.append("epoch1"), epoch=1, step=0, global_step=2)
    evaluator.evaluate(lambda: calls.append("epoch1"), epoch=1, step=1, global_step=3)

    assert calls == ["before", "epoch0", "epoch1"]

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from areal.api.cli_args import SaverConfig
from areal.utils.saver import Saver


class _RecordingEngine:
    def __init__(self):
        self.paths = []

    def save(self, meta):
        self.paths.append(meta.path)


def _config(tmp_path, keep_last: int | None = 2) -> SaverConfig:
    return SaverConfig(
        experiment_name="experiment",
        trial_name="trial",
        fileroot=str(tmp_path),
        keep_last=keep_last,
    )


def test_saver_config_rejects_invalid_keep_last(tmp_path):
    with pytest.raises(ValueError, match="keep_last"):
        _config(tmp_path, keep_last=0)


def test_saver_config_rejects_invalid_best_options(tmp_path):
    with pytest.raises(ValueError, match="keep_best_metric"):
        SaverConfig(
            experiment_name="experiment",
            trial_name="trial",
            fileroot=str(tmp_path),
            keep_best_metric="",
        )
    with pytest.raises(ValueError, match="keep_best_mode"):
        SaverConfig(
            experiment_name="experiment",
            trial_name="trial",
            fileroot=str(tmp_path),
            keep_best_mode="largest",
        )


def test_prune_checkpoints_keeps_two_newest_and_recovery(tmp_path):
    saver = object.__new__(Saver)
    saver.config = _config(tmp_path)
    root = Saver.get_model_save_root(
        saver.config.experiment_name,
        saver.config.trial_name,
        saver.config.fileroot,
    )

    names = [
        "epoch1epochstep0globalstep1",
        "epoch2epochstep0globalstep3",
        "epoch1epochstep1globalstep2",
        "recover_checkpoint",
        "globalstep4_hf",
        "epoch9epochstep0globalstep5.tmp",
    ]
    for name in names:
        Path(root, name).mkdir()

    saver._prune_checkpoints("default")

    remaining = {entry.name for entry in Path(root).iterdir()}
    assert remaining == {
        "epoch1epochstep1globalstep2",
        "epoch2epochstep0globalstep3",
        "recover_checkpoint",
        "globalstep4_hf",
        "epoch9epochstep0globalstep5.tmp",
    }


def test_prune_checkpoints_also_preserves_global_best(tmp_path):
    saver = object.__new__(Saver)
    saver.config = _config(tmp_path)
    saver.config.keep_best_metric = "ppo_actor/task_reward/avg"
    root = Path(
        Saver.get_model_save_root(
            saver.config.experiment_name,
            saver.config.trial_name,
            saver.config.fileroot,
        )
    )
    for step in (1, 2, 3, 4):
        (root / f"epoch{step}epochstep0globalstep{step}").mkdir()
    best_path = root / "epoch1epochstep0globalstep1"
    saver._record_best_checkpoint("default", str(best_path), 0.75)

    saver._prune_checkpoints("default")

    remaining = {entry.name for entry in root.iterdir() if entry.is_dir()}
    assert remaining == {
        "epoch1epochstep0globalstep1",
        "epoch3epochstep0globalstep3",
        "epoch4epochstep0globalstep4",
    }
    metadata = json.loads((root / "best_checkpoint.json").read_text())
    assert metadata["metric_value"] == 0.75
    assert metadata["checkpoint"] == "epoch1epochstep0globalstep1"


def test_best_metric_comparison_supports_max_and_min(tmp_path):
    saver = object.__new__(Saver)
    saver.config = _config(tmp_path)
    saver.config.keep_best_metric = "reward"
    root = Path(
        Saver.get_model_save_root(
            saver.config.experiment_name,
            saver.config.trial_name,
            saver.config.fileroot,
        )
    )
    checkpoint = root / "epoch1epochstep0globalstep1"
    checkpoint.mkdir()
    saver._record_best_checkpoint("default", str(checkpoint), 1.0)

    assert saver._is_best_metric("default", 1.1)
    assert not saver._is_best_metric("default", 0.9)

    min_saver = object.__new__(Saver)
    min_saver.config = _config(tmp_path / "min")
    min_saver.config.keep_best_metric = "loss"
    min_saver.config.keep_best_mode = "min"
    min_root = Path(
        Saver.get_model_save_root(
            min_saver.config.experiment_name,
            min_saver.config.trial_name,
            min_saver.config.fileroot,
        )
    )
    min_checkpoint = min_root / "epoch1epochstep0globalstep1"
    min_checkpoint.mkdir()
    min_saver._record_best_checkpoint("default", str(min_checkpoint), 1.0)
    assert min_saver._is_best_metric("default", 0.9)
    assert not min_saver._is_best_metric("default", 1.1)


def test_global_best_saves_outside_regular_frequency(tmp_path):
    saver = object.__new__(Saver)
    saver.config = _config(tmp_path)
    saver.config.keep_best_metric = "ppo_actor/task_reward/avg"
    saver.ft_spec = SimpleNamespace(steps_per_epoch=2)
    saver.freq_ctl = SimpleNamespace(check=lambda **_: False)
    saver._should_use_async = lambda _: False
    engine = _RecordingEngine()

    saver.save(
        engine,
        epoch=0,
        step=0,
        global_step=0,
        metrics={"ppo_actor/task_reward/avg": 0.5},
    )
    saver.save(
        engine,
        epoch=0,
        step=1,
        global_step=1,
        metrics={"ppo_actor/task_reward/avg": 0.4},
    )
    saver.save(
        engine,
        epoch=1,
        step=0,
        global_step=2,
        metrics={"ppo_actor/task_reward/avg": 0.6},
    )

    assert [Path(path).name for path in engine.paths] == [
        "epoch0epochstep0globalstep0",
        "epoch1epochstep0globalstep2",
    ]
    metadata = json.loads(
        Path(saver._best_checkpoint_metadata_path("default")).read_text()
    )
    assert metadata["metric_value"] == 0.6
    assert metadata["checkpoint"] == "epoch1epochstep0globalstep2"

from __future__ import annotations

from pathlib import Path

import pytest

from areal.api.cli_args import SaverConfig
from areal.utils.saver import Saver


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

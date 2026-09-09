# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import getpass
import json
import math
import os
import re
import shutil
from collections.abc import Mapping
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import AutoProcessor

from areal.api import FinetuneSpec, SaveLoadMeta, TrainEngine
from areal.api.cli_args import SaverConfig
from areal.infra import TrainController
from areal.utils import timeutil
from areal.utils.async_checkpoint import AsyncCheckpointManager, AsyncMode
from areal.utils.logging import getLogger

logger = getLogger("Saver")

_REGULAR_CHECKPOINT_RE = re.compile(
    r"^epoch(?P<epoch>\d+)epochstep(?P<step>\d+)globalstep(?P<global_step>\d+)$"
)
_BEST_CHECKPOINT_METADATA = "best_checkpoint.json"


class Saver:
    def __init__(self, config: SaverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )
        self._async_mode = AsyncMode(config.mode)
        self._managers: dict[str, AsyncCheckpointManager] = {}

    @staticmethod
    def get_save_root(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        path = os.path.join(
            f"{fileroot}/checkpoints/{getpass.getuser()}/{experiment_name}/{trial_name}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_root(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        epoch: int,
        step: int,
        globalstep: int,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, fileroot, name),
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_recover_checkpoint_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, fileroot, name),
            "recover_checkpoint",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _prune_checkpoints(self, name: str) -> None:
        keep_last = self.config.keep_last
        if keep_last is None:
            return

        root = Saver.get_model_save_root(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            name,
        )
        checkpoints: list[tuple[int, int, int, str]] = []
        for entry in os.scandir(root):
            if not entry.is_dir(follow_symlinks=False):
                continue
            match = _REGULAR_CHECKPOINT_RE.fullmatch(entry.name)
            if match is None:
                continue
            checkpoints.append(
                (
                    int(match.group("global_step")),
                    int(match.group("epoch")),
                    int(match.group("step")),
                    entry.path,
                )
            )

        checkpoints.sort()
        protected = {path for _, _, _, path in checkpoints[-keep_last:]}
        best = self._load_best_checkpoint(name)
        if best is not None:
            protected.add(os.path.join(root, str(best["checkpoint"])))
        for _, _, _, path in checkpoints:
            if path in protected:
                continue
            logger.info("Pruning old checkpoint: %s", path)
            shutil.rmtree(path)

    def _best_checkpoint_metadata_path(self, name: str) -> str:
        root = Saver.get_model_save_root(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            name,
        )
        return os.path.join(root, _BEST_CHECKPOINT_METADATA)

    def _load_best_checkpoint(self, name: str) -> dict | None:
        metadata_path = self._best_checkpoint_metadata_path(name)
        try:
            with open(metadata_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        if not isinstance(data, dict):
            raise ValueError(f"Invalid best checkpoint metadata: {metadata_path}")
        return data

    def _is_best_metric(self, name: str, metric_value: float) -> bool:
        best = self._load_best_checkpoint(name)
        if best is None:
            return True
        if best.get("metric") != self.config.keep_best_metric:
            raise ValueError(
                "Best-checkpoint metric changed for an existing run: "
                f"{best.get('metric')} -> {self.config.keep_best_metric}"
            )
        if best.get("mode") != self.config.keep_best_mode:
            raise ValueError(
                "Best-checkpoint mode changed for an existing run: "
                f"{best.get('mode')} -> {self.config.keep_best_mode}"
            )
        best_value = float(best["metric_value"])
        if self.config.keep_best_mode == "max":
            return metric_value > best_value
        return metric_value < best_value

    def _record_best_checkpoint(
        self,
        name: str,
        checkpoint_path: str,
        metric_value: float,
    ) -> None:
        metadata_path = self._best_checkpoint_metadata_path(name)
        tmp_path = f"{metadata_path}.tmp.{os.getpid()}"
        data = {
            "metric": self.config.keep_best_metric,
            "mode": self.config.keep_best_mode,
            "metric_value": metric_value,
            "checkpoint": os.path.basename(checkpoint_path),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, metadata_path)
        logger.info(
            "Recorded best checkpoint: %s (%s=%s)",
            checkpoint_path,
            self.config.keep_best_metric,
            metric_value,
        )

    def _complete_save(
        self,
        name: str,
        checkpoint_path: str,
        metric_value: float | None,
    ) -> None:
        if metric_value is not None:
            self._record_best_checkpoint(name, checkpoint_path, metric_value)
        self._prune_checkpoints(name)

    def state_dict(self):
        return self.freq_ctl.state_dict()

    def load_state_dict(self, state_dict):
        self.freq_ctl.load_state_dict(state_dict)

    @property
    def is_async(self) -> bool:
        if self._async_mode in (AsyncMode.ASYNC, AsyncMode.AUTO):
            # True if any manager is configured for async saves.
            return any(mgr.is_async for mgr in self._managers.values())
        return False

    def _should_use_async(self, engine: TrainEngine | TrainController) -> bool:
        """Decide whether to use async save for this engine."""
        from areal.experimental.engine.archon_engine import ArchonEngine

        if self._async_mode == AsyncMode.ASYNC:
            if not isinstance(engine, ArchonEngine):
                logger.warning(
                    "Async checkpoint only supports ArchonEngine, "
                    "got %s; falling back to sync",
                    type(engine).__name__,
                )
                return False
            return True
        if self._async_mode == AsyncMode.AUTO:
            return isinstance(engine, ArchonEngine)
        return False

    def save(
        self,
        engine: TrainEngine | TrainController,
        epoch: int,
        step: int,
        global_step: int,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
        metrics: Mapping[str, float] | None = None,
    ):
        regular_save = self.freq_ctl.check(
            epochs=int(step == self.ft_spec.steps_per_epoch - 1), steps=1
        )
        best_metric_value = None
        if self.config.keep_best_metric is not None and name == "default":
            if metrics is None or self.config.keep_best_metric not in metrics:
                raise ValueError(
                    "Configured best-checkpoint metric is missing: "
                    f"{self.config.keep_best_metric}"
                )
            metric_value = float(metrics[self.config.keep_best_metric])
            if not math.isfinite(metric_value):
                raise ValueError(
                    "Best-checkpoint metric must be finite, got "
                    f"{metric_value} for {self.config.keep_best_metric}"
                )
            if self._is_best_metric(name, metric_value):
                best_metric_value = metric_value
        if not regular_save and best_metric_value is None:
            return
        path = Saver.get_model_save_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            epoch,
            step,
            global_step,
            name,
        )

        if self._should_use_async(engine):
            self._async_save(
                engine,
                path,
                name,
                tokenizer,
                processor,
                best_metric_value,
            )
        else:
            meta = SaveLoadMeta(
                path=path,
                weight_format="hf",
                with_optim=False,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
            )
            engine.save(meta)
            self._complete_save(name, path, best_metric_value)

    def _async_save(
        self,
        engine: TrainEngine | TrainController,
        path: str,
        name: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: AutoProcessor | None,
        best_metric_value: float | None,
    ):
        """Archon async save."""
        from areal.experimental.engine.archon_engine import ArchonEngine

        assert isinstance(engine, ArchonEngine)

        mgr = self._managers.get(name)
        if mgr is None:
            mgr = AsyncCheckpointManager(AsyncMode.ASYNC)
            self._managers[name] = mgr

        from areal.experimental.engine.archon_checkpoint import save_model_to_hf

        save_model_to_hf(
            engine,
            path,
            tokenizer,
            processor,
            async_mgr=mgr,
            post_save_fn=lambda: self._complete_save(name, path, best_metric_value),
        )

    def maybe_wait_for_staging(self):
        """Wait for all engines' staging to complete. Call before ppo_update."""
        for mgr in self._managers.values():
            mgr.maybe_wait_for_staging()

    def finalize(self):
        """Training end: wait for last upload + cleanup."""
        for mgr in self._managers.values():
            mgr.finalize()
        self._managers.clear()

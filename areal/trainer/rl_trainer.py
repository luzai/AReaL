# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import json
import math
import os
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    RolloutWorkflow,
    SaveLoadMeta,
    Scheduler,
    StepInfo,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SGLangConfig,
    TrainDatasetConfig,
    ValidDatasetConfig,
    vLLMConfig,
)
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import (
    LocalScheduler,
    RayScheduler,
    RolloutController,
    SlurmScheduler,
    current_platform,
)
from areal.infra.data_service import DataController
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.rdataset import RDataset
from areal.infra.rpc.rtensor import RTensor
from areal.infra.utils.concurrent import call_maybe_async
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.environ import is_single_controller
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)

if TYPE_CHECKING:
    from datasets import Dataset

    from areal.engine import (
        FSDPPPOActor,
        FSDPPPOCritic,
        MegatronPPOActor,
        MegatronPPOCritic,
    )
    from areal.experimental.engine.archon_engine import ArchonPPOActor, ArchonPPOCritic
    from areal.trainer.ppo.actor import PPOActorController
    from areal.trainer.ppo.critic import PPOCriticController

logger = logging.getLogger("RLTrainer")

EPISODE_GRPO_ROLLOUT_FIELDS = (
    "rollout_episode_ids",
    "rollout_episode_returns",
    "rollout_episode_group_sizes",
)


def _require_episode_grpo_fields(
    batch: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    stage: str,
) -> None:
    """Fail closed when an episode-weighted batch loses reduction metadata."""
    for index, trajectory in enumerate(batch):
        missing = [field for field in fields if field not in trajectory]
        if missing:
            raise RuntimeError(
                f"episode-weighted PPO {stage} item {index} is missing: {missing}"
            )


def _parse_update_gate_global_steps(
    value: str | None,
    legacy_value: str | None = None,
) -> set[int]:
    """Parse one or more zero-based update steps used by managed-run gates."""

    raw = value if value not in (None, "") else legacy_value
    if raw in (None, ""):
        return set()
    try:
        steps = {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError(f"invalid managed update gate steps: {raw!r}") from exc
    if not steps or any(step < 0 for step in steps):
        raise ValueError(f"invalid managed update gate steps: {raw!r}")
    return steps


def _compute_logp_in_rpc_chunks(
    engine: Any,
    batch: list[dict[str, Any]],
    *,
    role: str,
    dp_size: int,
) -> list[Any]:
    """Bound controller-to-engine log-prob requests without breaking DP.

    The environment variable is intentionally opt-in for backward
    compatibility. Every request must contain a whole number of data-parallel
    groups because all ranks participate in each compute_logp collective.
    """

    raw_chunk_size = os.getenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE")
    if raw_chunk_size in (None, ""):
        return engine.compute_logp(batch)
    try:
        chunk_size = int(raw_chunk_size)
    except ValueError as exc:
        raise ValueError(
            "MAAPACMAN_LOGP_RPC_CHUNK_SIZE must be a positive integer"
        ) from exc
    if chunk_size <= 0 or chunk_size < dp_size or chunk_size % dp_size:
        raise ValueError(
            "MAAPACMAN_LOGP_RPC_CHUNK_SIZE must be a positive multiple "
            f"of dp_size={dp_size}, got {chunk_size}"
        )
    # Rollout workflows group ``n_samples`` under each dataset row. Chunking
    # the outer list alone would still send all samples in one RPC (for
    # example, 4 rows x 12 samples = 48 sequences). Localize once, split the
    # leading sample dimension, then restore the original grouped layout after
    # every bounded RPC completes.
    local_batch = RTensor.localize(batch)
    group_sizes: list[int] = []
    group_seqlens: list[int | None] = []
    flat_seqlens: list[int | None] = []
    flat_batch: list[dict[str, Any]] = []
    for trajectory in local_batch:
        attention_mask = trajectory.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim >= 1:
            group_size = int(attention_mask.shape[0])
        else:
            first_tensor = next(
                (
                    value
                    for value in trajectory.values()
                    if isinstance(value, torch.Tensor) and value.ndim >= 1
                ),
                None,
            )
            group_size = int(first_tensor.shape[0]) if first_tensor is not None else 1
        group_sizes.append(group_size)
        group_seqlens.append(
            int(attention_mask.shape[-1])
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim >= 2
            else None
        )
        for sample_index in range(group_size):
            sample: dict[str, Any] = {}
            for key, value in trajectory.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 1
                    and value.shape[0] == group_size
                ):
                    sample[key] = value[sample_index : sample_index + 1]
                elif isinstance(value, list) and len(value) == group_size:
                    sample[key] = [value[sample_index]]
                else:
                    sample[key] = value
            sample_attention_mask = sample.get("attention_mask")
            flat_seqlens.append(
                int(sample_attention_mask.sum(-1).max().item())
                if isinstance(sample_attention_mask, torch.Tensor)
                and sample_attention_mask.ndim >= 2
                else None
            )
            flat_batch.append(sample)

    flat_results: list[Any] = []
    n_chunks = (len(flat_batch) + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, len(flat_batch), chunk_size), 1):
        chunk = flat_batch[start : start + chunk_size]
        valid_size = len(chunk)
        # A workflow group can contain a variable number of decision samples,
        # so the flattened total need not be divisible by the DP world size.
        # Every rank must still enter the final collective. Repeat the last
        # read-only sample only for that RPC, then discard the padded results.
        padding = (-valid_size) % dp_size
        if padding:
            chunk = [*chunk, *([chunk[-1]] * padding)]
        logger.info(
            "MAAPACMAN_LOGP_RPC_CHUNK role=%s chunk=%s/%s size=%s "
            "valid_size=%s dp_size=%s",
            role,
            chunk_index,
            n_chunks,
            len(chunk),
            valid_size,
            dp_size,
        )
        chunk_results = RTensor.localize(engine.compute_logp(chunk))
        if len(chunk_results) != len(chunk):
            raise RuntimeError(
                f"{role} compute_logp returned {len(chunk_results)} results "
                f"for a chunk of {len(chunk)} trajectories"
            )
        flat_results.extend(chunk_results[:valid_size])

    results: list[Any] = []
    offset = 0
    for group_size, group_seqlen in zip(group_sizes, group_seqlens, strict=True):
        group = flat_results[offset : offset + group_size]
        expected_seqlens = flat_seqlens[offset : offset + group_size]
        offset += group_size
        if not group or any(value is None for value in group):
            results.append(None)
        elif group_size == 1 and not isinstance(group[0], torch.Tensor):
            results.append(group[0])
        elif all(isinstance(value, torch.Tensor) for value in group):
            # RTensor compacts each flattened trajectory independently before
            # the RPC, so results from one original group can have different
            # sequence widths. Restore its original compacted width before
            # concatenating. Never crop an oversized result: that would hide a
            # dispatch/reordering bug instead of failing closed.
            normalized_group: list[torch.Tensor] = []
            for value, expected_seqlen in zip(group, expected_seqlens, strict=True):
                if group_seqlen is not None and value.ndim >= 2:
                    result_seqlen = int(value.shape[-1])
                    if expected_seqlen is not None and result_seqlen != expected_seqlen:
                        raise RuntimeError(
                            f"{role} compute_logp returned sequence width "
                            f"{result_seqlen}, expected compacted input width "
                            f"{expected_seqlen}"
                        )
                    if result_seqlen < group_seqlen:
                        pad_shape = (
                            *value.shape[:-1],
                            group_seqlen - result_seqlen,
                        )
                        value = torch.cat((value, value.new_zeros(pad_shape)), dim=-1)
                normalized_group.append(value)
            results.append(
                normalized_group[0]
                if group_size == 1
                else torch.cat(normalized_group, dim=0)
            )
        else:
            raise TypeError(f"{role} compute_logp returned unsupported grouped results")
    if offset != len(flat_results):
        raise RuntimeError(
            f"{role} compute_logp regrouped {offset} results but received "
            f"{len(flat_results)}"
        )
    return results


def _audit_pacman_logprob_alignment(
    batch: list[dict[str, Any]],
    *,
    output_dir: str,
    global_step: int,
    max_abs_tolerance: float = 1.0e-3,
    mean_abs_tolerance: float = 1.0e-4,
    actor_reference_max_abs_tolerance: float = 1.0e-5,
    actor_reference_mean_abs_tolerance: float = 1.0e-6,
    acceptance_mode: str = "exact",
    behavior_ratio_lower: float = 0.8,
    behavior_ratio_upper: float = 1.25,
    behavior_branch_mean_abs_log_ratio: float = 0.05,
    behavior_max_filtered_fraction: float = 0.07,
    singleton_max_abs_tolerance: float = 1.0e-7,
    require_actor_reference_alignment: bool = True,
    expected_option_tokens: int | None = None,
) -> dict[str, Any]:
    """Verify rollout and proximal actor option probabilities.

    Reference probabilities are also verified when present. Reference-free
    training must explicitly disable actor/reference alignment; the rollout to
    proximal-actor checks remain mandatory in that mode.

    ``exact`` requires the inference and training backends to reproduce token
    log-probabilities numerically. ``behavior`` still reports that exact check,
    but evaluates compatibility using the same complete-option joint ratio and
    sequence mask as PPO. Branching-token differences remain diagnostics; they
    are not an additional rejection rule because token-level errors may cancel
    in the joint probability of an atomic option. Behavior mode requires one
    contiguous fixed-width option per sample so this audit matches sequence-level
    masking.
    """

    if acceptance_mode not in {"exact", "behavior"}:
        raise ValueError(
            "Pacman log-prob acceptance_mode must be 'exact' or 'behavior'"
        )
    if not (0.0 < behavior_ratio_lower <= 1.0 <= behavior_ratio_upper):
        raise ValueError(
            "Pacman behavior ratio bounds must satisfy 0 < lower <= 1 <= upper"
        )
    if behavior_branch_mean_abs_log_ratio < 0.0:
        raise ValueError("Pacman behavior mean log-ratio tolerance must be >= 0")
    if not 0.0 <= behavior_max_filtered_fraction <= 1.0:
        raise ValueError(
            "Pacman behavior filtered-fraction tolerance must be in [0, 1]"
        )

    reference_presence = ["ref_logp" in trajectory for trajectory in batch]
    if any(reference_presence) and not all(reference_presence):
        raise RuntimeError(
            "Pacman log-prob audit found inconsistent reference log-probs"
        )
    reference_available = bool(batch) and all(reference_presence)
    if require_actor_reference_alignment and not reference_available:
        raise RuntimeError(
            "Pacman log-prob audit requires reference log-probs when "
            "actor-reference alignment is enabled"
        )

    comparisons = {"rollout_actor": []}
    if reference_available:
        comparisons.update(
            {
                "rollout_reference": [],
                "actor_reference": [],
            }
        )
    records: list[dict[str, Any]] = []
    support_sizes: set[int] = set()
    for trajectory_index, trajectory in enumerate(batch):
        required = {
            "input_ids",
            "loss_mask",
            "logprobs",
            "prox_logp",
            "versions",
            "pacman_allowed_token_ids",
        }
        if reference_available:
            required.add("ref_logp")
        missing = required - trajectory.keys()
        if missing:
            raise RuntimeError(
                f"Pacman log-prob audit missing fields: {sorted(missing)}"
            )
        local = RTensor.localize({key: trajectory[key] for key in required})
        input_ids = local["input_ids"].detach().cpu()
        loss_mask = torch.roll(local["loss_mask"].detach().cpu().bool(), -1, dims=-1)
        rollout = torch.roll(local["logprobs"].detach().cpu().float(), -1, dims=-1)
        versions = torch.roll(local["versions"].detach().cpu(), -1, dims=-1)
        actor = local["prox_logp"].detach().cpu().float()
        reference = (
            local["ref_logp"].detach().cpu().float() if reference_available else None
        )
        sampled_ids = torch.roll(input_ids, -1, dims=-1)
        supports = torch.roll(
            local["pacman_allowed_token_ids"].detach().cpu(),
            -1,
            dims=-2,
        )
        expected_shape = rollout.shape
        shapes = (
            actor.shape,
            loss_mask.shape,
            sampled_ids.shape,
            versions.shape,
            supports.shape[:-1],
        )
        if any(shape != expected_shape for shape in shapes) or (
            reference is not None and reference.shape != expected_shape
        ):
            raise RuntimeError("Pacman log-prob audit tensor shapes disagree")
        support_active = supports.ne(0).any(dim=-1)
        if not torch.equal(support_active, loss_mask):
            raise RuntimeError(
                "Pacman option supports do not match the generated-token mask"
            )
        for batch_index, token_index in loss_mask.nonzero().tolist():
            encoded_support = supports[batch_index, token_index]
            support = [
                int(value) - 1 for value in encoded_support.tolist() if int(value) != 0
            ]
            sampled = int(sampled_ids[batch_index, token_index])
            if sampled not in support or len(support) != len(set(support)):
                raise RuntimeError(
                    "sampled token is absent from its unique rollout support"
                )
            values = {
                "rollout": float(rollout[batch_index, token_index]),
                "actor": float(actor[batch_index, token_index]),
            }
            if reference is not None:
                values["reference"] = float(reference[batch_index, token_index])
            if not bool(torch.isfinite(torch.tensor(list(values.values()))).all()):
                raise RuntimeError("Pacman log-prob audit found NaN or Inf")
            comparisons["rollout_actor"].append(values["actor"] - values["rollout"])
            if reference is not None:
                comparisons["rollout_reference"].append(
                    values["reference"] - values["rollout"]
                )
                comparisons["actor_reference"].append(
                    values["actor"] - values["reference"]
                )
            support_sizes.add(len(support))
            records.append(
                {
                    "trajectory_index": trajectory_index,
                    "sample_index": batch_index,
                    "token_index": token_index,
                    "sampled_token_id": sampled,
                    "allowed_token_ids": support,
                    "support_size": len(support),
                    "version": int(versions[batch_index, token_index]),
                    **values,
                }
            )
    if not records:
        raise RuntimeError("Pacman log-prob audit found no constrained tokens")
    if not any(size > 1 for size in support_sizes):
        raise RuntimeError("Pacman log-prob audit requires branching supports")
    metrics = {}
    exact_aligned = True
    for name, deltas in comparisons.items():
        absolute = [abs(value) for value in deltas]
        ratios = [math.exp(value) for value in deltas]
        relative = [abs(math.expm1(value)) for value in deltas]
        max_abs = max(absolute)
        mean_abs = sum(absolute) / len(absolute)
        metrics[name] = {
            "signed_mean_log_delta": sum(deltas) / len(deltas),
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
            "min_importance_ratio": min(ratios),
            "max_importance_ratio": max(ratios),
            "max_abs_relative_probability_delta": max(relative),
            "mean_abs_relative_probability_delta": (sum(relative) / len(relative)),
        }
        pair_max_tolerance = (
            actor_reference_max_abs_tolerance
            if name == "actor_reference"
            else max_abs_tolerance
        )
        pair_mean_tolerance = (
            actor_reference_mean_abs_tolerance
            if name == "actor_reference"
            else mean_abs_tolerance
        )
        metrics[name]["max_abs_tolerance"] = pair_max_tolerance
        metrics[name]["mean_abs_tolerance"] = pair_mean_tolerance
        exact_aligned = exact_aligned and max_abs <= pair_max_tolerance
        exact_aligned = exact_aligned and mean_abs <= pair_mean_tolerance

    branching_records = [record for record in records if record["support_size"] > 1]
    singleton_records = [record for record in records if record["support_size"] == 1]
    branching_log_ratios = [
        record["actor"] - record["rollout"] for record in branching_records
    ]
    branching_ratios = [math.exp(value) for value in branching_log_ratios]
    would_filter_records = [
        record
        for record in records
        if not behavior_ratio_lower
        <= math.exp(record["actor"] - record["rollout"])
        <= behavior_ratio_upper
    ]
    would_filter_fraction = len(would_filter_records) / len(records)
    kept_branching_ratios = [
        math.exp(record["actor"] - record["rollout"])
        for record in branching_records
        if behavior_ratio_lower
        <= math.exp(record["actor"] - record["rollout"])
        <= behavior_ratio_upper
    ]
    singleton_abs_deltas = [
        abs(record["actor"] - record["rollout"]) for record in singleton_records
    ]
    probability_roles = (
        ("rollout", "actor", "reference")
        if reference_available
        else ("rollout", "actor")
    )
    singleton_abs_logps = [
        abs(record[role]) for record in singleton_records for role in probability_roles
    ]
    transitivity_residuals = (
        [
            abs(
                (record["actor"] - record["rollout"])
                - (record["reference"] - record["rollout"])
                - (record["actor"] - record["reference"])
            )
            for record in records
        ]
        if reference_available
        else []
    )
    option_records = []
    for record in sorted(
        records,
        key=lambda item: (
            item["trajectory_index"],
            item["sample_index"],
            item["token_index"],
        ),
    ):
        if (
            not option_records
            or option_records[-1]["trajectory_index"] != record["trajectory_index"]
            or option_records[-1]["sample_index"] != record["sample_index"]
            or option_records[-1]["end_token_index"] + 1 != record["token_index"]
        ):
            option_records.append(
                {
                    "trajectory_index": record["trajectory_index"],
                    "sample_index": record["sample_index"],
                    "start_token_index": record["token_index"],
                    "end_token_index": record["token_index"],
                    "support_sizes": [],
                    "joint_rollout_logp": 0.0,
                    "joint_actor_logp": 0.0,
                }
            )
            if reference_available:
                option_records[-1]["joint_reference_logp"] = 0.0
        option = option_records[-1]
        option["end_token_index"] = record["token_index"]
        option["support_sizes"].append(record["support_size"])
        for role in probability_roles:
            option[f"joint_{role}_logp"] += record[role]
    for option in option_records:
        option["token_count"] = len(option["support_sizes"])
        log_ratio = option["joint_actor_logp"] - option["joint_rollout_logp"]
        option["actor_rollout_log_ratio"] = log_ratio
        option["behavior_importance_ratio"] = math.exp(log_ratio)
        for role in probability_roles:
            option[f"joint_{role}_probability"] = math.exp(option[f"joint_{role}_logp"])
    sample_keys = {
        (record["trajectory_index"], record["sample_index"]) for record in records
    }
    option_keys = {
        (option["trajectory_index"], option["sample_index"])
        for option in option_records
    }
    if len(option_records) != len(sample_keys) or option_keys != sample_keys:
        raise RuntimeError(
            "Pacman behavior audit requires exactly one contiguous option per sample"
        )
    if expected_option_tokens is not None and any(
        option["token_count"] != expected_option_tokens for option in option_records
    ):
        raise RuntimeError(
            "Pacman behavior audit found an unexpected option token count"
        )
    option_behavior_ratios = [
        option["behavior_importance_ratio"] for option in option_records
    ]
    would_filter_options = [
        option
        for option in option_records
        if not behavior_ratio_lower
        <= option["behavior_importance_ratio"]
        <= behavior_ratio_upper
    ]
    would_filter_option_fraction = len(would_filter_options) / len(option_records)
    would_filter_option_keys = {
        (option["trajectory_index"], option["sample_index"])
        for option in would_filter_options
    }
    kept_option_branching_log_ratios = [
        record["actor"] - record["rollout"]
        for record in branching_records
        if (record["trajectory_index"], record["sample_index"])
        not in would_filter_option_keys
    ]
    kept_option_ratios = [
        option["behavior_importance_ratio"]
        for option in option_records
        if behavior_ratio_lower
        <= option["behavior_importance_ratio"]
        <= behavior_ratio_upper
    ]
    actor_reference_aligned = (
        metrics["actor_reference"]["max_abs_delta"] <= actor_reference_max_abs_tolerance
        and metrics["actor_reference"]["mean_abs_delta"]
        <= actor_reference_mean_abs_tolerance
        if reference_available
        else None
    )
    behavior_metrics = {
        "branching_tokens": len(branching_records),
        "singleton_tokens": len(singleton_records),
        "branching_mean_abs_log_ratio": sum(
            abs(value) for value in branching_log_ratios
        )
        / len(branching_log_ratios),
        "kept_option_branching_tokens": len(kept_option_branching_log_ratios),
        "kept_option_branching_mean_abs_log_ratio": (
            sum(abs(value) for value in kept_option_branching_log_ratios)
            / len(kept_option_branching_log_ratios)
            if kept_option_branching_log_ratios
            else None
        ),
        "kept_option_branching_max_abs_log_ratio": (
            max(abs(value) for value in kept_option_branching_log_ratios)
            if kept_option_branching_log_ratios
            else None
        ),
        "branching_min_importance_ratio": min(branching_ratios),
        "branching_max_importance_ratio": max(branching_ratios),
        "kept_branching_min_importance_ratio": (
            min(kept_branching_ratios) if kept_branching_ratios else None
        ),
        "kept_branching_max_importance_ratio": (
            max(kept_branching_ratios) if kept_branching_ratios else None
        ),
        "would_filter_token_count": len(would_filter_records),
        "would_filter_fraction": would_filter_fraction,
        "would_filter_option_count": len(would_filter_options),
        "would_filter_option_fraction": would_filter_option_fraction,
        "kept_option_min_importance_ratio": (
            min(kept_option_ratios) if kept_option_ratios else None
        ),
        "kept_option_max_importance_ratio": (
            max(kept_option_ratios) if kept_option_ratios else None
        ),
        "max_filtered_fraction": behavior_max_filtered_fraction,
        "singleton_max_abs_delta": (
            max(singleton_abs_deltas) if singleton_abs_deltas else 0.0
        ),
        "singleton_max_abs_logp": (
            max(singleton_abs_logps) if singleton_abs_logps else 0.0
        ),
        "transitivity_max_abs_residual": (
            max(transitivity_residuals) if transitivity_residuals else None
        ),
        "option_count": len(option_records),
        "option_min_importance_ratio": min(option_behavior_ratios),
        "option_max_importance_ratio": max(option_behavior_ratios),
        "ratio_lower_bound": behavior_ratio_lower,
        "ratio_upper_bound": behavior_ratio_upper,
        "branching_mean_abs_log_ratio_tolerance": (behavior_branch_mean_abs_log_ratio),
        "branching_mean_abs_log_ratio_within_diagnostic_tolerance": (
            sum(abs(value) for value in branching_log_ratios)
            / len(branching_log_ratios)
            <= behavior_branch_mean_abs_log_ratio
        ),
        "singleton_max_abs_tolerance": singleton_max_abs_tolerance,
    }
    behavior_compatible = (
        (not require_actor_reference_alignment or bool(actor_reference_aligned))
        and bool(kept_option_ratios)
        and would_filter_option_fraction <= behavior_max_filtered_fraction
        and behavior_metrics["singleton_max_abs_logp"] <= singleton_max_abs_tolerance
        and (
            behavior_metrics["transitivity_max_abs_residual"] is None
            or behavior_metrics["transitivity_max_abs_residual"] <= 1.0e-7
        )
    )
    accepted = exact_aligned if acceptance_mode == "exact" else behavior_compatible
    version_values = sorted({record["version"] for record in records})
    if len(version_values) != 1 or version_values[0] < 0:
        raise RuntimeError(
            f"Pacman log-prob audit found inconsistent versions: {version_values}"
        )
    report = {
        "accepted": accepted,
        "acceptance_mode": acceptance_mode,
        "reference_available": reference_available,
        "require_actor_reference_alignment": require_actor_reference_alignment,
        "actor_reference_aligned": actor_reference_aligned,
        "aligned": exact_aligned,
        "behavior_compatible": behavior_compatible,
        "global_step": global_step,
        "tokens": len(records),
        "support_sizes": sorted(support_sizes),
        "versions": version_values,
        "expected_option_tokens": expected_option_tokens,
        "max_abs_tolerance": max_abs_tolerance,
        "mean_abs_tolerance": mean_abs_tolerance,
        "metrics": metrics,
        "behavior_metrics": behavior_metrics,
        "rejection_sampling_contract": {
            "level": "sequence",
            "action": "mask",
            "metric": "ratio",
            "agg": "sum",
            "lower": behavior_ratio_lower,
            "upper": behavior_ratio_upper,
        },
        "options": option_records,
        "records": records,
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"global-step-{global_step:06d}.json")
    with open(output_path, "x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if not accepted:
        raise RuntimeError(
            "Pacman log-prob alignment exceeded "
            f"{acceptance_mode} acceptance limits; see {output_path}"
        )
    logger.info(
        "MAAPACMAN_LOGPROB_ALIGNMENT_OK mode=%s exact=%s "
        "behavior_compatible=%s reference_available=%s global_step=%s "
        "tokens=%s report=%s",
        acceptance_mode,
        exact_aligned,
        behavior_compatible,
        reference_available,
        global_step,
        len(records),
        output_path,
    )
    return report


class _EmptyDataLoader:
    """Minimal dataloader for online mode that yields empty dicts.

    Compatible with ``cycle_dataloader()`` and ``len()`` expectations.
    ``steps_per_epoch`` controls how many steps constitute one epoch,
    derived from ``total_train_steps // total_train_epochs`` to ensure
    epoch-frequency-gated components (Saver, RecoverHandler) behave correctly.
    """

    def __init__(self, batch_size: int = 1, steps_per_epoch: int = 1):
        self.batch_size = batch_size
        self._steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self._steps_per_epoch

    def __iter__(self):
        while True:
            yield [{} for _ in range(self.batch_size)]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:  # noqa: ARG002
        pass


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        train_dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
    ):
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            # Set up file logging for controller process
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()
        self.data_controller: DataController | None = None
        self._train_rdataset: RDataset | None = None
        self._valid_rdataset: RDataset | None = None

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        # Parse per-engine allocations from config.
        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")
        self.rollout_alloc = ModelAllocation.from_str(
            config.rollout.backend, name="rollout"
        )
        self._should_offload_rollout = self._is_actor_rollout_colocated(config)
        self._should_offload_actor = (
            self._should_offload_rollout or config.actor.offload
        )
        self._should_offload_critic = (
            config.critic is not None and config.critic.offload
        )
        self._should_offload_ref = config.ref is not None and config.ref.offload
        self._should_offload_teacher = (
            config.teacher is not None and config.teacher.offload
        )

        # Validate config before proceeding with weight initialization
        self._validate_cfg()

        self._amend_xccl_weight_update_envvar()

        agent_cfg = config.rollout.agent
        self._online_mode = agent_cfg is not None and agent_cfg.mode == "online"

        if self._online_mode and config.valid_dataset is not None:
            raise ValueError(
                "valid_dataset must not be set when using online RL mode "
                "(agent.mode='online'). Online mode does not support "
                "validation datasets."
            )

        # -- Dataset loading --------------------------------------------------
        if not self._online_mode and train_dataset is None:
            raise ValueError(
                "train_dataset must be provided unless using online RL mode "
                "(agent.mode='online')."
            )

        # Create models: actor, critic, ref — each with its own allocation.
        self.actor = self._create_train_engine(config.actor, self.actor_alloc)
        self.critic = None
        if config.critic is not None:
            critic_alloc = ModelAllocation.from_str(
                config.critic.backend, name="critic"
            )
            self.critic = self._create_critic(config.critic, critic_alloc)
        self.ref = None
        if config.actor.kl_ctl > 0 and config.ref is not None:
            ref_alloc = ModelAllocation.from_str(config.ref.backend, name="ref")
            self.ref = self._create_train_engine(config.ref, ref_alloc)

        self.teacher = None
        self.teacher_alloc = None
        if config.teacher is not None:
            if config.teacher.engine_type == "rollout":
                self.teacher_alloc = ModelAllocation.from_str(
                    config.teacher.rollout.backend, name="teacher"
                )
            else:
                assert config.teacher.train is not None
                self.teacher_alloc = ModelAllocation.from_str(
                    self.config.teacher.train.backend, name="teacher"
                )
                logger.warning(
                    "teacher.engine_type='train' uses legacy train-engine teacher path "
                    "and is deprecated; please migrate to engine_type='rollout'."
                )

        steps_per_epoch: int | None = None
        self.train_dataloader: StatefulDataLoader | _EmptyDataLoader
        if self._online_mode:
            if config.total_train_steps is None:
                raise ValueError(
                    "total_train_steps must be set for online mode. "
                    "Both total_train_epochs and total_train_steps are needed "
                    "to compute steps_per_epoch."
                )
            steps_per_epoch = config.total_train_steps // config.total_train_epochs
            if steps_per_epoch < 1:
                raise ValueError(
                    f"total_train_steps ({config.total_train_steps}) must be >= "
                    f"total_train_epochs ({config.total_train_epochs}) so that "
                    f"steps_per_epoch >= 1."
                )
            self.train_dataloader = _EmptyDataLoader(
                batch_size=config.train_dataset.batch_size,
                steps_per_epoch=steps_per_epoch,
            )
        else:
            assert train_dataset is not None
            if is_single_controller() and isinstance(train_dataset, RDataset):
                ds_cfg = DataServiceConfig.from_dataset_config(
                    config.train_dataset, seed=config.seed
                )
                assert self.scheduler is not None
                controller = DataController(ds_cfg, self.scheduler)
                controller.initialize(
                    role="data", num_dataset_workers=ds_cfg.num_workers
                )
                self.data_controller = controller
                train_dataset.connect(
                    controller,
                    dataset_id=f"{config.experiment_name}_{config.trial_name}_train",
                    tokenizer_or_processor_path=config.tokenizer_path,
                    shuffle=config.train_dataset.shuffle,
                    drop_last=config.train_dataset.drop_last,
                )
                self._train_rdataset = train_dataset

            self.train_dataloader = self._create_dataloader(
                train_dataset,
                dataset_config=self.config.train_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        self.valid_dataloader: StatefulDataLoader | None = None
        if self.config.valid_dataset is not None and valid_dataset is not None:
            assert self.config.valid_dataset is not None
            if is_single_controller() and isinstance(valid_dataset, RDataset):
                assert self.data_controller is not None
                valid_dataset.connect(
                    self.data_controller,
                    dataset_id=f"{config.experiment_name}_{config.trial_name}_valid",
                    tokenizer_or_processor_path=config.tokenizer_path,
                    shuffle=self.config.valid_dataset.shuffle,
                    drop_last=self.config.valid_dataset.drop_last,
                )
                self._valid_rdataset = valid_dataset

            self.valid_dataloader = self._create_dataloader(
                valid_dataset,
                dataset_config=self.config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        # -- FinetuneSpec -----------------------------------------------------
        if self._online_mode:
            assert steps_per_epoch is not None
            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=steps_per_epoch * config.train_dataset.batch_size,
                train_batch_size=config.train_dataset.batch_size,
            )
        else:
            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=len(self.train_dataloader)
                * config.train_dataset.batch_size,
                train_batch_size=config.train_dataset.batch_size,
            )

        # Initialize engines first — the scheduler must know about roles
        # before the data controller can colocate with them.
        engine_init_kwargs = {"addr": None, "ft_spec": ft_spec}
        self.actor.initialize(**engine_init_kwargs, role="actor")
        if self.critic is not None:
            self.critic.initialize(**engine_init_kwargs, role="critic")
        if self.ref is not None:
            self.ref.initialize(**engine_init_kwargs, role="ref")

        if (
            self.config.teacher is not None
            and self.config.teacher.engine_type == "train"
        ):
            assert self.config.teacher.train is not None
            self.teacher = self._create_train_engine(
                self.config.teacher.train, self.teacher_alloc
            )
            self.teacher.initialize(**engine_init_kwargs, role="teacher")

        # Save initial LoRA weights if enabled (for inference server pre-loading)
        initial_lora_path = self._save_initial_lora_weights()

        # Initialize inference with LoRA path
        self.rollout = self._init_rollout(
            config.rollout, is_eval=False, lora_path=initial_lora_path
        )

        self.eval_rollout = None
        if not self._online_mode:
            self.eval_rollout = self._init_rollout(
                config.rollout, is_eval=True, lora_path=initial_lora_path
            )
        if (
            self.config.teacher is not None
            and self.config.teacher.engine_type == "rollout"
        ):
            self.teacher = self._init_teacher_rollout(self.config.teacher.rollout)

        # Proxy worker initialization (lazy, for AgentWorkflow support)
        self._proxy_started = False

        # Prepare weight update meta and connect to inference engine.
        # v2 controllers pick transport from use_lora: LoRA must go through
        # disk (P2P transports cannot carry PEFT-wrapped tensors); non-LoRA
        # uses awex. v1 keeps the legacy weight_update_mode dispatch.
        if self.config.actor._version == "v2":
            if config.actor.use_lora:
                disk_kwargs: dict[str, Any] = {
                    "experiment_name": config.experiment_name,
                    "trial_name": config.trial_name,
                    "file_root": config.cluster.fileroot,
                    "name": "default",
                    "clear_checkpoint_after_load": True,
                    "use_lora": config.actor.use_lora,
                    "lora_name": config.gconfig.lora_name,
                    "base_model_name": config.actor.path,
                    # Keep enough recent adapter versions for off-policy
                    # rollouts (max_head_offpolicyness) plus a safety margin;
                    # older versions are unloaded to bound sglang VRAM and
                    # avoid the adapter-accumulation hang.
                    "lora_keep_versions": config.rollout.max_head_offpolicyness + 2,
                }
                self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
            else:
                self.weight_update_meta = WeightUpdateMeta.from_awex()
        elif self.config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            if config.actor.use_lora:
                disk_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                        # Keep enough recent adapter versions for off-policy
                        # rollouts (max_head_offpolicyness) plus a safety margin;
                        # older versions are unloaded to bound sglang VRAM and
                        # avoid the adapter-accumulation hang.
                        "lora_keep_versions": config.rollout.max_head_offpolicyness + 2,
                    }
                )
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif self.config.actor.weight_update_mode == "xccl":
            # NCCL/XCCL weight update (v1 only)
            xccl_kwargs: dict[str, Any] = {
                "gen_allocation": self.rollout_alloc,
            }

            if config.actor.use_lora:
                xccl_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                    }
                )

            if self.actor_alloc.backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    **xccl_kwargs
                )
            else:
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        else:
            raise ValueError(
                f"Invalid weight update mode: {self.config.actor.weight_update_mode}"
            )

        self.actor.connect_engine(self.rollout, self.weight_update_meta)

        # Set up evaluation (skip in online mode)
        self.evaluator = Evaluator(config.evaluator, ft_spec)

        # Set up save as HF model
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)

        # Set up statistics logging (wandb, tensoboard, etc.)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Set up checkpointing for recover
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )

        # After recovery, sync the staleness manager so its capacity formula
        # stays bounded despite the version jumping from 0 to recovery_version.
        if self.recover_info is not None:
            recovery_version = self.recover_info.last_step_info.global_step + 1
            if is_single_controller():
                sm = self.rollout.staleness_manager
            else:
                sm = self.rollout.workflow_executor.staleness_manager
            if sm is not None:
                sm.on_version_recovered(recovery_version)

        self._config_perf_tracer()
        self._apply_initial_offload_policy()

    @staticmethod
    def _is_colocation(strategy: SchedulingStrategy | None) -> bool:
        if strategy is None:
            return False
        return strategy.type in (
            SchedulingStrategyType.colocation,
            SchedulingStrategyType.colocation.value,
            "colocation",
        )

    def _is_actor_rollout_colocated(self, config: PPOConfig) -> bool:
        actor_s = config.actor.scheduling_strategy
        rollout_s = config.rollout.scheduling_strategy
        return (self._is_colocation(actor_s) and actor_s.target == "rollout") or (
            self._is_colocation(rollout_s) and rollout_s.target == "actor"
        )

    def _onload_model(self, engine, role: str) -> None:
        with (
            stats_tracker.record_timing(f"{role}_onload"),
            perf_tracer.trace_scope(
                f"train.{role}_onload",
                category=Category.IO,
            ),
        ):
            engine.onload()

    def _offload_model(self, engine, role: str) -> None:
        with (
            stats_tracker.record_timing(f"{role}_offload"),
            perf_tracer.trace_scope(
                f"train.{role}_offload",
                category=Category.IO,
            ),
        ):
            engine.offload()

    def _clear_consumed_rollout_batch(self, rollout_batch) -> None:
        """Release rollout RTensors once the self-contained advantage batch exists."""

        self.actor.clear_batches(rollout_batch)
        if self.critic is not None:
            self.critic.clear_batches(rollout_batch)
        if self.ref is not None:
            self.ref.clear_batches(rollout_batch)
        logger.info("PPO_CPU_STAGING_AUDIT phase=clear_rollout_batch status=ok")

    def _reset_actor_memory_before_ppo(self, did_recompute_logp: bool) -> None:
        """Discard recompute allocator cache before the optimizer state is materialized.

        With actor offload enabled, a recompute forward can leave tens of GiB in the
        CUDA caching allocator even after its tensors are released.  PPO then creates
        gradients and Adam state on top of that cache.  A TMS offload/onload cycle
        preserves the live actor/optimizer allocations while rebuilding the CUDA
        allocator from only the live set.
        """

        if not (self._should_offload_actor and did_recompute_logp):
            return
        logger.info("PPO_MEMORY_RESET_AUDIT phase=before_offload status=start")
        self._offload_model(self.actor, role="actor_pre_ppo_reset")
        self._onload_model(self.actor, role="actor_pre_ppo_reset")
        self.actor.get_device_stats().log("actor memory reset before ppo")
        logger.info("PPO_MEMORY_RESET_AUDIT phase=after_onload status=ok")

    def _offload_rollout(self, is_eval: bool = False):
        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        with (
            stats_tracker.record_timing("rollout_pause"),
            perf_tracer.trace_scope(
                "train.rollout_pause",
                category=Category.INSTR,
            ),
        ):
            rollout.pause()

        with (
            stats_tracker.record_timing("rollout_pause_generation"),
            perf_tracer.trace_scope(
                "train.rollout_pause_generation",
                category=Category.INSTR,
            ),
        ):
            call_maybe_async(rollout.pause_generation)

        with (
            stats_tracker.record_timing("rollout_offload"),
            perf_tracer.trace_scope(
                "train.rollout_offload",
                category=Category.IO,
            ),
        ):
            rollout.offload()

    def _onload_rollout(self, is_eval: bool = False) -> None:
        cleanup_error: Exception | None = None

        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        try:
            with (
                stats_tracker.record_timing("rollout_onload"),
                perf_tracer.trace_scope(
                    "train.rollout_onload",
                    category=Category.IO,
                ),
            ):
                rollout.onload()
        except Exception as exc:  # noqa: BLE001
            cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_continue_generation"),
                perf_tracer.trace_scope(
                    "train.rollout_continue_generation",
                    category=Category.INSTR,
                ),
            ):
                call_maybe_async(rollout.continue_generation)
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_resume"),
                perf_tracer.trace_scope(
                    "train.rollout_resume",
                    category=Category.INSTR,
                ),
            ):
                rollout.resume()
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        if cleanup_error is not None:
            raise cleanup_error

    def _apply_initial_offload_policy(self) -> None:
        if self._should_offload_rollout:
            self._offload_rollout()
        if self._should_offload_ref:
            self._offload_model(self.ref, role="ref")
        if self._should_offload_critic:
            self._offload_model(self.critic, role="critic")
        if self._should_offload_teacher:
            self._offload_model(self.teacher, role="teacher")
        if self._should_offload_actor:
            self._offload_model(self.actor, role="actor")

    def train(
        self,
        workflow: WorkflowLike | None = None,
        eval_workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        total_epochs: int | None = None,
    ):
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive: {total_epochs}")
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        # Initialize proxy workers if not using RolloutWorkflow
        if workflow is None:
            agent_cfg = self.config.rollout.agent
            if agent_cfg is not None and agent_cfg.mode == "online":
                self._ensure_proxy_started()
            else:
                raise ValueError(
                    "workflow must be specified for train() unless "
                    "agent.mode='online' is configured. "
                    "Pass a RolloutWorkflow, AgentWorkflow, or callable."
                )
        elif self._requires_proxy_workflow(workflow):
            self._ensure_proxy_started()

        for global_step in range(start_step, max_steps):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            if self._should_offload_rollout:
                self._onload_rollout()
            with (
                stats_tracker.record_timing("rollout"),
                perf_tracer.trace_scope(
                    "train.rollout",
                    category=Category.COMPUTE,
                    args={
                        "global_step": global_step,
                        "epoch_step": step,
                    },
                ),
            ):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=dynamic_filter_fn,
                    group_size=config.gconfig.n_samples,
                    dynamic_bs=self.config.dynamic_bs,
                )
            if self._should_offload_rollout:
                self._offload_rollout()

            if self.critic is not None:
                if self._should_offload_critic:
                    self._onload_model(self.critic, role="critic")
                with (
                    stats_tracker.record_timing("critic_values"),
                    perf_tracer.trace_scope(
                        "train.compute_values",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    values = self.critic.compute_values(rollout_batch)
                    for traj, v in zip(rollout_batch, values):
                        traj["values"] = v
                    self.critic.get_device_stats().log("critic values")
                # Critic stays onloaded — offloaded after ppo_update below

            if self.ref is not None:
                if self._should_offload_ref:
                    self._onload_model(self.ref, role="ref")
                with (
                    stats_tracker.record_timing("ref_logp"),
                    perf_tracer.trace_scope(
                        "train.ref_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    ref_logps = _compute_logp_in_rpc_chunks(
                        self.ref,
                        rollout_batch,
                        role="ref",
                        dp_size=ModelAllocation.from_str(
                            config.ref.backend, name="ref"
                        ).parallel.dp_size,
                    )
                    for traj, logp in zip(rollout_batch, ref_logps):
                        traj["ref_logp"] = logp
                    self.ref.get_device_stats().log("ref logp")
                if self._should_offload_ref:
                    self._offload_model(self.ref, role="ref")

            if self.teacher is not None:
                if self._should_offload_teacher:
                    self._onload_model(self.teacher, role="teacher")
                with (
                    stats_tracker.record_timing("teacher_logp"),
                    perf_tracer.trace_scope(
                        "train.teacher_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    teacher_logps = _compute_logp_in_rpc_chunks(
                        self.teacher,
                        rollout_batch,
                        role="teacher",
                        dp_size=self.teacher_alloc.parallel.dp_size,
                    )
                    for traj, logp in zip(rollout_batch, teacher_logps):
                        traj["teacher_logp"] = logp
                        traj["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                        traj["distill_loss_weight"] = (
                            self.config.teacher.distill_loss_weight
                        )
                if self._should_offload_teacher:
                    self._offload_model(self.teacher, role="teacher")

            if self._should_offload_actor:
                self._onload_model(self.actor, role="actor")
            if config.actor.should_compute_prox_logp():
                with (
                    stats_tracker.record_timing("recompute_logp"),
                    perf_tracer.trace_scope(
                        "train.recompute_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    prox_logps = _compute_logp_in_rpc_chunks(
                        self.actor,
                        rollout_batch,
                        role="actor",
                        dp_size=self.actor_alloc.parallel.dp_size,
                    )
                    for traj, logp in zip(rollout_batch, prox_logps):
                        traj["prox_logp"] = logp
                    self.actor.get_device_stats().log("recompute logp")

            alignment_dir = os.getenv("MAAPACMAN_LOGPROB_AUDIT_DIR")
            alignment_steps = _parse_update_gate_global_steps(
                os.getenv("MAAPACMAN_LOGPROB_AUDIT_GLOBAL_STEPS"),
                "0",
            )
            if alignment_dir and global_step in alignment_steps:
                if not config.actor.should_compute_prox_logp():
                    raise RuntimeError(
                        "Pacman log-prob audit requires proximal actor log-probs"
                    )
                acceptance_mode = os.getenv(
                    "MAAPACMAN_LOGPROB_ACCEPTANCE_MODE", "exact"
                )
                behavior_ratio_lower = float(
                    os.getenv("MAAPACMAN_LOGPROB_RATIO_LOWER", "0.8")
                )
                behavior_ratio_upper = float(
                    os.getenv("MAAPACMAN_LOGPROB_RATIO_UPPER", "1.25")
                )
                if acceptance_mode == "behavior":
                    rejection = config.actor.rejection_sampling
                    reward_norm = config.actor.reward_norm
                    reward_contract = getattr(config, "reward_objective_contract", None)
                    episode_grpo_contract = (
                        reward_contract == "episode_return_group_v1"
                        and reward_norm is not None
                        and reward_norm.mean_level == "group"
                        and reward_norm.std_level == "group"
                        and reward_norm.group_size == config.gconfig.n_samples
                        and not reward_norm.mean_leave1out
                        and config.actor.adv_norm is None
                    )
                    unnormalized_option_contract = (
                        reward_contract
                        in {
                            "step_local_raw_v1",
                            "option_return_raw_v1",
                        }
                        and reward_norm is None
                        and config.actor.adv_norm is None
                    )
                    rejection_contract = (
                        rejection is not None
                        and rejection.level == "sequence"
                        and rejection.action == "mask"
                        and rejection.metric == "ratio"
                        and rejection.agg == "sum"
                        and rejection.lower is not None
                        and rejection.lower == behavior_ratio_lower
                        and rejection.upper == behavior_ratio_upper
                    )
                    behavior_training_contract = (
                        config.actor.use_decoupled_loss
                        and config.actor.prox_logp_method == "recompute"
                        and config.actor.kl_logprob_source == "proximal"
                        and not config.actor.use_sapo_loss
                        and not config.actor.use_cispo_loss
                        and config.actor.ppo_n_minibatches == 1
                        and (episode_grpo_contract or unnormalized_option_contract)
                        and config.critic is None
                        and config.teacher is None
                    )
                    if not behavior_training_contract or not rejection_contract:
                        raise RuntimeError(
                            "Pacman behavior acceptance requires matching "
                            "decoupled option-joint rejection and proximal KL"
                        )
                _audit_pacman_logprob_alignment(
                    rollout_batch,
                    output_dir=alignment_dir,
                    global_step=global_step,
                    max_abs_tolerance=float(
                        os.getenv("MAAPACMAN_LOGPROB_MAX_ABS_TOLERANCE", "1e-3")
                    ),
                    mean_abs_tolerance=float(
                        os.getenv("MAAPACMAN_LOGPROB_MEAN_ABS_TOLERANCE", "1e-4")
                    ),
                    actor_reference_max_abs_tolerance=float(
                        os.getenv("MAAPACMAN_ACTOR_REF_MAX_ABS_TOLERANCE", "1e-5")
                    ),
                    actor_reference_mean_abs_tolerance=float(
                        os.getenv("MAAPACMAN_ACTOR_REF_MEAN_ABS_TOLERANCE", "1e-6")
                    ),
                    acceptance_mode=acceptance_mode,
                    behavior_ratio_lower=behavior_ratio_lower,
                    behavior_ratio_upper=behavior_ratio_upper,
                    behavior_branch_mean_abs_log_ratio=float(
                        os.getenv(
                            "MAAPACMAN_LOGPROB_BRANCH_MEAN_ABS_LOG_RATIO",
                            "0.05",
                        )
                    ),
                    behavior_max_filtered_fraction=float(
                        os.getenv("MAAPACMAN_LOGPROB_MAX_FILTERED_FRACTION", "0.07")
                    ),
                    singleton_max_abs_tolerance=float(
                        os.getenv(
                            "MAAPACMAN_LOGPROB_SINGLETON_MAX_ABS_TOLERANCE",
                            "1e-7",
                        )
                    ),
                    require_actor_reference_alignment=os.getenv(
                        "MAAPACMAN_REQUIRE_ACTOR_REFERENCE_ALIGNMENT", "1"
                    )
                    .strip()
                    .lower()
                    not in {"0", "false", "no"},
                    expected_option_tokens=int(
                        os.getenv("MAAPACMAN_EXPECTED_OPTION_TOKENS", "1")
                    ),
                )

            with (
                stats_tracker.record_timing("compute_advantage"),
                perf_tracer.trace_scope(
                    "train.compute_advantage",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                if (
                    getattr(config, "reward_objective_contract", None)
                    == "episode_return_group_v1"
                ):
                    _require_episode_grpo_fields(
                        rollout_batch,
                        EPISODE_GRPO_ROLLOUT_FIELDS,
                        stage="rollout",
                    )
                elif (
                    getattr(config, "reward_objective_contract", None)
                    == "option_return_raw_v1"
                ):
                    _require_episode_grpo_fields(
                        rollout_batch,
                        ("rollout_episode_ids",),
                        stage="rollout",
                    )
                repartition_for_ppo = (
                    is_single_controller()
                    and self.actor_alloc.parallel.dp_size > 1
                    and getattr(config, "reward_objective_contract", None)
                    in {"option_return_raw_v1", "episode_return_group_v1"}
                )
                if repartition_for_ppo:
                    adv_batch = self.actor.compute_advantages(
                        rollout_batch, repartition_for_ppo=True
                    )
                    logger.info(
                        "PPO_OPTION_SEQUENCE_REPARTITION prompt_groups=%s "
                        "option_sequences=%s actor_dp=%s",
                        len(rollout_batch),
                        len(adv_batch),
                        self.actor_alloc.parallel.dp_size,
                    )
                else:
                    adv_batch = self.actor.compute_advantages(rollout_batch)
                if getattr(config, "reward_objective_contract", None) in {
                    "option_return_raw_v1",
                    "episode_return_group_v1",
                }:
                    _require_episode_grpo_fields(
                        adv_batch,
                        ("episode_loss_weights",),
                        stage="advantage",
                    )
                self.actor.get_device_stats().log("compute advantages")

            # ``adv_batch`` contains every field needed by actor/critic PPO.
            # Keeping the source rollout RTensors alive until step end doubles
            # the large token-level payload on the training GPUs.  Drain those
            # shards now; the final clear remains intentionally idempotent.
            if is_single_controller():
                self._clear_consumed_rollout_batch(rollout_batch)
                self.actor.get_device_stats().log("clear rollout before ppo")

            # The advantage payload is self-contained at this point.  Rebuild the
            # actor allocator before PPO so cached recompute buffers do not compete
            # with newly materialized gradient and optimizer state.
            self._reset_actor_memory_before_ppo(
                did_recompute_logp=config.actor.should_compute_prox_logp()
            )

            # Wait for async checkpoint staging to complete before modifying parameters
            self.saver.maybe_wait_for_staging()

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
            ):
                self.actor.start_memory_profile(config.memory_profiler.max_entries)

            with (
                stats_tracker.record_timing("train_step"),
                perf_tracer.trace_scope(
                    "train.ppo_update",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self.actor.ppo_update(adv_batch)
                self.actor.step_lr_scheduler()
                self.actor.get_device_stats().log("ppo update")
                audit_step = int(
                    os.getenv("MAAPACMAN_FIRST_UPDATE_GATE_GLOBAL_STEP", "16")
                )
                if (
                    os.getenv("MAAPACMAN_STRICT_RUNTIME_AUDIT") == "1"
                    and global_step == audit_step
                ):
                    self.actor._custom_function_call(
                        "runtime_state_audit",
                        expected_phase="cuda",
                        rpc_meta={"broadcast": False},
                    )

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
            ):
                log_dir = StatsLogger.get_log_path(config.stats_logger)
                snapshot_dir = os.path.join(
                    log_dir, "memory_snapshots", f"step_{global_step}"
                )
                os.makedirs(snapshot_dir, exist_ok=True)
                self.actor.stop_memory_profile(snapshot_dir)
                logger.info(f"Memory snapshots saved to {snapshot_dir}")

            if self.critic is not None:
                with (
                    stats_tracker.record_timing("critic_train_step"),
                    perf_tracer.trace_scope(
                        "train.critic_ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.critic.ppo_update(adv_batch)
                    self.critic.step_lr_scheduler()
                    self.critic.get_device_stats().log("ppo critic update")
                if self._should_offload_critic:
                    self._offload_model(self.critic, role="critic")

            # pause inference for updating weights, save, and evaluation
            self.rollout.pause()

            # Actor already onloaded; engine-internal _offload_aware_context
            # calls in update_weights/save are no-ops.

            with (
                stats_tracker.record_timing("update_weights"),
                perf_tracer.trace_scope(
                    "train.update_weights",
                    category=Category.COMM,
                    args={"global_step": global_step},
                ),
            ):
                # Use versioned path for weight updates
                new_version = global_step + 1
                versioned_meta = self.weight_update_meta.with_version(new_version)
                self.actor.update_weights(versioned_meta)

                self.actor.set_version(new_version)
                if self.critic is not None:
                    self.critic.set_version(new_version)
                self.rollout.set_version(new_version)
                if self.eval_rollout is not None:
                    self.eval_rollout.set_version(new_version)

            # Peek at this update's actor metrics before deciding whether a
            # newly observed global-best checkpoint must be saved. Keep the
            # trackers intact for the normal logging/export path below.
            if config.saver.keep_best_metric is not None:
                self._actor_stats_for_checkpoint = self.actor.export_stats(reset=False)
            else:
                self._actor_stats_for_checkpoint = {}

            with (
                stats_tracker.record_timing("save"),
                perf_tracer.trace_scope(
                    "train.save",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)

            with (
                stats_tracker.record_timing("checkpoint_for_recover"),
                perf_tracer.trace_scope(
                    "train.checkpoint",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_recover_checkpoint(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            # Offload actor before eval
            if self._should_offload_actor:
                self._offload_model(self.actor, role="actor")

            if self._should_offload_rollout:
                self._onload_rollout(is_eval=True)
            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self._evaluate(
                    eval_workflow=eval_workflow,
                    eval_workflow_kwargs=eval_workflow_kwargs,
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )
            if self._should_offload_rollout:
                self._offload_rollout(is_eval=True)

            with (
                stats_tracker.record_timing("clear_batches"),
                perf_tracer.trace_scope(
                    "train.clear_batches",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                # Each role runs in its own Python process with a
                # process-local ``_fetch_buffer``; one HTTP DELETE to the
                # storage owner clears ``_storage`` but not per-consumer
                # caches. Fan out ``clear_batches`` to every role that
                # localized the batch — see areal-project/AReaL#1209.
                # SPMD mode never populates ``_fetch_buffer`` (no RTensor
                # round-trip), so the fan-out is single-controller only.
                if is_single_controller():
                    self.actor.clear_batches(rollout_batch, adv_batch)
                    if self.critic is not None:
                        self.critic.clear_batches(rollout_batch, adv_batch)
                    if self.ref is not None:
                        self.ref.clear_batches(rollout_batch)
                    if self.data_controller is not None:
                        self.data_controller.clear_batches()

            with perf_tracer.trace_scope(
                "train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            gate_dir = os.getenv("MAAPACMAN_UPDATE_GATE_DIR") or os.getenv(
                "MAAPACMAN_FIRST_UPDATE_GATE_DIR"
            )
            gate_steps = _parse_update_gate_global_steps(
                os.getenv("MAAPACMAN_UPDATE_GATE_GLOBAL_STEPS"),
                os.getenv("MAAPACMAN_FIRST_UPDATE_GATE_GLOBAL_STEP", "16"),
            )
            if gate_dir and global_step in gate_steps:
                os.makedirs(gate_dir, exist_ok=True)
                completed_iteration = global_step + 1
                waiting_path = os.path.join(
                    gate_dir, f"WAITING_AFTER_ITER{completed_iteration}"
                )
                allow_path = os.path.join(
                    gate_dir, f"ALLOW_AFTER_ITER{completed_iteration}"
                )
                abort_path = os.path.join(
                    gate_dir, f"ABORT_AFTER_ITER{completed_iteration}"
                )
                with open(waiting_path, "w", encoding="utf-8") as stream:
                    stream.write(
                        f"global_step={global_step}\n"
                        f"utc={datetime.utcnow().isoformat()}Z\n"
                    )
                logger.info(
                    "FIRST_UPDATE_GATE_WAIT global_step=%s waiting=%s allow=%s abort=%s",
                    global_step,
                    waiting_path,
                    allow_path,
                    abort_path,
                )
                while not os.path.exists(allow_path):
                    if os.path.exists(abort_path):
                        logger.warning(
                            "FIRST_UPDATE_GATE_ABORT global_step=%s marker=%s",
                            global_step,
                            abort_path,
                        )
                        return
                    time.sleep(5)
                logger.info(
                    "FIRST_UPDATE_GATE_RELEASE global_step=%s marker=%s",
                    global_step,
                    allow_path,
                )

            # Resume rollout
            self.rollout.resume()

            self._save_perf_tracer(step=global_step)

    def close(self):
        self.saver.finalize()
        if hasattr(self, "_train_rdataset") and self._train_rdataset is not None:
            self._train_rdataset.close()
        if hasattr(self, "_valid_rdataset") and self._valid_rdataset is not None:
            self._valid_rdataset.close()
        if hasattr(self, "data_controller") and self.data_controller is not None:
            self.data_controller.destroy()
        self.stats_logger.close()
        if self.eval_rollout is not None:
            self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.teacher is not None:
            self.teacher.destroy()
        if self.ref is not None:
            self.ref.destroy()
        if self.critic is not None:
            self.critic.destroy()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def _config_perf_tracer(self):
        rank = int(os.getenv("RANK", "0"))
        if self.config.perf_tracer is None:
            return
        perf_tracer.configure(self.config.perf_tracer, rank=rank, role="master")

        if not is_single_controller():
            return

        self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        if self.critic is not None:
            self.critic.config_perf_tracer(self.config.perf_tracer, role="critic")
        if self.ref is not None:
            self.ref.config_perf_tracer(self.config.perf_tracer, role="ref")
        self.rollout.config_perf_tracer(self.config.perf_tracer, role="rollout")
        if self.eval_rollout is not None:
            self.eval_rollout.config_perf_tracer(
                self.config.perf_tracer, role="eval-rollout"
            )

    def _save_perf_tracer(self, step: int):
        self.actor.save_perf_tracer(step=step)
        if self.ref is not None:
            self.ref.save_perf_tracer(step=step)
        if self.critic is not None:
            self.critic.save_perf_tracer(step=step)
        if self.eval_rollout is not None:
            self.eval_rollout.save_perf_tracer(step=step)
        self.rollout.save_perf_tracer(step=step)
        perf_tracer.save(step=step)

    def _init_scheduler(self) -> Scheduler:
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        elif cfg.type == "ray":
            return RayScheduler(exp_config=self.config)
        elif cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _create_dataloader(
        self,
        dataset: Dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
        )

    def _amend_xccl_weight_update_envvar(self):
        if not is_single_controller():
            # These environs are set by the launcher in the SPMD mode.
            return
        if self.rollout_alloc.backend != "sglang":
            return

        # Disable some environ for NCCL weight update.
        for spec in self.config.actor.scheduling_spec:
            spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

    def _create_train_engine(
        self, actor_config: PPOActorConfig, alloc: ModelAllocation
    ) -> FSDPPPOActor | MegatronPPOActor | ArchonPPOActor | PPOActorController:
        """Create a training engine (actor or ref) based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOActor

            actor_cls = FSDPPPOActor
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOActor

            actor_cls = MegatronPPOActor
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOActor

            actor_cls = ArchonPPOActor
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor

    def _create_critic(
        self, critic_config: PPOCriticConfig, alloc: ModelAllocation
    ) -> FSDPPPOCritic | MegatronPPOCritic | ArchonPPOCritic | PPOCriticController:
        """Create a critic engine based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOCritic

            critic_cls = FSDPPPOCritic
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOCritic

            critic_cls = MegatronPPOCritic
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOCritic

            critic_cls = ArchonPPOCritic
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            critic = critic_cls.as_controller(critic_config, self.scheduler)
        else:
            critic = critic_cls(config=critic_config)
        critic.create_process_group(parallel_strategy=alloc.parallel)
        return critic

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
        lora_path: str | None = None,
    ) -> InferenceEngine | RolloutController:
        if lora_path is not None and not is_single_controller():
            raise ValueError(
                "LoRA is only supported in single-controller mode. "
                "Use `python3 train.py scheduler.type=local` instead of "
                "`python3 -m areal.infra.launcher.local`."
            )
        # Create a working copy of config
        config = deepcopy(rollout_config)
        if is_eval:
            # NOTE: eval does not have any offpolicyness control
            config.max_head_offpolicyness = int(1e12)
            # eval-rollout uses the same inference servers as rollout
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        # Determine engine class and server args based on backend
        rollout_backend = self.rollout_alloc.backend
        if rollout_backend == "sglang":
            if self.config.rollout.return_routed_experts:
                self.config.sglang.enable_return_routed_experts = True
            if lora_path is not None and self.config.actor.use_lora:
                self.config.sglang.lora_paths = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
                base_gpu_id=0,
            )
        elif rollout_backend == "vllm":
            if self.config.rollout.return_routed_experts:
                raise ValueError(
                    "return_routed_experts is not supported with vLLM backend. Please disable return_routed_experts or switch to SGLang backend."
                )
            if lora_path is not None and self.config.actor.use_lora:
                self.config.vllm.lora_modules = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
            )
            # vLLM does not require LoRA paths during initialization.
            # LoRA is attached to generation requests.
        else:
            raise ValueError(
                f"Invalid backend: {rollout_backend}, expected sglang or vllm"
            )

        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine

        # Single-controller mode - no engine instantiation needed
        if config._version == "v2":
            controller = RolloutControllerV2(
                config=config, scheduler=cast(Scheduler, self.scheduler)
            )
        else:
            controller = engine_cls.as_controller(config, self.scheduler)
        init_kwargs = dict(
            role="rollout",
            server_args=server_args,
        )
        if is_eval:
            assert len(self.rollout.server_infos) > 0
            init_kwargs["server_infos"] = self.rollout.server_infos
            init_kwargs["role"] = "eval-rollout"
        controller.initialize(**init_kwargs)
        return controller

    def _init_teacher_rollout(
        self, rollout_config: InferenceEngineConfig
    ) -> InferenceEngine | RolloutController:
        if self.teacher_alloc is None:
            raise RuntimeError("teacher_alloc is not initialized")
        rollout_alloc = self.teacher_alloc
        config = deepcopy(rollout_config)
        if rollout_alloc.backend == "sglang":
            engine_cls = RemoteSGLangEngine
            teacher_sglang_cfg = deepcopy(self.config.sglang)
            if self.config.teacher is not None and self.config.teacher.path:
                teacher_sglang_cfg.model_path = self.config.teacher.path
            server_args = SGLangConfig.build_args(
                sglang_config=teacher_sglang_cfg,
                tp_size=rollout_alloc.parallel.tp_size,
                pp_size=rollout_alloc.parallel.pp_size,
                base_gpu_id=0,
            )
        elif rollout_alloc.backend == "vllm":
            engine_cls = RemotevLLMEngine
            teacher_vllm_cfg = deepcopy(self.config.vllm)
            if self.config.teacher is not None and self.config.teacher.path:
                teacher_vllm_cfg.model = self.config.teacher.path
                if not rollout_config.tokenizer_path:
                    config.tokenizer_path = self.config.teacher.path
            server_args = vLLMConfig.build_args(
                vllm_config=teacher_vllm_cfg,
                tp_size=rollout_alloc.parallel.tp_size,
                pp_size=rollout_alloc.parallel.pp_size,
            )
        else:
            raise ValueError(
                f"Invalid teacher rollout backend: {rollout_alloc.backend}, expected sglang or vllm"
            )
        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine
        controller = engine_cls.as_controller(config, self.scheduler)
        controller.initialize(role="teacher", server_args=server_args)
        return controller

    def _save_initial_lora_weights(self) -> str | None:
        """Save initial LoRA weights for inference server pre-loading.

        Returns path to saved LoRA weights, or None if LoRA is disabled.
        """
        if not self.config.actor.use_lora:
            return None

        path = os.path.join(
            Saver.get_model_save_root(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.cluster.fileroot,
                name="actor",
            ),
            "initial_lora",
        )

        meta = SaveLoadMeta(
            path=path,
            weight_format="hf",
            with_optim=False,
            tokenizer=self.tokenizer,
            processor=self.processor,
            base_model_path=self.config.actor.path,
        )
        # Save LoRA weights using engine's HuggingFace save
        self.actor.save(meta=meta)

        return path

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        # Save as HF models for evaluation
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
            metrics=self._actor_stats_for_checkpoint,
        )
        if self.critic is not None:
            self.saver.save(
                self.critic,
                epoch,
                epoch_step,
                global_step,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name="critic",
            )
        # Async mode: synchronization handled by AsyncCheckpointManager
        if not self.saver.is_async and not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        # Save recoverable checkpoints
        to_save: dict = dict(default=self.actor)
        if self.critic is not None:
            to_save["critic"] = self.critic
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=len(self.train_dataloader),
        )
        self.recover_handler.dump(
            to_save,
            step_info,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs,
    ):
        if self.actor.is_data_parallel_head():
            cnt = 0
            for data in self.valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self.config.eval_gconfig.n_samples,
                        is_eval=True,
                    )
                    cnt += 1
            self.eval_rollout.wait(cnt, timeout=None)

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _evaluate(
        self,
        eval_workflow: WorkflowLike | None,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        if (
            self.eval_rollout is None
            or self.valid_dataloader is None
            or eval_workflow is None
        ):
            return
        self.evaluator.evaluate(
            functools.partial(
                self._evaluate_fn,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
            ),
            epoch,
            epoch_step,
            global_step,
        )
        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        # Upload statistics to the logger (e.g., wandb)
        stats = self.actor.export_stats()
        stats.update(self.rollout.export_stats())
        if self.eval_rollout is not None:
            stats.update(self.eval_rollout.export_stats())
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _validate_cfg(self):
        """validate config for incompatible settings before weight initialization, to avoid wasted resources on spawning workers and loading models."""
        rollout_backend = self.rollout_alloc.backend
        actor_backend = self.actor_alloc.backend
        requires_train_engine_offload = any(
            (
                self._should_offload_rollout,
                self._should_offload_actor,
                self._should_offload_critic,
                self._should_offload_ref,
                self._should_offload_teacher,
            )
        )

        if requires_train_engine_offload and not self.config.enable_offload:
            raise ValueError(
                "enable_offload must be True when colocation scheduling or train-engine "
                "offload is enabled. Please set enable_offload=True."
            )

        if (
            self._is_actor_rollout_colocated(self.config)
            and self.config.actor.weight_update_mode != "disk"
        ):
            raise ValueError(
                "weight_update_mode must be 'disk' when colocation scheduling is enabled. "
                "Please set actor.weight_update_mode=disk."
            )

        if rollout_backend == "vllm" and self.config.rollout.return_routed_experts:
            raise ValueError(
                "return_routed_experts is only supported with SGLang backend. "
                "Please disable return_routed_experts or switch to SGLang backend."
            )
        if (
            actor_backend == "megatron"
            and self.config.actor.use_lora
            and rollout_backend == "sglang"
        ):
            raise ValueError(
                "Megatron actor with LoRA is not supported with SGLang rollout in "
                "RL trainer. Please use vLLM rollout backend, or disable LoRA, or "
                "switch actor backend from Megatron."
            )

        # Ensure actor and rollout controller versions match.
        actor_version = self.config.actor._version
        rollout_version = self.config.rollout._version
        if actor_version != rollout_version:
            raise ValueError(
                f"actor._version ('{actor_version}') and rollout._version "
                f"('{rollout_version}') must match. Both must be 'v1' or both 'v2'."
            )

    def _requires_proxy_workflow(self, workflow: WorkflowLike | None) -> bool:
        """Check if workflow requires proxy workers (i.e., not a RolloutWorkflow).

        Returns True if:
        - Workflow is NOT a RolloutWorkflow instance
        - Workflow is NOT a RolloutWorkflow class
        - Workflow is a string that does NOT import to a RolloutWorkflow

        This enables any callable object with a compatible signature to work
        without requiring inheritance from AgentWorkflow.
        """
        # None workflow is handled separately in train()
        if workflow is None:
            return False

        # Direct RolloutWorkflow instances
        if isinstance(workflow, RolloutWorkflow):
            return False

        # RolloutWorkflow classes
        if isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return False

        # String import paths
        if isinstance(workflow, str):
            from areal.utils.dynamic_import import import_from_string

            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError):
                # If import fails, assume it needs proxy (fail-safe)
                return True

            # Check if imported object is RolloutWorkflow
            if isinstance(imported_obj, RolloutWorkflow):
                return False
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                return False

        # Everything else requires proxy workers
        return True

    def _ensure_proxy_started(self) -> None:
        """Lazily initialize proxy workers when agent workflows are used.

        This method is called before training when a non-RolloutWorkflow is detected
        or when online mode is configured. It creates proxy workers colocated with
        rollout workers to handle OpenAI-compatible API requests.

        In online mode, also starts the proxy gateway for external access.
        """
        if self._proxy_started:
            return

        # Only initialize proxy in single-controller mode with RolloutController
        if not is_single_controller():
            raise NotImplementedError("Proxy workers not supported in SPMD mode")

        if not isinstance(self.rollout, RolloutController):
            self._proxy_started = True
            return

        # v1 controller needs an explicit proxy launch call
        logger.info("Initializing proxy workers for AgentWorkflow support")
        self.rollout.start_proxy()
        if self.eval_rollout is not None:
            self.eval_rollout.start_proxy()

        # Start proxy gateway for online mode.
        agent_cfg = self.config.rollout.agent
        if agent_cfg is not None and agent_cfg.mode == "online":
            self.rollout.start_proxy_gateway()
            logger.info(
                "Proxy gateway available at %s",
                self.rollout.proxy_gateway_addr,
            )

        self._proxy_started = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Training failed with exception: {exc_value}", exc_info=True)
        self.close()
        return False

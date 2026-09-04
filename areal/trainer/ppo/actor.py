# SPDX-License-Identifier: Apache-2.0

import copy
import functools
from typing import Any

import torch

from areal.api import TrainEngine
from areal.api.cli_args import MicroBatchSpec, PPOActorConfig, RejectionSamplingConfig
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.data import (
    KLEstimator,
    Normalization,
    TrajBatchMeta,
    batched_call,
    split_padded_tensor_dict_into_mb_list,
    tensor_container_to,
)
from areal.utils.functional import (
    cispo_loss_fn,
    ppo_actor_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.perf_tracer import trace_perf
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)

logger = logging.getLogger("PPOActor")

ROLLOUT_EPISODE_METADATA_KEYS = (
    "rollout_episode_ids",
    "rollout_episode_returns",
    "rollout_episode_group_sizes",
)
EPISODE_LOSS_WEIGHT_KEY = "episode_loss_weights"


def _split_ppo_batch_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a post-advantage PPO batch into independent sequence rows.

    Tensor rows retain a leading batch dimension of one. Token-aligned tensors
    are trimmed on axis 1 using the corresponding attention mask, including
    vector-valued fields shaped ``[batch, sequence, width]``. Batch-aligned
    lists (notably ``multi_modal_input``) move with the matching row.
    """

    attention_mask = data.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError(
            "PPO sequence repartition requires a 2-D attention_mask, got "
            f"{type(attention_mask).__name__}"
        )
    batch_size, padded_seqlen = attention_mask.shape
    if batch_size <= 0:
        raise ValueError("PPO sequence repartition requires a non-empty batch")

    rows: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        mask = attention_mask[row_index].bool()
        valid_seqlen = int(mask.sum().item())
        if valid_seqlen <= 0:
            raise ValueError(
                f"PPO sequence repartition found an empty row at index {row_index}"
            )
        if not bool(mask[:valid_seqlen].all()) or bool(mask[valid_seqlen:].any()):
            raise ValueError(
                "PPO sequence repartition requires right-padded attention masks; "
                f"row {row_index} is not contiguous"
            )

        row: dict[str, Any] = {}
        for key, value in data.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and value.shape[0] == batch_size
            ):
                row_value = value[row_index : row_index + 1]
                if value.ndim >= 2 and value.shape[1] == padded_seqlen:
                    row_value = row_value[:, :valid_seqlen, ...]
                row[key] = row_value
            elif isinstance(value, list):
                if len(value) != batch_size:
                    raise ValueError(
                        f"PPO batch list field {key!r} has length {len(value)}, "
                        f"expected {batch_size}"
                    )
                row[key] = [copy.deepcopy(value[row_index])]
            else:
                row[key] = copy.deepcopy(value)
        rows.append(row)
    return rows


def _flatten_ppo_advantage_groups(
    grouped_rows: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten worker-preserved prompt groups after advantage computation."""

    if not isinstance(grouped_rows, list) or any(
        not isinstance(group, list) for group in grouped_rows
    ):
        raise TypeError("PPO repartition expected list[list[dict]] worker results")
    rows = [row for group in grouped_rows for row in group]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise TypeError("PPO repartition produced an empty or invalid sequence list")
    return rows


def _normalize_rollout_episode_returns(
    episode_returns: torch.Tensor,
    episode_ids: torch.Tensor,
    expected_group_sizes: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    trajectory_group_sizes: list[int],
    normalizer: Normalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize episode returns and build equal-episode token loss weights.

    A grouped agent rollout can contain many model calls per episode. The leading
    tensor dimension therefore counts decisions, not episodes. Episode IDs must be
    contiguous inside each initial-state trajectory group; each unique episode is
    counted once regardless of how many decisions it contains. Each episode's valid
    tokens receive weights summing to one, so the PPO reducer averages within an
    episode before averaging across episodes.
    """
    tensors = {
        "rollout_episode_returns": episode_returns,
        "rollout_episode_ids": episode_ids,
        "rollout_episode_group_sizes": expected_group_sizes,
    }
    batch_size = episode_returns.shape[0]
    for name, value in tensors.items():
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(
                f"{name} must have shape [{batch_size}], got {tuple(value.shape)}"
            )
    if loss_mask.ndim != 2 or loss_mask.shape[0] != batch_size:
        raise ValueError(
            "whole-episode GRPO loss_mask must have shape [batch, sequence], "
            f"got {tuple(loss_mask.shape)}"
        )
    if sum(trajectory_group_sizes) != batch_size:
        raise ValueError(
            "trajectory group sizes must cover every decision sample: "
            f"sum={sum(trajectory_group_sizes)} batch={batch_size}"
        )
    if normalizer.mean_level != "group" or normalizer.std_level != "group":
        raise ValueError("whole-episode GRPO requires group-level reward mean and std")

    unique_returns: list[torch.Tensor] = []
    sample_to_episode: list[torch.Tensor] = []
    episode_loss_weights: list[torch.Tensor] = []
    episode_group_sizes: list[int] = []
    sample_offset = 0
    episode_offset = 0
    for decision_count in trajectory_group_sizes:
        group_slice = slice(sample_offset, sample_offset + decision_count)
        ids = episode_ids[group_slice]
        returns = episode_returns[group_slice]
        declared_sizes = expected_group_sizes[group_slice]
        if decision_count <= 0:
            raise ValueError("trajectory groups must contain at least one decision")
        if not torch.all(declared_sizes == declared_sizes[0]):
            raise ValueError("episode group-size metadata changed within one prompt")
        expected_episodes = int(declared_sizes[0].item())
        if expected_episodes != normalizer.group_size:
            raise ValueError(
                "episode group-size metadata does not match reward normalization: "
                f"metadata={expected_episodes} norm={normalizer.group_size}"
            )

        starts = torch.ones_like(ids, dtype=torch.bool)
        starts[1:] = ids[1:] != ids[:-1]
        first_indices = starts.nonzero(as_tuple=False).flatten()
        unique_ids = ids[first_indices]
        if unique_ids.unique().numel() != unique_ids.numel():
            raise ValueError(
                "one rollout episode appears in multiple non-contiguous spans"
            )
        n_episodes = int(unique_ids.numel())
        if n_episodes != expected_episodes:
            raise ValueError(
                "whole-episode GRPO requires the declared number of complete "
                f"episodes: expected={expected_episodes} actual={n_episodes}"
            )

        local_episode_index = starts.long().cumsum(0) - 1
        group_episode_returns = returns[first_indices]
        if not torch.equal(returns, group_episode_returns[local_episode_index]):
            raise ValueError("episode return changed between decisions")
        unique_returns.append(group_episode_returns)
        sample_to_episode.append(local_episode_index + episode_offset)

        group_loss_mask = loss_mask[group_slice].float()
        episode_token_counts = torch.zeros(
            n_episodes,
            dtype=torch.float32,
            device=group_loss_mask.device,
        )
        episode_token_counts.scatter_add_(
            0,
            local_episode_index,
            group_loss_mask.sum(dim=-1),
        )
        if torch.any(episode_token_counts <= 0):
            raise ValueError(
                "every complete rollout episode must contain at least one valid token"
            )
        per_sample_token_weight = episode_token_counts[local_episode_index].reciprocal()
        episode_loss_weights.append(
            group_loss_mask * per_sample_token_weight.unsqueeze(-1)
        )
        episode_group_sizes.append(n_episodes)
        episode_offset += n_episodes
        sample_offset += decision_count

    normalized_unique = normalizer(
        torch.cat(unique_returns), group_sizes=episode_group_sizes
    )
    return (
        normalized_unique[torch.cat(sample_to_episode)],
        torch.cat(episode_loss_weights),
    )


def _build_equal_episode_loss_weights(
    episode_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Give every rollout episode the same total PPO loss mass.

    Each valid token receives ``1 / n_valid_tokens_in_episode``. The weights
    therefore sum to one for every episode even when episodes contain different
    numbers of option decisions. The distributed reducer subsequently averages
    those unit episode masses across the complete rollout batch.
    """

    if episode_ids.ndim != 1:
        raise ValueError(
            "rollout_episode_ids must be one-dimensional, got "
            f"{tuple(episode_ids.shape)}"
        )
    if loss_mask.ndim != 2 or loss_mask.shape[0] != episode_ids.shape[0]:
        raise ValueError(
            "episode loss weighting requires loss_mask shape [batch, sequence]: "
            f"ids={tuple(episode_ids.shape)} mask={tuple(loss_mask.shape)}"
        )

    unique_ids, sample_to_episode = torch.unique(
        episode_ids, sorted=False, return_inverse=True
    )
    token_mask = loss_mask.float()
    episode_token_counts = torch.zeros(
        unique_ids.numel(), dtype=torch.float32, device=token_mask.device
    )
    episode_token_counts.scatter_add_(0, sample_to_episode, token_mask.sum(dim=-1))
    if torch.any(episode_token_counts <= 0):
        raise ValueError(
            "every rollout episode must contain at least one valid loss token"
        )
    return token_mask * episode_token_counts[sample_to_episode].reciprocal().unsqueeze(
        -1
    )


def _actor_loss_weight(data: dict[str, Any]) -> torch.Tensor:
    """Return the reducer mass used by both local and distributed loss scaling."""
    episode_weights = data.get(EPISODE_LOSS_WEIGHT_KEY)
    if episode_weights is not None:
        return episode_weights.sum()
    return data["loss_mask"].count_nonzero()


def _pad_ppo_batch_to_size(
    data: dict[str, Any], target_batch_size: int
) -> dict[str, Any]:
    """Append valid-forward, zero-loss rows without changing real PPO samples."""

    attention_mask = data.get("attention_mask")
    loss_mask = data.get("loss_mask")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError("PPO padding requires a 2-D attention_mask")
    if not isinstance(loss_mask, torch.Tensor) or loss_mask.ndim != 2:
        raise ValueError("PPO padding requires a 2-D loss_mask")

    batch_size = attention_mask.shape[0]
    if batch_size <= 0:
        raise ValueError("PPO padding requires a non-empty local batch")
    if target_batch_size < batch_size:
        raise ValueError(
            f"PPO padding target {target_batch_size} < local batch {batch_size}"
        )
    pad_count = target_batch_size - batch_size
    if pad_count == 0:
        return data

    # Reuse the shortest real row to minimize dummy forward work. The current
    # PPO repartition path deliberately stages this batch on CPU before RPC.
    template_index = int(attention_mask.sum(-1).argmin().item())
    padded = dict(data)
    zero_mass_keys = {"loss_mask", EPISODE_LOSS_WEIGHT_KEY}
    for key, value in data.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and value.shape[0] == batch_size
        ):
            template = value[template_index : template_index + 1]
            repeats = (pad_count,) + (1,) * (value.ndim - 1)
            padding = template.repeat(repeats)
            if key in zero_mass_keys:
                padding.zero_()
            padded[key] = torch.cat((value, padding), dim=0)
        elif isinstance(value, list) and len(value) == batch_size:
            padded[key] = list(value) + [
                copy.deepcopy(value[template_index]) for _ in range(pad_count)
            ]

    if padded["attention_mask"].shape[0] != target_batch_size:
        raise RuntimeError("PPO padding did not reach the synchronized batch size")
    return padded


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.rejection_sampling is not None:
                rs = config.rejection_sampling
                logger.info(
                    f"  Rejection sampling: level={rs.level}, metric={rs.metric}, "
                    f"action={rs.action}, upper={rs.upper}"
                    + (f", lower={rs.lower}" if rs.lower is not None else "")
                    + (f", agg={rs.agg}" if rs.level == "sequence" else "")
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info("=" * 70)

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        return batched_call(self._compute_logp, data)

    def _compute_logp(self, data: dict[str, Any]) -> torch.Tensor | None:
        self.engine.eval()
        data = data.copy()
        for key in ROLLOUT_EPISODE_METADATA_KEYS:
            data.pop(key, None)
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(
        self,
        data: list[dict[str, Any]],
        *,
        repartition_for_ppo: bool = False,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        grouped = batched_call(self._compute_advantages, data, pass_meta=True)
        if not repartition_for_ppo:
            return grouped

        # Stage each complete prompt group once before creating thousands of
        # singleton views. The RPC layer can then remotize the rows without a
        # per-field, per-row GPU-to-CPU transfer.
        grouped = tensor_container_to(grouped, "cpu")
        return [_split_ppo_batch_rows(group) for group in grouped]

    def _compute_advantages(
        self, data: dict[str, Any], meta: TrajBatchMeta | None = None
    ) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Reward Scaling
        reward_score = data["rewards"]
        episode_ids, episode_returns, episode_group_sizes = [
            data.get(key) for key in ROLLOUT_EPISODE_METADATA_KEYS
        ]
        has_any_episode_return_metadata = (
            episode_returns is not None or episode_group_sizes is not None
        )
        if has_any_episode_return_metadata and not all(
            value is not None
            for value in (episode_ids, episode_returns, episode_group_sizes)
        ):
            raise ValueError(
                "whole-episode GRPO metadata must include IDs, returns, and group sizes"
            )
        has_episode_return_metadata = all(
            value is not None
            for value in (episode_ids, episode_returns, episode_group_sizes)
        )
        if has_episode_return_metadata:
            if self.reward_norm is None:
                raise ValueError(
                    "whole-episode GRPO metadata requires reward normalization"
                )
            if not torch.equal(reward_score.float(), episode_returns.float()):
                raise ValueError(
                    "rewards must equal rollout_episode_returns before normalization"
                )
            reward_score = (episode_returns + self.reward_bias) * self.reward_scaling
            if meta is None:
                raise ValueError(
                    "whole-episode GRPO requires trajectory group metadata"
                )
            reward_score, episode_loss_weights = _normalize_rollout_episode_returns(
                reward_score,
                episode_ids,
                episode_group_sizes,
                loss_mask,
                trajectory_group_sizes=meta.traj_group_sizes,
                normalizer=self.reward_norm,
            )
            data[EPISODE_LOSS_WEIGHT_KEY] = episode_loss_weights
            # Episode returns are normalized before clipping so large game-score
            # scales cannot collapse every member of the group to reward_clip.
            reward_score = torch.clip(
                reward_score, max=self.reward_clip, min=-self.reward_clip
            )
        else:
            if episode_ids is not None:
                data[EPISODE_LOSS_WEIGHT_KEY] = _build_equal_episode_loss_weights(
                    episode_ids, loss_mask
                )
            reward_score = (reward_score + self.reward_bias) * self.reward_scaling
            reward_score = torch.clip(
                reward_score, max=self.reward_clip, min=-self.reward_clip
            )
        # Use actual trajectory group sizes when available so group-level
        # normalization handles failed/filtered rollout samples without slicing
        # across prompts. Direct calls without batched metadata keep the legacy
        # fixed-group-size behavior.
        group_sizes = meta.traj_group_sizes if meta is not None else None
        if self.reward_norm and not has_episode_return_metadata:
            reward_score = self.reward_norm(reward_score, group_sizes=group_sizes)

        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        if self.config.kl_logprob_source == "proximal":
            kl_policy_logp = data.get("prox_logp")
            if kl_policy_logp is None:
                raise ValueError(
                    "prox_logp is required when kl_logprob_source='proximal'"
                )
            kl_policy_logp = kl_policy_logp * loss_mask
        else:
            kl_policy_logp = old_logp
        rewards = -self.kl_ctl * self.kl_estimator(kl_policy_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            # Use the same actual trajectory group sizes as reward normalization;
            # ignored when adv_norm is batch-level.
            advantages = self.adv_norm(advantages, loss_mask, group_sizes=group_sizes)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp
        for key in ROLLOUT_EPISODE_METADATA_KEYS:
            data.pop(key, None)

        return data

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(
        self,
        data: list[dict[str, Any]],
        *,
        _ppo_target_batch_size: int | None = None,
    ) -> None:
        batched_call(
            functools.partial(
                self._ppo_update,
                target_batch_size=_ppo_target_batch_size,
            ),
            data,
            unpack=False,
        )

    def _ppo_update(
        self,
        data: dict[str, Any],
        *,
        target_batch_size: int | None = None,
    ) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        # Pure-DP RPC workers may deliberately keep the complete advantages
        # batch on CPU to avoid duplicating it on CUDA before minibatch
        # construction.  Stats are reduced with the engine's NCCL DP group,
        # so retain only the comparatively small tracker tensors on CUDA.
        def _stats_tensor(value: torch.Tensor) -> torch.Tensor:
            return value.to(self.engine.device, non_blocking=True)

        reward_score_stats = _stats_tensor(reward_score)
        seqlens_stats = _stats_tensor(seqlens)
        loss_mask_stats = _stats_tensor(loss_mask)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score_stats > 0).bool(),
            "incorrect_n_seqs": (reward_score_stats <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = _stats_tensor((data["begin_of_trajectory"] > 0).bool())
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score_stats, dtype=torch.bool),
            n_tokens=_stats_tensor(infer_token_denominator(data, loss_mask)),
            n_valid_tokens=loss_mask_stats.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens_stats.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens_stats.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=_stats_tensor(data["advantages"]),
            kl_rewards=_stats_tensor(data["kl_rewards"]),
            final_reward=_stats_tensor(data["tot_rewards"]),
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens_stats == attn_mask.shape[-1]).float(),
            task_reward=reward_score_stats.float(),
            prompt_len=_stats_tensor(prompt_lens).float(),
            seq_len=seqlens_stats.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.rejection_sampling is not None:
            rs = self.config.rejection_sampling
            scalars["rs_upper"] = rs.upper
            if rs.lower is not None:
                scalars["rs_lower"] = rs.lower
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{
                    k: _stats_tensor(data[k]).float()
                    for k in self.config.log_agent_stats_keys
                },
                denominator="agent",
            )
        ########## Logging code ends ##########

        del (
            global_denominators,
            result_denominators,
            reward_score_stats,
            seqlens_stats,
            loss_mask_stats,
            stats,
            seq_stats,
        )

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept if needed for approximation/metrics in loss function
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        # Synchronized sequence packing may raise its required micro-batch
        # count to another DP rank's value. Equal row counts guarantee that
        # every rank can form that many non-empty micro-batches, even when the
        # post-advantage sequence total is not divisible by DP size.
        local_batch_size = int(data["attention_mask"].shape[0])
        synchronized_batch_size = (
            local_batch_size if target_batch_size is None else int(target_batch_size)
        )
        if synchronized_batch_size < local_batch_size:
            raise ValueError(
                "PPO synchronized batch size cannot be smaller than the local "
                f"batch: target={synchronized_batch_size} local={local_batch_size}"
            )
        if synchronized_batch_size > local_batch_size:
            logger.info(
                "PPO_SEQUENCE_PADDING local_rows=%d target_rows=%d padding_rows=%d",
                local_batch_size,
                synchronized_batch_size,
                synchronized_batch_size - local_batch_size,
            )
        data = _pad_ppo_batch_to_size(data, synchronized_batch_size)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        eps_clip=self.config.eps_clip,
                        eps_clip_higher=self.config.eps_clip_higher,
                        c_clip=self.config.c_clip,
                        rejection_sampling=self.config.rejection_sampling,
                        m2_threshold=self.m2_threshold,
                        importance_sampling_level=self.config.importance_sampling_level,
                        current_version=current_version,
                        prox_logp_method=self.config.prox_logp_method,
                        use_sapo_loss=self.config.use_sapo_loss,
                        sapo_tau_pos=self.config.sapo_tau_pos,
                        sapo_tau_neg=self.config.sapo_tau_neg,
                        use_cispo_loss=self.config.use_cispo_loss,
                        use_decoupled_loss=self.config.use_decoupled_loss,
                    ),
                    loss_weight_fn=_actor_loss_weight,
                )
                stats_tracker.scalar(**train_stat)


class PPOActorController(TrainController):
    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_logp", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def compute_advantages(self, *args, **kwargs):
        repartition_for_ppo = bool(kwargs.get("repartition_for_ppo", False))
        result = self._custom_function_call(
            "compute_advantages", *args, rpc_meta={"broadcast": True}, **kwargs
        )
        if repartition_for_ppo:
            return _flatten_ppo_advantage_groups(result)
        return result

    def ppo_update(self, *args, **kwargs) -> None:
        self._custom_ppo_function_call(
            "ppo_update", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class PPOActorControllerV2(GatewayTrainController):
    def compute_logp(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/actor/compute_logp", payload)

    def compute_advantages(self, *args, **kwargs):
        repartition_for_ppo = bool(kwargs.get("repartition_for_ppo", False))
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        result = self._gateway_post_result("/ppo/actor/compute_advantages", payload)
        if repartition_for_ppo:
            return _flatten_ppo_advantage_groups(result)
        return result

    def ppo_update(self, *args, **kwargs) -> None:
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post("/ppo/actor/update", payload)


def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_cispo_loss: bool = False,
    use_decoupled_loss: bool = False,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
    vocab_mean_logits: torch.Tensor | None = None,
    vocab_norm_logits: torch.Tensor | None = None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    episode_loss_weights = input_data.get(EPISODE_LOSS_WEIGHT_KEY)
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    entropy = entropy.detach()

    if ProxLogpMethod(prox_logp_method) == ProxLogpMethod.REUSE_TRAIN_LOGP:
        prox_logp_gt = logprobs.detach()

    # Resolve proximal log-probabilities based on method
    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # Apply M2PO masking if threshold is set
    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    # Use CISPO, SAPO, or PPO loss
    if use_cispo_loss:
        if episode_loss_weights is not None:
            raise ValueError("whole-episode GRPO reduction does not support CISPO")
        if use_sapo_loss:
            raise ValueError(
                "CISPO and SAPO are mutually exclusive surrogates. "
                "Set at most one of use_cispo_loss / use_sapo_loss."
            )
        if importance_sampling_level != "token":
            raise ValueError(
                "CISPO only supports importance_sampling_level='token'. "
                "Sequence-level (GSPO-style) CISPO has no published surrogate."
            )
        loss, stat = cispo_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=prox_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            old_logprobs=old_logp,
            rejection_sampling=rejection_sampling,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    elif use_sapo_loss:
        if episode_loss_weights is not None:
            raise ValueError("whole-episode GRPO reduction does not support SAPO")
        if use_decoupled_loss:
            raise ValueError(
                "SAPO is not compatible with `use_decoupled_loss=True`. "
                "Please set `actor.use_decoupled_loss=false` in your configuration."
            )
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=c_clip,
            proximal_logprobs=prox_logp,
            rejection_sampling=rejection_sampling,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
            loss_reduction_weights=episode_loss_weights,
        )

    # Joint Distillation KL Loss
    teacher_logp = input_data.get("teacher_logp")
    rkl_stat = None
    if teacher_logp is not None:
        if episode_loss_weights is not None:
            raise ValueError(
                "whole-episode GRPO reduction does not support teacher loss"
            )
        # Coefficients for RL and Knowledge Distillation
        rl_loss_weight = input_data.get("rl_loss_weight", 1.0)
        distill_loss_weight = input_data.get("distill_loss_weight", 0.005)

        teacher_logp = (
            teacher_logp.detach()
        )  # detach to prevent gradient backprop to teacher

        if rl_loss_weight == 0:
            # Pure KD using reverse KL (importance-sampling)
            rkl_reward = teacher_logp - logprobs.detach()
            importance_weight = torch.exp(logprobs - old_logp)

            rkl_weighted_term = importance_weight * rkl_reward * loss_mask

            kd_coef = -1 * distill_loss_weight
            loss = kd_coef * rkl_weighted_term.sum() / loss_mask.sum().clamp(min=1)

            rkl_stat = -1 * rkl_weighted_term
        else:
            # KDRL: Knowledge Distillation + Reinforcement Learning (joint loss)
            rkl_penalty_per_token = (logprobs - teacher_logp) * loss_mask
            rkl_penalty = rkl_penalty_per_token.sum() / loss_mask.sum().clamp(min=1)

            loss = rl_loss_weight * loss + distill_loss_weight * rkl_penalty

            rkl_stat = rkl_penalty_per_token

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, loss_mask),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    if rkl_stat is not None:
        stats_tracker.stat(
            rkl_loss=rkl_stat,
            denominator="n_valid_tokens",
        )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    if "filtered_fraction" in stat:
        stats_tracker.scalar(rs_filtered_fraction=stat["filtered_fraction"])

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    # Log SAPO-specific statistics
    if use_sapo_loss:
        stats_tracker.stat(
            sapo_soft_gate=stat["sapo_soft_gate"],
            sapo_scaled_gate_pos=stat["sapo_scaled_gate_pos"],
            sapo_scaled_gate_neg=stat["sapo_scaled_gate_neg"],
            denominator="n_valid_tokens",
        )
    else:
        # Log clipping statistics (PPO only)
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )

    # Log proximal approximation metrics
    compute_logp_mask = stat.get("behave_mask", loss_mask)
    _log_proximal_approximation_stats(
        prox_logp_method=prox_logp_method,
        prox_logp_gt=prox_logp_gt,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
        compute_logp_mask=compute_logp_mask,
    )

    # Log version staleness metrics
    if "versions" in input_data and current_version is not None:
        version_metrics_mask = stat.get("behave_mask", loss_mask)
        _log_version_staleness_stats(
            versions=input_data["versions"],
            current_version=current_version,
            version_metrics_mask=version_metrics_mask,
        )

    return loss


# =============================================================================
# Core Functions
# =============================================================================


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, metrics).
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (should be detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor.

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


def _apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance.

    Args:
        old_logp: Behavior policy log-probabilities.
        prox_logp: Proximal policy log-probabilities.
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering.

    Returns:
        Updated loss mask with M2PO filtering applied.
    """
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask


# =============================================================================
# Logging Helper Functions
# =============================================================================

_EPSILON = 1e-8  # Small constant for numerical stability in relative error calculations


def _compute_importance_weight(
    logp_numerator: torch.Tensor,
    logp_denominator: torch.Tensor,
) -> torch.Tensor:
    """Compute importance weight as exp(logp_num - logp_denom)."""
    return torch.exp(logp_numerator - logp_denominator).float()


def _compute_approximation_errors(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute error metrics between ground truth and approximation.

    Returns:
        Dictionary with abs_error, rel_error, and squared_error tensors.
    """
    diff = ground_truth - approximation
    abs_error = torch.abs(diff).float()
    rel_error = torch.abs(diff / (torch.abs(ground_truth) + _EPSILON)).float()
    squared_error = (diff * diff).float()
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": squared_error,
    }


def _tensor_scalar_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Compute scalar statistics (avg, max, min) for a tensor.

    Args:
        tensor: Input tensor to compute statistics on.

    Returns:
        Dictionary with avg, max, min as Python floats.
    """
    t = tensor.float()
    return {
        "avg": t.mean().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def _log_approximation_metrics_for_method(
    method_name: str,
    approx_logp: torch.Tensor,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    prox_logp_gt: torch.Tensor | None = None,
) -> None:
    """
    Log metrics for a single approximation method.

    Args:
        method_name: Name of the approximation method (e.g., "loglinear").
        approx_logp: Approximated proximal log-probabilities.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities.
        prox_logp_gt: Ground truth proximal logp, or None if unavailable.
    """
    # Compute importance weights from approximation
    behave_imp_weight = _compute_importance_weight(approx_logp, old_logp)
    importance_weight = _compute_importance_weight(logprobs, approx_logp)

    metrics = {
        f"{method_name}/approx_logp": approx_logp.float(),
        f"{method_name}/behave_imp_weight": behave_imp_weight,
        f"{method_name}/importance_weight": importance_weight,
    }

    # Add error metrics if ground truth is available
    if prox_logp_gt is not None:
        # Log-probability errors
        logp_errors = _compute_approximation_errors(prox_logp_gt, approx_logp)
        metrics.update(
            {
                f"{method_name}/abs_error": logp_errors["abs_error"],
                f"{method_name}/rel_error": logp_errors["rel_error"],
                f"{method_name}/squared_error": logp_errors["squared_error"],
            }
        )

        # Ground truth importance weights for comparison
        behave_imp_weight_gt = _compute_importance_weight(prox_logp_gt, old_logp)
        importance_weight_gt = _compute_importance_weight(logprobs, prox_logp_gt)

        # Importance weight errors
        behave_errors = _compute_approximation_errors(
            behave_imp_weight_gt, behave_imp_weight
        )
        imp_errors = _compute_approximation_errors(
            importance_weight_gt, importance_weight
        )

        metrics.update(
            {
                f"{method_name}/behave_imp_weight_abs_error": behave_errors[
                    "abs_error"
                ],
                f"{method_name}/behave_imp_weight_rel_error": behave_errors[
                    "rel_error"
                ],
                f"{method_name}/importance_weight_abs_error": imp_errors["abs_error"],
                f"{method_name}/importance_weight_rel_error": imp_errors["rel_error"],
            }
        )

    stats_tracker.stat(**metrics, denominator="n_valid_tokens")


def _log_proximal_approximation_stats(
    prox_logp_method: str,
    prox_logp_gt: torch.Tensor | None,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
    compute_logp_mask: torch.Tensor,
) -> None:
    """
    Log proximal policy approximation metrics based on the method.

    Args:
        prox_logp_method: The proximal logp method being used.
        prox_logp_gt: Ground truth proximal logp, or None if skipped.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.
        compute_logp_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("compute_logp"):
        stats_tracker.denominator(n_valid_tokens=compute_logp_mask.bool())

        # Log ground truth when available
        if prox_logp_gt is not None:
            stats_tracker.stat(
                prox_logp_gt=prox_logp_gt.float(),
                denominator="n_valid_tokens",
            )

        # Skip if versions not available
        if versions is None or current_version is None:
            return

        if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
            # Loglinear mode: log approximation without error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=None,  # No ground truth in loglinear mode
                )

        elif prox_logp_method == PROX_LOGP_METHOD_METRICS and prox_logp_gt is not None:
            # Metrics mode: compute all methods with error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=None,  # Compute all methods
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=prox_logp_gt,
                )

        if logprobs is not None:
            # Log KL divergence estimators to check for policy drift between the
            # training-time policy (logprobs) and the inference-time policy (old_logp).
            log_ratio = (logprobs.float() - old_logp.float()).detach()

            # Implementation of different estimators for KL divergence.
            # See: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl
            kl_div_estimator_direct = -log_ratio
            kl_div_estimator_taylor = log_ratio**2 / 2.0
            kl_div_estimator_dual = log_ratio.exp() - 1 - log_ratio

            # Register these to TensorBoard
            stats_tracker.stat(
                kl_div_direct=kl_div_estimator_direct,
                kl_div_taylor=kl_div_estimator_taylor,
                kl_div_dual=kl_div_estimator_dual,
                denominator="n_valid_tokens",
            )


def _log_version_staleness_stats(
    versions: torch.Tensor,
    current_version: int,
    version_metrics_mask: torch.Tensor,
) -> None:
    """
    Log sample staleness metrics based on policy versions.

    Args:
        versions: Per-token policy versions from rollout.
        current_version: Current training version.
        version_metrics_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("version_stats"):
        stats_tracker.denominator(n_valid_tokens=version_metrics_mask.bool())

        v_proximal = current_version - 1
        v_theta = current_version
        v_behave = versions.float()

        # Filter to generated tokens only (version >= 0)
        valid_generated_mask = version_metrics_mask & (versions >= 0)

        if not valid_generated_mask.any():
            return

        # Compute staleness for valid tokens
        staleness_proximal = (v_proximal - v_behave)[valid_generated_mask]
        staleness_theta = (v_theta - v_behave)[valid_generated_mask]

        # Compute and log statistics
        proximal_stats = _tensor_scalar_stats(staleness_proximal)
        theta_stats = _tensor_scalar_stats(staleness_theta)

        stats_tracker.scalar(
            sample_staleness_proximal_avg=proximal_stats["avg"],
            sample_staleness_proximal_max=proximal_stats["max"],
            sample_staleness_proximal_min=proximal_stats["min"],
            sample_staleness_theta_avg=theta_stats["avg"],
            sample_staleness_theta_max=theta_stats["max"],
            sample_staleness_theta_min=theta_stats["min"],
            v_theta=v_theta,
            v_proximal=v_proximal,
            n_valid_generated_tokens=valid_generated_mask.sum().item(),
        )

import json

import pytest
import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.infra.rpc.rtensor import RTensor
from areal.trainer.rl_trainer import _audit_pacman_logprob_alignment
from areal.utils.functional import apply_rejection_sampling


def make_trajectory(
    actor_delta: float = 0.0, behavior_logp_delta: float = 0.0
):
    # Prompt rows 0/1; sampled tokens on rows 2/3. The actor/ref tensors are
    # already causal-position aligned, while rollout metadata is token aligned.
    return {
        "input_ids": torch.tensor([[10, 11, 20, 21]]),
        "loss_mask": torch.tensor([[0, 0, 1, 1]]),
        "logprobs": torch.tensor([[0.0, 0.0, -0.25, 0.0]]),
        "prox_logp": torch.tensor(
            [[0.0, -0.25 + actor_delta + behavior_logp_delta, 0.0, 0.0]]
        ),
        "ref_logp": torch.tensor(
            [[0.0, -0.25 + behavior_logp_delta, 0.0, 0.0]]
        ),
        "versions": torch.tensor([[-1, -1, 0, 0]]),
        "pacman_allowed_token_ids": torch.tensor(
            [[[0, 0], [0, 0], [21, 31], [22, 0]]]
        ),
    }


def make_seven_token_trajectory(branch_log_ratio: float = 0.0):
    sampled = list(range(20, 27))
    length = 2 + len(sampled)
    logprobs = torch.zeros(1, length)
    logprobs[0, 2:] = -0.25
    logprobs[0, -1] = 0.0
    aligned = torch.zeros(1, length)
    aligned[0, 1:7] = -0.25 + branch_log_ratio
    aligned[0, 7] = 0.0
    supports = torch.zeros(1, length, 2, dtype=torch.long)
    for offset, token_id in enumerate(sampled):
        token_row = 2 + offset
        supports[0, token_row, 0] = token_id + 1
        if offset < 6:
            supports[0, token_row, 1] = token_id + 101
    return {
        "input_ids": torch.tensor([[10, 11, *sampled]]),
        "loss_mask": torch.tensor([[0, 0, *([1] * 7)]]),
        "logprobs": logprobs,
        "prox_logp": aligned.clone(),
        "ref_logp": aligned.clone(),
        "versions": torch.tensor([[-1, -1, *([0] * 7)]]),
        "pacman_allowed_token_ids": supports,
    }


def test_pacman_logprob_alignment_writes_exact_support_report(tmp_path) -> None:
    report = _audit_pacman_logprob_alignment(
        [make_trajectory()], output_dir=str(tmp_path), global_step=0
    )

    assert report["aligned"]
    assert report["tokens"] == 2
    assert report["support_sizes"] == [1, 2]
    stored = json.loads(
        (tmp_path / "global-step-000000.json").read_text(encoding="utf-8")
    )
    assert stored["records"][0]["sampled_token_id"] == 20
    assert stored["records"][0]["allowed_token_ids"] == [20, 30]


def test_pacman_logprob_alignment_fails_closed_on_delta(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="exceeded exact acceptance limits"):
        _audit_pacman_logprob_alignment(
            [make_trajectory(actor_delta=0.1)],
            output_dir=str(tmp_path),
            global_step=0,
        )


def test_pacman_logprob_behavior_mode_reports_nonexact_acceptance(tmp_path) -> None:
    report = _audit_pacman_logprob_alignment(
        [make_trajectory(behavior_logp_delta=0.03)],
        output_dir=str(tmp_path),
        global_step=0,
        acceptance_mode="behavior",
    )

    assert report["accepted"]
    assert not report["aligned"]
    assert report["behavior_compatible"]
    assert report["behavior_metrics"]["branching_tokens"] == 1
    assert report["behavior_metrics"]["singleton_tokens"] == 1
    assert report["behavior_metrics"]["option_count"] == 1
    assert report["options"][0]["token_count"] == 2
    assert report["options"][0]["support_sizes"] == [2, 1]
    assert report["behavior_metrics"]["branching_max_importance_ratio"] == (
        pytest.approx(torch.exp(torch.tensor(0.03)).item())
    )


def test_pacman_logprob_behavior_mode_fails_outside_ratio_bound(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="exceeded behavior acceptance limits"):
        _audit_pacman_logprob_alignment(
            [make_trajectory(behavior_logp_delta=0.3)],
            output_dir=str(tmp_path),
            global_step=0,
            acceptance_mode="behavior",
        )


def test_pacman_logprob_behavior_mode_accepts_bounded_masked_fraction(
    tmp_path,
) -> None:
    trajectories = [make_trajectory(behavior_logp_delta=0.3)]
    trajectories.extend(make_trajectory() for _ in range(100))

    report = _audit_pacman_logprob_alignment(
        trajectories,
        output_dir=str(tmp_path),
        global_step=0,
        acceptance_mode="behavior",
    )

    assert report["accepted"]
    assert report["rejection_sampling_contract"] == {
        "level": "sequence",
        "action": "mask",
        "metric": "ratio",
        "agg": "sum",
        "lower": 0.8,
        "upper": 1.25,
    }
    behavior = report["behavior_metrics"]
    assert behavior["would_filter_token_count"] == 1
    assert behavior["would_filter_fraction"] == pytest.approx(1 / 202)
    assert behavior["would_filter_option_count"] == 1
    assert behavior["would_filter_option_fraction"] == pytest.approx(1 / 101)
    assert (
        behavior["would_filter_option_fraction"]
        < behavior["max_filtered_fraction"]
    )


def test_pacman_behavior_audit_matches_runtime_sequence_mask(tmp_path) -> None:
    # Each of the six branching-token ratios is 1.04 and individually inside
    # [0.8, 1.25], while their option joint ratio 1.04**6 exceeds 1.25.
    trajectories = [make_seven_token_trajectory(float(torch.log(torch.tensor(1.04))))]
    trajectories.extend(make_seven_token_trajectory() for _ in range(14))
    report = _audit_pacman_logprob_alignment(
        trajectories,
        output_dir=str(tmp_path),
        global_step=0,
        acceptance_mode="behavior",
        expected_option_tokens=7,
    )

    proximal = torch.stack(
        [trajectory["prox_logp"][0, 1:8] for trajectory in trajectories]
    )
    rollout = torch.stack(
        [trajectory["logprobs"][0, 2:9] for trajectory in trajectories]
    )
    runtime = apply_rejection_sampling(
        proximal_logprobs=proximal,
        old_logprobs=rollout,
        loss_mask=torch.ones_like(proximal),
        cu_seqlens=None,
        config=RejectionSamplingConfig(
            level="sequence",
            action="mask",
            metric="ratio",
            agg="sum",
            lower=0.8,
            upper=1.25,
        ),
    )
    audit_rejected = {
        (option["trajectory_index"], option["sample_index"])
        for option in report["options"]
        if not 0.8 <= option["behavior_importance_ratio"] <= 1.25
    }
    runtime_rejected = {
        (index, 0)
        for index, row in enumerate(runtime.loss_mask)
        if not row.bool().any()
    }
    assert audit_rejected == runtime_rejected == {(0, 0)}
    assert runtime.filtered_fraction == pytest.approx(1 / 15)
    assert report["behavior_metrics"]["would_filter_token_count"] == 0


def test_pacman_behavior_gate_uses_joint_option_ratio_not_token_mean(
    tmp_path,
) -> None:
    trajectory = make_seven_token_trajectory()
    # Six branching tokens alternate between exp(+/-0.2). Their token-level
    # mean absolute log-ratio is 0.2, but the complete option joint ratio is 1.
    deltas = torch.tensor([0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
    trajectory["prox_logp"][0, 1:7] += deltas
    trajectory["ref_logp"][0, 1:7] += deltas

    report = _audit_pacman_logprob_alignment(
        [trajectory],
        output_dir=str(tmp_path),
        global_step=0,
        acceptance_mode="behavior",
        expected_option_tokens=7,
    )

    behavior = report["behavior_metrics"]
    assert report["accepted"]
    assert behavior["branching_mean_abs_log_ratio"] == pytest.approx(0.2)
    assert not behavior["branching_mean_abs_log_ratio_within_diagnostic_tolerance"]
    assert behavior["would_filter_option_count"] == 0
    assert report["options"][0]["behavior_importance_ratio"] == pytest.approx(1.0)


def test_pacman_behavior_option_fraction_boundary_is_fail_closed(tmp_path) -> None:
    at_limit = [make_trajectory(behavior_logp_delta=0.3) for _ in range(7)]
    at_limit.extend(make_trajectory() for _ in range(93))
    report = _audit_pacman_logprob_alignment(
        at_limit,
        output_dir=str(tmp_path / "at-limit"),
        global_step=0,
        acceptance_mode="behavior",
    )
    assert report["behavior_metrics"]["would_filter_option_fraction"] == pytest.approx(
        0.07
    )

    above_limit = [make_trajectory(behavior_logp_delta=0.3) for _ in range(8)]
    above_limit.extend(make_trajectory() for _ in range(92))
    with pytest.raises(RuntimeError, match="exceeded behavior acceptance limits"):
        _audit_pacman_logprob_alignment(
            above_limit,
            output_dir=str(tmp_path / "above-limit"),
            global_step=0,
            acceptance_mode="behavior",
        )
    failed = json.loads(
        (tmp_path / "above-limit" / "global-step-000000.json").read_text()
    )
    assert failed["behavior_metrics"]["branching_mean_abs_log_ratio"] < 0.05
    assert failed["behavior_metrics"]["would_filter_option_fraction"] > 0.07


def test_pacman_logprob_behavior_mode_rejects_nonzero_singleton(tmp_path) -> None:
    trajectory = make_trajectory()
    trajectory["logprobs"][0, 3] = -0.1
    trajectory["prox_logp"][0, 2] = -0.1
    trajectory["ref_logp"][0, 2] = -0.1
    with pytest.raises(RuntimeError, match="exceeded behavior acceptance limits"):
        _audit_pacman_logprob_alignment(
            [trajectory],
            output_dir=str(tmp_path),
            global_step=0,
            acceptance_mode="behavior",
        )


def test_pacman_logprob_learned_mode_does_not_require_reference_match(
    tmp_path,
) -> None:
    trajectory = make_trajectory()
    trajectory["logprobs"][0, 2] = -0.05
    trajectory["prox_logp"][0, 1] = -0.05
    report = _audit_pacman_logprob_alignment(
        [trajectory],
        output_dir=str(tmp_path),
        global_step=1,
        acceptance_mode="behavior",
        require_actor_reference_alignment=False,
    )

    assert report["accepted"]
    assert not report["aligned"]
    assert not report["require_actor_reference_alignment"]


def test_pacman_logprob_alignment_localizes_rtensors(tmp_path) -> None:
    trajectory = make_trajectory()
    trajectory = {
        key: RTensor(shard=None, data=value)
        for key, value in trajectory.items()
    }

    report = _audit_pacman_logprob_alignment(
        [trajectory], output_dir=str(tmp_path), global_step=0
    )

    assert report["aligned"]
    assert report["tokens"] == 2


def test_pacman_logprob_alignment_rejects_missing_branch(tmp_path) -> None:
    trajectory = make_trajectory()
    trajectory["pacman_allowed_token_ids"][0, 2] = torch.tensor([21, 0])
    with pytest.raises(RuntimeError, match="requires branching supports"):
        _audit_pacman_logprob_alignment(
            [trajectory], output_dir=str(tmp_path), global_step=0
        )


def test_pacman_logprob_alignment_allows_branching_without_singletons(tmp_path) -> None:
    trajectory = make_trajectory()
    trajectory["pacman_allowed_token_ids"][0, 3] = torch.tensor([22, 23])
    report = _audit_pacman_logprob_alignment(
        [trajectory], output_dir=str(tmp_path), global_step=0
    )
    assert report["accepted"]
    assert report["support_sizes"] == [2]
    assert report["behavior_metrics"]["branching_tokens"] == 2
    assert report["behavior_metrics"]["singleton_tokens"] == 0
    assert report["behavior_metrics"]["singleton_max_abs_logp"] == 0.0


def test_pacman_logprob_alignment_checks_expected_option_width(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="unexpected option token count"):
        _audit_pacman_logprob_alignment(
            [make_trajectory()],
            output_dir=str(tmp_path),
            global_step=0,
            expected_option_tokens=7,
        )

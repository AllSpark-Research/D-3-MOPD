# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
from collections.abc import Callable

import torch

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.utils.types import RolloutBatch


def _cat_log_probs(batch: RolloutBatch, key: str, *, device: torch.device) -> torch.Tensor:
    values = batch.get(key)
    if not values:
        raise ValueError(f"GKD OPD requires batch['{key}'], but it is missing or empty.")
    return torch.cat(values, dim=0).to(device=device, dtype=torch.float32)


def sampled_forward_kl_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sampled forward-KL / weighted CE loss for GKD-style OPD.

    The rollout trajectory is treated as stop-gradient data. Teacher log-probs
    are collected during rollout by ``slime.rollout.on_policy_distillation``.
    For each sampled token y_t, this loss uses the importance weight

        w_t = stop(exp(log pi_teacher(y_t|s_t) - log pi_old(y_t|s_t)))

    and optimizes weighted negative log-likelihood under the current student:

        L = E[-w_t * log pi_theta(y_t|s_t)].

    This is a sampled-token approximation to forward KL, reusing the existing
    OPD SGLang data path without requiring full-vocabulary teacher logits.
    """

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    student_log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0).to(dtype=torch.float32)
    device = student_log_probs.device
    teacher_log_probs = _cat_log_probs(batch, "teacher_log_probs", device=device)

    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            "GKD OPD shape mismatch: "
            f"student_log_probs={tuple(student_log_probs.shape)}, "
            f"teacher_log_probs={tuple(teacher_log_probs.shape)}"
        )

    old_source = getattr(args, "gkd_old_logprob_source", "rollout")
    if old_source == "rollout":
        old_log_probs = _cat_log_probs(batch, "rollout_log_probs", device=device)
    elif old_source == "current":
        old_log_probs = student_log_probs.detach()
    else:
        raise ValueError(f"Unknown gkd_old_logprob_source={old_source!r}; expected 'rollout' or 'current'.")

    if old_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            "GKD OPD shape mismatch: "
            f"old_log_probs={tuple(old_log_probs.shape)}, "
            f"student_log_probs={tuple(student_log_probs.shape)}"
        )

    log_weights = (teacher_log_probs - old_log_probs).detach()
    raw_weights = torch.exp(log_weights)
    raw_weights = torch.nan_to_num(raw_weights, nan=0.0, posinf=1.0e20, neginf=0.0)

    min_weight = getattr(args, "gkd_weight_min", 0.0)
    max_weight = getattr(args, "gkd_weight_clip", 10.0)
    weights = raw_weights
    if min_weight is not None and min_weight > 0:
        weights = torch.clamp(weights, min=min_weight)
    if max_weight is not None and max_weight > 0:
        weights = torch.clamp(weights, max=max_weight)

    clipped = (weights != raw_weights).to(dtype=torch.float32)

    if getattr(args, "gkd_normalize_weights", False):
        weight_mean = torch.clamp_min(sum_of_sample_mean(weights).detach(), 1.0e-6)
        weights = weights / weight_mean

    weighted_nll = -weights * student_log_probs
    loss = sum_of_sample_mean(weighted_nll) * getattr(args, "gkd_loss_coef", 1.0)

    if student_log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    sampled_fkl = weights * (teacher_log_probs - student_log_probs)

    log = {
        "loss": loss.clone().detach(),
        "gkd_weighted_ce": sum_of_sample_mean(weighted_nll).clone().detach(),
        "gkd_sampled_fkl": sum_of_sample_mean(sampled_fkl).clone().detach(),
        "gkd_weight_mean": sum_of_sample_mean(weights).clone().detach(),
        "gkd_weight_max": weights.max().clone().detach() if weights.numel() > 0 else loss.new_tensor(0.0),
        "gkd_weight_clipfrac": sum_of_sample_mean(clipped).clone().detach(),
        "gkd_teacher_logp": sum_of_sample_mean(teacher_log_probs).clone().detach(),
        "gkd_student_logp": sum_of_sample_mean(student_log_probs).clone().detach(),
        "gkd_old_logp": sum_of_sample_mean(old_log_probs).clone().detach(),
    }
    return loss, log

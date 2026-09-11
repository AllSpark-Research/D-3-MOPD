#!/bin/bash
#
# D³-MOPD example launch script.
#
# This is a *reference* script — every path, model, and dataset below is a
# placeholder. Fill in the ``<...>`` values for your environment before running.
#
# Node layout assumed here (edit *_NNODES to reshape):
#   TRAIN_NNODES   Megatron actor nodes                (default 4)
#   ROLLOUT_NNODES SGLang student rollout nodes        (default 2)
#   TEACHER_NNODES teacher-serving nodes               (default 1)
#   → total WORLD_SIZE = TRAIN + ROLLOUT + TEACHER      (here: 7)
#
# The teacher node colocates N_TEACHERS SGLang instances (one per domain,
# each with tensor-parallel = TEACHER_TP GPUs).
#
# On a separate machine (any host with wandb + shared-fs access), launch the
# watcher after training starts — see the banner printed at the end of this
# script for the exact command.
#

set -euo pipefail

# ============================================================================
# 1. Node / GPU layout
# ============================================================================
export TRAIN_NNODES="${TRAIN_NNODES:-4}"
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-2}"
export TEACHER_NNODES="${TEACHER_NNODES:-1}"
export N_TEACHERS="${N_TEACHERS:-4}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# ============================================================================
# 2. Teacher pool (index-aligned pipe-separated lists — one entry per teacher)
# ============================================================================
# Each teacher owns one domain. Sample routing key is `metadata.data_source`
# by default; each teacher's TEACHER_DATA_SOURCES entry lists the comma-
# separated data_source values whose samples get routed to that teacher.
#
# Example wiring (4 teachers, domains = math / code / if / tool):
#   TEACHER_NAMES        = "math|code|if|tool"
#   TEACHER_DATA_SOURCES = "math_ds1,math_ds2|code_ds1|if_ds1,if_ds2|tool_ds1"
#
# Colocation on one 8-GPU node with TP=2 → 4 teachers × 2 GPUs each:
#   TEACHER_CUDA_VISIBLE_DEVICES_LIST = "0,1|2,3|4,5|6,7"
export TEACHER_MODEL_PATHS="<path/to/teacher_math_hf>|<path/to/teacher_code_hf>|<path/to/teacher_if_hf>|<path/to/teacher_tool_hf>"
export TEACHER_NAMES="math|code|if|tool"
export TEACHER_DATA_SOURCES="<math_ds1,math_ds2>|<code_ds1>|<if_ds1,if_ds2>|<tool_ds1>"
export TEACHER_CUDA_VISIBLE_DEVICES_LIST="${TEACHER_CUDA_VISIBLE_DEVICES_LIST:-0,1|2,3|4,5|6,7}"
export TEACHER_TP="${TEACHER_TP:-2}"
export TEACHER_BASE_PORT="${TEACHER_BASE_PORT:-45141}"
export TEACHER_PORT_STRIDE="${TEACHER_PORT_STRIDE:-100}"
export TEACHER_DIST_INIT_BASE_PORT="${TEACHER_DIST_INIT_BASE_PORT:-25131}"
export TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.70}"
export TEACHER_MAX_RUNNING_REQUESTS="${TEACHER_MAX_RUNNING_REQUESTS:-128}"
export TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"

# Sample-to-teacher routing key (must exist in every sample's metadata dict).
export OPD_TEACHER_ROUTE_KEY="${OPD_TEACHER_ROUTE_KEY:-data_source}"
export OPD_TEACHER_TIMEOUT_SECONDS="${OPD_TEACHER_TIMEOUT_SECONDS:-600}"

# ============================================================================
# 3. Student model + training data
# ============================================================================
export MODEL_PATH="${MODEL_PATH:-<path/to/student_hf>}"
export train_dataset="${train_dataset:-<path/to/train_merged_multi_domain.jsonl>}"
# Every row in train_dataset must have metadata.data_source matching one of the
# comma-separated values in TEACHER_DATA_SOURCES above.

# ============================================================================
# 4. Training schedule
# ============================================================================
export TRAIN_STEPS="${TRAIN_STEPS:-256}"
export rollout_batch_size="${rollout_batch_size:-128}"
export mini_batch_size="${mini_batch_size:-128}"
export num_steps_per_rollout="${num_steps_per_rollout:-4}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
export save_freq="${save_freq:-32}"
export test_freq="${test_freq:-32}"
export TRAIN_MAX_RESPONSE_LEN="${TRAIN_MAX_RESPONSE_LEN:-8192}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-8192}"
export max_prompt_len="${max_prompt_len:-16384}"

# ============================================================================
# 5. OPD loss / policy loss
# ============================================================================
export OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
export POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-cispo}"    # cispo | grpo | gspo
export CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
export DISABLE_GRPO_STD_NORMALIZATION="${DISABLE_GRPO_STD_NORMALIZATION:-False}"
export ENABLE_TIS="${ENABLE_TIS:-True}"
export TIS_CLIP="${TIS_CLIP:-2.0}"
export UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
export ROUTER_REPLAY="${ROUTER_REPLAY:-R3}"             # R3 | R2 | False (MoE-only)
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-64}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-2}"
export DISABLE_THINKING="${DISABLE_THINKING:-False}"

# ============================================================================
# 6. D³-MOPD wiring (env vars read by the plugins + watcher)
# ============================================================================
# Auto-derive DOMAIN_MAP from TEACHER_NAMES + TEACHER_DATA_SOURCES so the
# training-side per-domain logging plugin stays aligned with the watcher.
IFS='|' read -r -a _TN <<<"${TEACHER_NAMES}"
IFS='|' read -r -a _TDS <<<"${TEACHER_DATA_SOURCES}"
_DOMAIN_MAP=""
for i in "${!_TN[@]}"; do
    [[ -n "${_DOMAIN_MAP}" ]] && _DOMAIN_MAP+="|"
    _DOMAIN_MAP+="${_TN[$i]}=${_TDS[$i]}"
done
export D3MOPD_DOMAIN_MAP="${_DOMAIN_MAP}"

# Enable dynamic per-batch mixture quotas (uncomment / set to disable):
export D3MOPD_DATA_SOURCE_PATH="${D3MOPD_DATA_SOURCE_PATH:-slime_plugins.data_sources.stratified.DynamicRatioStratifiedRolloutDataSourceWithBuffer}"

# Path to the JSON the watcher writes and the data source reads:
export D3MOPD_LOGS_ROOT="${D3MOPD_LOGS_ROOT:-./logs}"
export D3MOPD_STATUS_PATH="${D3MOPD_STATUS_PATH:-${D3MOPD_LOGS_ROOT}/d3_status/status.json}"
mkdir -p "$(dirname "${D3MOPD_STATUS_PATH}")"

# Per-batch multiplicative ratio jitter, α ∈ [0, 1). 0 = strict Hare-quota;
# 0.30 restores roughly the batch-level σ of random shuffle.
export D3MOPD_RATIO_JITTER="${D3MOPD_RATIO_JITTER:-0.30}"

# Optional: rehearsal-domain gate (Patch B). Comma-separated domain names
# whose "teacher" is really the frozen student (initial_kl ≈ 0). These
# domains are pinned to the ratio floor instead of hoarding the mixture.
# export D3MOPD_REHEARSAL_DOMAINS="math"

# Static-mode downsample filter (optional). Only wire this if you use
# --mode static / static-ratio in the watcher.
# export D3MOPD_FILTER_PATH="slime_plugins.filters.d3mopd_downsample_filter.d3mopd_downsample_filter"

# ============================================================================
# 7. Experiment name / save path / wandb
# ============================================================================
export EXP_NAME="${EXP_NAME:-d3mopd_example}"
export SAVE_PATH="${SAVE_PATH:-./checkpoints/${EXP_NAME}}"
export WANDB_PROJECT="${WANDB_PROJECT:-d3mopd}"
export WANDB_GROUP="${WANDB_GROUP:-${EXP_NAME}}"
# export WANDB_API_KEY=<your_wandb_api_key>

# ============================================================================
# 8. Hand off to slime's async training entrypoint
# ============================================================================
# The remainder of the launch (Ray cluster bring-up, per-node role assignment,
# teacher SGLang server start, actor/train worker start, ...) lives in your
# organization's cluster orchestrator. The env vars above are the complete
# surface that both this training run and the companion watcher consume.
#
# For a concrete single-node quick-start (no multi-teacher, no dynamic ratio),
# see scripts/examples/run-qwen3-4B.sh in the upstream slime tree.

cat <<'BANNER'
================================================================================
D³-MOPD run configured. Start the companion watcher on any host with
wandb + shared-fs access (observer mode first — omit --write-status until
you trust the judgments):

  python tools/d3mopd/watcher.py \
      --wandb-project      "$WANDB_PROJECT" \
      --wandb-group        "$EXP_NAME" \
      --domain-map         "$D3MOPD_DOMAIN_MAP" \
      --status-path        "$D3MOPD_STATUS_PATH" \
      --mode               dynamic-composite \
      --temperature        0.5 \
      --ratio-floor        0.10 \
      --progress-rolling-k 3 \
      --update-every-steps 10 \
      --initial-kl-seed-points 5 \
      --ema-window         10 \
      --poll-interval      300 \
      --write-status
================================================================================
BANNER

# python train_async.py \
#     --model-path "${MODEL_PATH}" \
#     --prompt-data "${train_dataset}" \
#     --data-source-path "${D3MOPD_DATA_SOURCE_PATH}" \
#     --custom-rm-path slime.rollout.multi_teacher_distillation.reward_func \
#     --custom-reward-post-process-path slime.rollout.multi_teacher_distillation.post_process_rewards \
#     --custom-log-rollout-function-path slime_plugins.logging.d3mopd_rollout_log.log_rollout_data \
#     ...  # (fill in the remaining slime / megatron / sglang flags for your cluster)

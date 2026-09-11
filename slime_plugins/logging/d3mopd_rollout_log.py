"""Per-domain rollout logger for D³-MOPD.

Rolls per-``data_source`` sample counts and reward scores up into per-domain
wandb metrics (``rollout/domain_count/{domain}``, ``rollout/domain_ratio/{domain}``,
``rollout/domain_score/{domain}``) using the ``data_source -> domain`` mapping
in the ``D3MOPD_DOMAIN_MAP`` env var (same string format as elsewhere:
``domain=ds1,ds2|domain=ds3,ds4|...``).

Wire it in via slime's ``--custom-log-rollout-function-path`` flag, pointing at
``slime_plugins.logging.d3mopd_rollout_log.log_rollout_data``.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from numbers import Real
from typing import Any

logger = logging.getLogger(__name__)


def _load_domain_map() -> dict[str, str]:
    """Parse ``D3MOPD_DOMAIN_MAP`` env into a flat data_source -> domain dict.

    Env format (typically set by the launch script from ``TEACHER_NAMES`` +
    ``TEACHER_DATA_SOURCES``):
        ``"math=math_dapo,acereason_math|code=simulation__codeio,...|if=..."``
    Returns an empty dict when the env is unset or malformed — caller must
    treat that as "no domain aggregation available, skip domain-level emit".
    """
    env = os.environ.get("D3MOPD_DOMAIN_MAP", "")
    if not env:
        return {}
    ds_to_domain: dict[str, str] = {}
    for entry in env.split("|"):
        if "=" not in entry:
            continue
        domain, ds_csv = entry.split("=", 1)
        domain = domain.strip()
        if not domain:
            continue
        for ds in ds_csv.split(","):
            ds = ds.strip()
            if ds:
                ds_to_domain[ds] = domain
    return ds_to_domain


def _extract_score(reward: Any) -> float | None:
    if isinstance(reward, dict):
        score = reward.get("score")
        if isinstance(score, Real):
            return float(score)
        return None

    if isinstance(reward, Real):
        return float(reward)

    return None


def _get_rollout_step(args, rollout_id: int) -> int:
    from slime.ray.rollout import compute_rollout_step

    return compute_rollout_step(args, rollout_id)


def _log_metrics(args, log_dict: dict[str, float]) -> None:
    from slime.utils import logging_utils

    logging_utils.log(args, log_dict, step_key="rollout/step")


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    scores_by_data_source = defaultdict(list)
    counts_by_data_source: dict[str, int] = defaultdict(int)

    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        reward_spec = metadata.get("reward_spec") if isinstance(metadata.get("reward_spec"), dict) else {}
        data_source = reward_spec.get("data_source")

        if not data_source:
            continue

        # Count samples per data_source (independent of reward extraction) so we
        # can roll up to domain-level batch composition, including the real
        # ratios after dynamic-mixture adjustments kick in.
        counts_by_data_source[str(data_source)] += 1

        score = _extract_score(getattr(sample, "reward", None))
        if score is None:
            continue

        scores_by_data_source[str(data_source)].append(score)

    if not scores_by_data_source and not counts_by_data_source:
        return False

    log_dict = {
        f"rollout/data_source_score/{data_source}": sum(scores) / len(scores)
        for data_source, scores in scores_by_data_source.items()
    }
    # Roll per-data_source counts and reward scores up into per-domain metrics
    # (e.g. math / code / if / tool), so wandb stays aligned with the training
    # setup's N-teacher / N-domain mental model. Falls back to no-op when the
    # env that provides the ds -> domain map isn't set.
    ds_to_domain = _load_domain_map()
    if ds_to_domain and counts_by_data_source:
        domain_counts: dict[str, int] = defaultdict(int)
        for ds, count in counts_by_data_source.items():
            domain = ds_to_domain.get(ds)
            if domain:
                domain_counts[domain] += count
        total_domain_counted = sum(domain_counts.values())
        for domain, count in domain_counts.items():
            log_dict[f"rollout/domain_count/{domain}"] = float(count)
            log_dict[f"rollout/domain_ratio/{domain}"] = count / max(total_domain_counted, 1)
    if ds_to_domain and scores_by_data_source:
        # Count-weighted per-domain mean reward (so a 3773-sample data_source
        # weighs more than a 227-sample one within the same domain).
        domain_score_num: dict[str, float] = defaultdict(float)
        domain_score_den: dict[str, int] = defaultdict(int)
        for ds, scores in scores_by_data_source.items():
            domain = ds_to_domain.get(ds)
            if not domain:
                continue
            domain_score_num[domain] += sum(scores)
            domain_score_den[domain] += len(scores)
        for domain, den in domain_score_den.items():
            if den > 0:
                log_dict[f"rollout/domain_score/{domain}"] = domain_score_num[domain] / den
    log_dict["rollout/step"] = _get_rollout_step(args, rollout_id)

    logger.info(f"d3mopd rollout {rollout_id}: {log_dict}")
    _log_metrics(args, log_dict)
    return False

#!/usr/bin/env python3
"""D³-MOPD watcher.

Polls wandb for per-domain training metrics, judges convergence (KL plateau AND
reward stagnation), and (optionally) writes a status JSON that the D³-MOPD
data source / filter consumes to adjust per-domain mixture ratios or to
downsample saturated domains.

Combined observer/writer modes in one script. Default is observer-only
(``--write-status=False``) — runs the same judgment logic and prints every
poll's verdict so you can eyeball whether (W, ε, K) are correct before any
downsampling actually triggers. Add ``--write-status`` to flip into writer mode.

Watcher is an external process — restart-safe (no state on disk), and the
training side does not care whether it is up.

Typical invocation::

    python tools/d3mopd/watcher.py \\
        --wandb-project d3mopd \\
        --wandb-group <EXP_NAME> \\
        --status-path ./logs/d3_status/<EXP_NAME>_status.json \\
        # add --write-status after you trust the judgments
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb

logger = logging.getLogger("d3mopd_watcher")


def parse_domain_map(raw: str) -> dict[str, list[str]]:
    """Parse ``math=ds1,ds2|code=ds3,ds4|if=ds5,ds6`` into ``{domain: [ds, ...]}``.

    Mirrors the format that launch scripts export as ``D3MOPD_DOMAIN_MAP``.
    Empty / malformed input returns ``{}``.
    """
    out: dict[str, list[str]] = {}
    if not raw:
        return out
    for entry in raw.split("|"):
        if "=" not in entry:
            continue
        domain, ds_csv = entry.split("=", 1)
        domain = domain.strip()
        if not domain:
            continue
        out[domain] = [s.strip() for s in ds_csv.split(",") if s.strip()]
    return out


# ---------- wandb fetching ----------


def pick_run(api: wandb.Api, entity_project: str, group_prefix: str, run_id: str | None):
    """Pick the latest wandb run.

    ``run_id`` short-circuits everything. Otherwise we scan the most recent
    ~50 runs and pick the latest whose group **startswith** ``group_prefix``
    — slime appends an 8-char random hash to ``--wandb-group=EXP_NAME``, so
    exact match would miss every real run.
    """
    if run_id:
        return api.run(f"{entity_project}/{run_id}")
    runs = list(api.runs(entity_project, order="-created_at", per_page=50))
    for r in runs:
        if r.group and r.group.startswith(group_prefix):
            return r
    raise RuntimeError(f"no wandb runs in {entity_project} with group startswith {group_prefix!r}")


def fetch_metrics(run, keys: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Return ``{key: [(step, value), ...]}``. ``step`` is ``rollout/step``.

    Rows where the key is missing are skipped silently.
    """
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    all_keys = keys + ["rollout/step"]
    for row in run.scan_history(keys=all_keys):
        step = row.get("rollout/step")
        if step is None:
            continue
        for k in keys:
            v = row.get(k)
            if v is None:
                continue
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                continue
            out[k].append((int(step), v_f))
    return out


# ---------- convergence judgment ----------


def ema(values: list[float], alpha: float | None = None) -> float | None:
    if not values:
        return None
    if alpha is None:
        alpha = 2 / (len(values) + 1)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def judge_domain(
    kl_series: list[tuple[int, float]],
    reward_series: list[tuple[int, float]],
    window_W: int,
    kl_eps: float,
    *,
    require_reward_gate: bool = True,
) -> dict[str, Any]:
    """Per-domain single-poll convergence vote.

    Returns a dict with:
      - ``step`` — latest rollout step (None if not enough data)
      - ``kl_ema`` — latest EMA over W
      - ``kl_rel_change`` — relative change vs EMA W steps back
      - ``kl_plateau`` — bool
      - ``reward_stagnant`` — bool (no new reward high in last W); None if disabled
      - ``converged`` — KL plateau (AND reward stagnant if gate enabled)

    ``require_reward_gate=False`` is for pure OPD KL settings where there is no
    independent task reward to use as a second signal — converged collapses to
    just ``kl_plateau``.
    """
    out: dict[str, Any] = {
        "step": None, "kl_ema": None, "kl_rel_change": None,
        "kl_plateau": False, "reward_stagnant": None, "converged": False,
    }
    if len(kl_series) < 2 * window_W:
        return out
    out["step"] = kl_series[-1][0]
    recent_kls = [v for _, v in kl_series[-window_W:]]
    earlier_kls = [v for _, v in kl_series[-2 * window_W:-window_W]]
    ema_now = ema(recent_kls)
    ema_earlier = ema(earlier_kls)
    if ema_earlier is None or ema_earlier == 0:
        return out
    rel = abs(ema_now - ema_earlier) / abs(ema_earlier)
    out["kl_ema"] = ema_now
    out["kl_rel_change"] = rel
    out["kl_plateau"] = rel < kl_eps

    if not require_reward_gate:
        # KL-only mode: converged = kl_plateau, reward field stays None.
        out["converged"] = out["kl_plateau"]
        return out

    if reward_series and len(reward_series) >= window_W:
        recent_rewards = [v for _, v in reward_series[-window_W:]]
        earlier_max = max((v for _, v in reward_series[:-window_W]), default=float("-inf"))
        recent_max = max(recent_rewards)
        out["reward_stagnant"] = recent_max <= earlier_max
    else:
        out["reward_stagnant"] = False

    out["converged"] = out["kl_plateau"] and bool(out["reward_stagnant"])
    return out


# ---------- dynamic-mode ratio computation ----------


def compute_dynamic_ratios(
    ema_kl_per_domain: dict[str, float],
    initial_kl_per_domain: dict[str, float],
    temperature: float,
    ratio_floor: float,
    mapping: str = "softmax",
    sigmoid_center: float = 0.20,
    sigmoid_slope: float = 0.03,
    normalize_signal: bool = False,
) -> dict[str, float]:
    """Per-domain mixture ratios from initial-KL-normalized signal with floor.

    For each domain d:
      normalized[d] = ema_kl[d] / initial_kl[d]     (1.0 = no progress; <1 = learning)

    If ``normalize_signal=True``, additionally divide by max(normalized) so the
    signal is scaled to [0, 1] BEFORE softmax. This makes T semantics
    scale-invariant (same T value produces same relative sharpness regardless
    of signal magnitude — needed for fair comparison with delta/composite
    modes whose raw signals live on different scales).

    Two mappings:
      mapping="softmax" (default):
        weight[d]   = exp(normalized[d] / T)
        Higher normalized gets relatively more — but plateau domains still get ~1/n share
        because softmax is normalized across domains.

      mapping="sigmoid" (plateau-aware):
        weight[d]   = 1 / (1 + exp(-(normalized[d] - sigmoid_center) / sigmoid_slope))
        Absolute threshold at sigmoid_center; domains with normalized < center
        sharply drop to ≈0 weight → ratio → floor. This is the differentiable
        equivalent of static-mode's "KL plateau triggers DOWN" logic.

      In either case:
        soft[d]     = weight[d] / Σ weight[·]
        ratio[d]    = floor + (1 - n_domains * floor) * soft[d]

    Guarantees: Σ ratio = 1.0; ratio[d] ≥ floor for every domain.
    """
    domains = sorted(ema_kl_per_domain)
    if not domains:
        return {}
    n = len(domains)
    if ratio_floor < 0 or ratio_floor * n > 1.0:
        raise ValueError(
            f"ratio_floor={ratio_floor} infeasible for n={n} (n*floor must be < 1)"
        )

    eps = 1e-8
    normalized = [
        ema_kl_per_domain[d] / max(initial_kl_per_domain[d], eps) for d in domains
    ]

    # Optional: max-normalize signal so T has scale-invariant meaning.
    if normalize_signal:
        max_signal = max(normalized)
        if max_signal > 1e-12:
            normalized = [x / max_signal for x in normalized]

    if mapping == "softmax":
        # Subtract max for numerical stability before exp.
        max_n = max(normalized)
        exps = [math.exp((x - max_n) / temperature) for x in normalized]
        z = sum(exps)
        soft_vals = [e / z for e in exps]
    elif mapping == "sigmoid":
        # Independent sigmoid per domain (absolute plateau threshold), then normalize sum=1.
        sig_vals = [1.0 / (1.0 + math.exp(-(x - sigmoid_center) / sigmoid_slope))
                    for x in normalized]
        z = sum(sig_vals)
        if z < 1e-12:
            # All domains way below center → fall back to equal weights → all floor.
            soft_vals = [1.0 / n] * n
        else:
            soft_vals = [s / z for s in sig_vals]
    else:
        raise ValueError(f"unknown mapping={mapping!r} (expected 'softmax' or 'sigmoid')")

    remaining = 1.0 - n * ratio_floor
    return {d: ratio_floor + remaining * sm for d, sm in zip(domains, soft_vals)}


def compute_composite_ratios(
    ema_now_per_domain: dict[str, float],
    ema_earlier_per_domain: dict[str, float],
    initial_kl_per_domain: dict[str, float],
    temperature: float,
    ratio_floor: float,
    progress_smooth: bool = False,
    progress_smooth_threshold: float = 0.02,
    progress_smooth_scale: float = 0.02,
    rolling_deltas_per_domain: dict[str, list[float]] | None = None,
    velocity_floor: float = 0.0,
    abs_kl_floor: float = 0.0,
    rehearsal_domains: frozenset[str] | None = None,
) -> dict[str, dict[str, float]]:
    """M5e-composite signal: (normalized gap) × (recent relative progress), max-normalized.

    For each domain d:
      normalized[d] = ema_now[d] / initial_kl[d]                           (remaining gap %)
      progress[d]   = max(0, (ema_earlier[d] - ema_now[d]) / |ema_earlier[d]|)
                                                                            (recent KL drop rate)
      raw[d]        = normalized[d] * progress[d]
      signal[d]     = raw[d] / max(raw[·])                                  (max-normalized to [0,1])

    Then:
      softmax[d]    = exp(signal[d] / T) / Σ exp(signal[·] / T)
      ratio[d]      = floor + (1 - n_domains * floor) * softmax[d]

    Guarantees: Σ ratio = 1.0; ratio[d] ≥ floor.

    Two optional modifiers (used exclusively):
      • ``progress_smooth=True`` — replace hard clip with sigmoid smoothing
        (M5g v1). See progress_smooth_threshold / scale.
      • ``rolling_deltas_per_domain`` — pass K signed relative deltas per
        domain (e.g. from K non-overlapping windows). If provided, progress
        is computed from the AVG of those deltas rather than the single
        (ema_now, ema_earlier) pair. Fixes M5e's over-sensitivity to a
        single transient KL bump (Q19 rolling-avg K=3 idea).

    Two backward-compatible robustness knobs (default off = current behavior):
      • ``velocity_floor ∈ [0, 1]`` (Patch A) — remap progress from [0, 1] into
        [velocity_floor, 1] after the three progress branches, so a domain
        whose KL rebounds does NOT get its composite signal zeroed while its
        remaining gap is still large. ``0.0`` (default) reproduces the paper
        formulation bit-exact; ``0.2`` retains 20% of the gap signal as a
        gap-only fallback when velocity ≤ 0.
      • ``abs_kl_floor`` / ``rehearsal_domains`` (Patch B) — for merge /
        anti-forgetting scenarios where a "teacher" is the frozen student
        itself: such a domain has near-zero KL from the start, so
        normalized = ema / initial_kl explodes on any noise and hogs the
        mixture. Domains listed in ``rehearsal_domains`` (explicit tag) or
        whose ``initial_kl < abs_kl_floor`` (auto-detect by absolute KL
        magnitude) get ``raw_signal[d] = 0`` → softmax uniform → they land
        on ``ratio_floor``. Defaults (empty set / 0.0) disable both paths.
    """
    domains = sorted(ema_now_per_domain)
    if not domains:
        return {}
    n = len(domains)
    if ratio_floor < 0 or ratio_floor * n > 1.0:
        raise ValueError(
            f"ratio_floor={ratio_floor} infeasible for n={n} (n*floor must be < 1)"
        )

    eps = 1e-8
    rehearsal_set = rehearsal_domains or frozenset()
    normalized: dict[str, float] = {}
    progress: dict[str, float] = {}
    raw_signal: dict[str, float] = {}
    is_rehearsal_map: dict[str, bool] = {}
    for d in domains:
        ema_n = ema_now_per_domain[d]
        ema_e = ema_earlier_per_domain[d]
        ikl = initial_kl_per_domain[d]
        normalized[d] = ema_n / max(ikl, eps)

        if rolling_deltas_per_domain is not None:
            # Rolling-avg mode: use average of K signed deltas per domain.
            deltas = rolling_deltas_per_domain.get(d, [])
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                progress[d] = max(0.0, -avg_delta)
            else:
                progress[d] = 0.0
        elif progress_smooth:
            # signed relative change; negative = KL falling = learning
            delta_rel = (ema_n - ema_e) / max(abs(ema_e), eps)
            # sigmoid so that: delta_rel << -threshold → progress ≈ 1 (learning fast)
            #                  delta_rel ≈ -threshold → progress = 0.5 (moderate)
            #                  delta_rel >> -threshold → progress ≈ 0 (plateau / KL rising)
            progress[d] = 1.0 / (1.0 + math.exp((delta_rel + progress_smooth_threshold) / progress_smooth_scale))
        else:
            progress[d] = max(0.0, (ema_e - ema_n) / max(abs(ema_e), eps))

        # Patch A: velocity floor. When >0, remap progress from [0, 1] into
        # [velocity_floor, 1] so a KL-rebounding domain with large remaining
        # gap keeps a gap-only fallback signal instead of getting zeroed.
        # velocity_floor = 0.0 (default) is a no-op — no branch is even taken,
        # keeping the paper's product-form composite bit-exact.
        if velocity_floor > 0.0:
            progress[d] = velocity_floor + (1.0 - velocity_floor) * min(progress[d], 1.0)

        raw_signal[d] = normalized[d] * progress[d]

        # Patch B: rehearsal-domain / low-abs-KL gate. Domains whose "teacher"
        # is the frozen student itself have initial_kl ≈ 0, so normalized
        # explodes on any noise — the composite then falsely awards them the
        # bulk of the mixture. Explicit tag (rehearsal_domains) or auto-detect
        # by absolute KL magnitude (initial_kl < abs_kl_floor) → force
        # raw_signal to 0 → softmax collapses to uniform → the domain lands
        # on ratio_floor. Defaults (empty set, 0.0) disable both paths.
        is_rehearsal = (
            d in rehearsal_set
            or (abs_kl_floor > 0.0 and ikl < abs_kl_floor)
        )
        is_rehearsal_map[d] = is_rehearsal
        if is_rehearsal:
            raw_signal[d] = 0.0

    # Rehearsal-gated domains are pinned to ratio_floor exactly (Patch B), and
    # the softmax is computed over the remaining domains only. Without this
    # bypass, forcing raw_signal=0 would leave softmax(0/T) > 0 → the rehearsal
    # domain would land slightly above floor (a few % of leak), and worse, that
    # leak would show up in the max_raw denominator only if EVERY domain were
    # rehearsal. When no domain is gated (default path), this reduces to the
    # exact same softmax + floor mapping used before Patch B.
    non_rehearsal = [d for d in domains if not is_rehearsal_map[d]]
    n_rehearsal = n - len(non_rehearsal)

    if not non_rehearsal:
        # Every domain gated → each on floor + equal split of the remainder.
        # (Same fallback the "all-plateau" branch used to hit: uniform 1/n.)
        signal_norm = {d: 0.0 for d in domains}
        ratios = {d: 1.0 / n for d in domains}
    else:
        # Max-normalize over non-rehearsal only so T semantics stay scale-free.
        non_rh_raw = [raw_signal[d] for d in non_rehearsal]
        max_raw = max(non_rh_raw)
        if max_raw < 1e-12:
            # All non-rehearsal plateaued: uniform among them; rehearsal on floor.
            signal_norm = {d: 0.0 for d in domains}
        else:
            signal_norm = {
                d: (raw_signal[d] / max_raw) if not is_rehearsal_map[d] else 0.0
                for d in domains
            }
        sig_vals = [signal_norm[d] for d in non_rehearsal]
        max_s = max(sig_vals)
        exps = [math.exp((x - max_s) / temperature) for x in sig_vals]
        z = sum(exps)
        softmax_vals = [e / z for e in exps]
        # Non-rehearsal domains share (1 - n_rehearsal * floor); each still ≥ floor.
        non_rh_remaining = 1.0 - n_rehearsal * ratio_floor - len(non_rehearsal) * ratio_floor
        ratios: dict[str, float] = {d: ratio_floor for d in domains if is_rehearsal_map[d]}
        for d, sm in zip(non_rehearsal, softmax_vals):
            ratios[d] = ratio_floor + non_rh_remaining * sm

    return {
        d: {
            "ratio": ratios[d],
            "normalized": normalized[d],
            "progress": progress[d],
            "raw_signal": raw_signal[d],
            "signal": signal_norm[d],
            "is_rehearsal": is_rehearsal_map[d],
        }
        for d in domains
    }


def compute_delta_softmax_ratios(
    ema_now_per_domain: dict[str, float],
    ema_earlier_per_domain: dict[str, float],
    temperature: float,
    ratio_floor: float,
    signal_cap: float = 1.0,
    normalize_signal: bool = False,
) -> dict[str, dict[str, float]]:
    """Per-domain mixture ratios via Δ (recent KL drop speed) softmax with floor.

    For each domain d (direction A: reward fast-learners, penalize plateau):
      signed_delta[d] = (ema_now[d] - ema_earlier[d]) / max(|ema_earlier[d]|, eps)
                       = signed relative change; negative = KL falling (= learning)
      signal[d]       = clip(max(0, -signed_delta[d]), [0, signal_cap])
                       = relative DROP magnitude; plateau OR KL-rising → 0
      (if normalize_signal: signal[d] /= max(signal))       # scale to [0,1]
      softmax[d]      = exp(signal[d] / T) / Σ exp(signal[·] / T)
      ratio[d]        = floor + (1 - n_domains * floor) * softmax[d]

    Guarantees: Σ ratio = 1.0; ratio[d] ≥ floor for every domain.
    Plateau / KL-rising → signal = 0 → only floor (catastrophic-forgetting safety).
    All-plateau case degrades cleanly to equal 1/n distribution.

    Returns dict per domain with keys: ratio, signed_delta, signal (latter two
    persisted to status JSON for observability).
    """
    domains = sorted(ema_now_per_domain)
    if not domains:
        return {}
    n = len(domains)
    if ratio_floor < 0 or ratio_floor * n > 1.0:
        raise ValueError(
            f"ratio_floor={ratio_floor} infeasible for n={n} (n*floor must be < 1)"
        )
    if signal_cap <= 0:
        raise ValueError(f"signal_cap={signal_cap} must be > 0")

    eps = 1e-8
    signed_deltas: dict[str, float] = {}
    signals: dict[str, float] = {}
    for d in domains:
        ema_e = ema_earlier_per_domain[d]
        ema_n = ema_now_per_domain[d]
        sd = (ema_n - ema_e) / max(abs(ema_e), eps)
        signed_deltas[d] = sd
        # KL drop magnitude (positive when learning), clipped to [0, cap].
        signals[d] = min(max(0.0, -sd), signal_cap)

    # Optional: max-normalize signal so T has scale-invariant meaning across
    # gap/delta/composite. Keep the pre-norm values for status/logging.
    signals_for_softmax = dict(signals)
    if normalize_signal:
        max_signal = max(signals_for_softmax.values())
        if max_signal > 1e-12:
            signals_for_softmax = {d: v / max_signal for d, v in signals_for_softmax.items()}

    # Softmax with numerical-stable max-subtraction.
    sig_vals = [signals_for_softmax[d] for d in domains]
    max_s = max(sig_vals)
    exps = [math.exp((x - max_s) / temperature) for x in sig_vals]
    z = sum(exps)
    softmax_vals = [e / z for e in exps]
    remaining = 1.0 - n * ratio_floor
    ratios = {d: ratio_floor + remaining * sm for d, sm in zip(domains, softmax_vals)}

    return {
        d: {
            "ratio": ratios[d],
            "signed_delta": signed_deltas[d],
            "signal": signals[d],
        }
        for d in domains
    }


# ---------- writer state ----------


class WriterState:
    """Tracks per-domain patience counter, downsampled flag, and up-sample logic."""

    def __init__(
        self,
        patience_K: int,
        rehearsal_rate: float,
        domain_map: dict[str, list[str]],
    ) -> None:
        self.patience_K = patience_K
        self.rehearsal_rate = rehearsal_rate
        self.domain_map = domain_map
        # ----- static-mode fields -----
        self.patience: dict[str, int] = {d: 0 for d in domain_map}
        self.downsampled: dict[str, bool] = {d: False for d in domain_map}
        self.since_step: dict[str, int | None] = {d: None for d in domain_map}
        # For up-sample (new reward high) detection: peak reward at the moment
        # we flipped to downsampled.
        self.peak_reward_at_freeze: dict[str, float] = {}
        # ----- dynamic-gap mode fields -----
        # initial_kl[d] is locked once on the Nth poll where domain d has
        # ≥ seed_points history; persisted to status JSON and reloaded on
        # restart so we never re-baseline mid-training.
        self.initial_kl: dict[str, float | None] = {d: None for d in domain_map}
        self.mixture_ratio: dict[str, float] = {}
        self.current_ema_kl: dict[str, float] = {}
        self.normalized_kl: dict[str, float] = {}
        self.last_update_step: int | None = None
        # ----- dynamic-delta mode fields (velocity-only signal variant) -----
        # No "lock-once" phase; ema_earlier/now are recomputed every update.
        # Stored for status JSON observability + restart restore of last ratios.
        self.ema_earlier_per_domain: dict[str, float] = {}
        self.signed_delta_per_domain: dict[str, float] = {}
        self.signal_per_domain: dict[str, float] = {}
        # ----- composite Patch B observability -----
        # Which domains were classified as rehearsal on the last update.
        self.is_rehearsal_per_domain: dict[str, bool] = {}

    def step(
        self,
        judgments: dict[str, dict[str, Any]],
        domain_reward: dict[str, list[tuple[int, float]]],
    ) -> None:
        for domain in self.domain_map:
            j = judgments.get(domain, {})
            if not self.downsampled[domain]:
                if j.get("converged"):
                    self.patience[domain] += 1
                else:
                    self.patience[domain] = 0
                if self.patience[domain] >= self.patience_K:
                    self.downsampled[domain] = True
                    self.since_step[domain] = j.get("step")
                    series = domain_reward.get(domain, [])
                    self.peak_reward_at_freeze[domain] = (
                        max((v for _, v in series), default=float("-inf"))
                    )
                    logger.warning(
                        "DOMAIN %s DOWNSAMPLED at step %s (rehearsal=%.2f, peak_reward_at_freeze=%.4f)",
                        domain, self.since_step[domain], self.rehearsal_rate,
                        self.peak_reward_at_freeze[domain],
                    )
            else:
                # up-sample if reward creates a new high
                series = domain_reward.get(domain, [])
                if series:
                    recent_max = max(v for _, v in series)
                    peak = self.peak_reward_at_freeze.get(domain, float("-inf"))
                    if recent_max > peak:
                        logger.warning(
                            "DOMAIN %s UP-SAMPLED (new reward high %.4f > %.4f)",
                            domain, recent_max, peak,
                        )
                        self.downsampled[domain] = False
                        self.since_step[domain] = None
                        self.patience[domain] = 0
                        self.peak_reward_at_freeze.pop(domain, None)

    def to_status_dict(self, exp_name: str) -> dict[str, Any]:
        return {
            "exp_name": exp_name,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "domains": {
                domain: {
                    "downsampled": self.downsampled[domain],
                    "rehearsal_rate": self.rehearsal_rate if self.downsampled[domain] else 1.0,
                    "since_step": self.since_step[domain],
                    # Persist peak reward at freeze so a watcher restart doesn't
                    # forget it and falsely up-sample on next poll.
                    "peak_reward_at_freeze": self.peak_reward_at_freeze.get(domain),
                    "data_sources": self.domain_map[domain],
                }
                for domain in self.domain_map
            },
        }

    def to_status_dict_static_ratio(
        self, exp_name: str, ratio_floor: float,
    ) -> dict[str, Any]:
        """Static-ratio mode (naive baseline for the dynamic-mixture family).

        Reuses static-mode plateau-detection state (self.downsampled) but writes
        ``mode: "static-ratio"`` + per-domain ``mixture_ratio`` so the data source
        (DynamicRatioStratifiedRolloutDataSourceWithBuffer) can cut per-batch
        quota directly instead of relying on a rejection filter. Ratio rule:

        - 0 domain converged                 → uniform  1/n each
        - 1..(n-1) domain converged          → converged=floor each, remainder split
                                               equally among unconverged domains
        - all n converged                    → uniform 1/n (restore full mixture)

        This one-shot-per-domain adjustment is a naive baseline against the
        dynamic family's continuous, per-window ratio updates.
        """
        n = len(self.domain_map)
        n_converged = sum(1 for d in self.domain_map if self.downsampled[d])
        if n_converged == 0 or n_converged == n:
            mixture = {d: 1.0 / n for d in self.domain_map}
        else:
            unconverged_share = (1.0 - n_converged * ratio_floor) / (n - n_converged)
            mixture = {
                d: ratio_floor if self.downsampled[d] else unconverged_share
                for d in self.domain_map
            }
        return {
            "exp_name": exp_name,
            "mode": "static-ratio",
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "ratio_floor": ratio_floor,
            "n_converged": n_converged,
            "domains": {
                domain: {
                    "mixture_ratio": mixture[domain],
                    "downsampled": self.downsampled[domain],
                    "since_step": self.since_step[domain],
                    "data_sources": self.domain_map[domain],
                }
                for domain in self.domain_map
            },
        }

    @classmethod
    def from_status_file(
        cls,
        path: Path,
        patience_K: int,
        rehearsal_rate: float,
        domain_map: dict[str, list[str]],
    ) -> "WriterState":
        """Construct a WriterState, seeding from an existing status file if present.

        Without this, a watcher restart in writer mode would overwrite the
        existing status with all-False on the next poll, dropping any active
        downsample decisions for the 15 min it takes to re-detect plateau.
        """
        state = cls(patience_K=patience_K, rehearsal_rate=rehearsal_rate, domain_map=domain_map)
        try:
            with path.open() as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return state
        domains_in = data.get("domains")
        if not isinstance(domains_in, dict):
            return state
        for domain, cfg in domains_in.items():
            if domain not in domain_map or not isinstance(cfg, dict):
                continue
            if cfg.get("downsampled"):
                state.downsampled[domain] = True
                state.since_step[domain] = cfg.get("since_step")
                # Patience already crossed the threshold for this domain.
                state.patience[domain] = patience_K
                peak = cfg.get("peak_reward_at_freeze")
                if isinstance(peak, (int, float)):
                    state.peak_reward_at_freeze[domain] = float(peak)
                else:
                    # Old status files (pre-fix) didn't persist this field.
                    # Use -inf so the next poll will treat any reward as a
                    # candidate "new high"; the step() up-sample check will
                    # then re-seed peak_reward_at_freeze from current history.
                    state.peak_reward_at_freeze[domain] = float("-inf")
                logger.info(
                    "WriterState seeded: domain %s already downsampled (since_step=%s, peak=%s)",
                    domain, state.since_step[domain], state.peak_reward_at_freeze.get(domain),
                )
        return state

    # ---------- dynamic mode ----------

    def step_dynamic(
        self,
        domain_kl: dict[str, list[tuple[int, float]]],
        ema_window: int,
        seed_points: int,
        update_every_steps: int,
        temperature: float,
        ratio_floor: float,
        mapping: str = "softmax",
        sigmoid_center: float = 0.20,
        sigmoid_slope: float = 0.03,
        normalize_signal: bool = False,
    ) -> dict[str, Any]:
        """One poll's dynamic-mode work; returns diagnostic dict for logging.

        - Phase 1: lock initial_kl[d] when domain d first has ≥ seed_points rows.
        - Phase 2: gated on (a) all initial_kl locked and (b) latest_step ≥
          last_update_step + update_every_steps.
        - Phase 3: compute EMA over last ema_window points per domain,
          normalize by initial_kl, softmax with floor → new mixture_ratio.
        """
        diag: dict[str, Any] = {
            "updated": False, "reason": None, "latest_step": None,
            "normalized": {}, "ratio": {},
        }
        # Phase 1: one-shot initial_kl lock
        for domain in self.domain_map:
            if self.initial_kl.get(domain) is None:
                series = domain_kl.get(domain, [])
                if len(series) >= seed_points:
                    seed_vals = [v for _, v in series[:seed_points]]
                    self.initial_kl[domain] = sum(seed_vals) / len(seed_vals)
                    logger.warning(
                        "DOMAIN %s initial_kl LOCKED = %.6g (mean of first %d points, steps %s..%s)",
                        domain, self.initial_kl[domain], seed_points,
                        series[0][0], series[seed_points - 1][0],
                    )

        # Phase 2: all-locked + cadence check
        if any(self.initial_kl.get(d) is None for d in self.domain_map):
            unlocked = [d for d in self.domain_map if self.initial_kl.get(d) is None]
            diag["reason"] = f"awaiting_initial_kl_lock_for_{unlocked}"
            return diag

        latest_step = max(
            (series[-1][0] for series in domain_kl.values() if series),
            default=-1,
        )
        diag["latest_step"] = latest_step
        if latest_step < 0:
            diag["reason"] = "no_data_yet"
            return diag
        if (
            self.last_update_step is not None
            and latest_step < self.last_update_step + update_every_steps
        ):
            need = self.last_update_step + update_every_steps - latest_step
            diag["reason"] = f"waiting_{need}_more_steps_(next_update_step≥{self.last_update_step + update_every_steps})"
            return diag

        # Phase 3: compute EMA over last ema_window points per domain
        ema_per_domain: dict[str, float] = {}
        for domain in self.domain_map:
            series = domain_kl.get(domain, [])
            if len(series) < ema_window:
                diag["reason"] = f"insufficient_ema_window_data_in_{domain}_(n={len(series)}<W={ema_window})"
                return diag
            recent_vals = [v for _, v in series[-ema_window:]]
            ema_per_domain[domain] = ema(recent_vals)

        new_ratios = compute_dynamic_ratios(
            ema_per_domain,
            {d: float(v) for d, v in self.initial_kl.items()},
            temperature=temperature,
            ratio_floor=ratio_floor,
            mapping=mapping,
            sigmoid_center=sigmoid_center,
            sigmoid_slope=sigmoid_slope,
            normalize_signal=normalize_signal,
        )
        # Commit
        self.mixture_ratio = new_ratios
        self.current_ema_kl = ema_per_domain
        self.normalized_kl = {
            d: ema_per_domain[d] / max(float(self.initial_kl[d]), 1e-8)
            for d in self.domain_map
        }
        self.last_update_step = latest_step

        diag["updated"] = True
        diag["normalized"] = dict(self.normalized_kl)
        diag["ratio"] = dict(new_ratios)
        return diag

    def to_status_dict_dynamic(self, exp_name: str) -> dict[str, Any]:
        return {
            "exp_name": exp_name,
            "mode": "dynamic",
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "last_update_step": self.last_update_step,
            "initial_kl_per_domain": {
                d: float(v) for d, v in self.initial_kl.items() if v is not None
            },
            "domains": {
                domain: {
                    "mixture_ratio": self.mixture_ratio.get(domain),
                    "current_ema_kl": self.current_ema_kl.get(domain),
                    "normalized_kl": self.normalized_kl.get(domain),
                    "data_sources": self.domain_map[domain],
                }
                for domain in self.domain_map
            },
        }

    @classmethod
    def from_status_file_dynamic(
        cls,
        path: Path,
        domain_map: dict[str, list[str]],
    ) -> "WriterState":
        """Construct a dynamic-mode WriterState, restoring initial_kl + ratios from JSON.

        Static-mode fields (patience_K / rehearsal_rate) are passed as 0 / 0.0
        since they're unused in dynamic mode.
        """
        state = cls(patience_K=0, rehearsal_rate=0.0, domain_map=domain_map)
        try:
            with path.open() as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return state

        if data.get("mode") != "dynamic":
            logger.warning(
                "from_status_file_dynamic: existing status file %s has mode=%r (not 'dynamic'); "
                "ignoring it and starting fresh.",
                path, data.get("mode"),
            )
            return state

        initial = data.get("initial_kl_per_domain") or {}
        for d, v in initial.items():
            if d in domain_map and isinstance(v, (int, float)):
                state.initial_kl[d] = float(v)

        lus = data.get("last_update_step")
        if isinstance(lus, int):
            state.last_update_step = lus

        domains_in = data.get("domains") or {}
        for d, cfg in domains_in.items():
            if not isinstance(cfg, dict) or d not in domain_map:
                continue
            r = cfg.get("mixture_ratio")
            e = cfg.get("current_ema_kl")
            n = cfg.get("normalized_kl")
            if isinstance(r, (int, float)):
                state.mixture_ratio[d] = float(r)
            if isinstance(e, (int, float)):
                state.current_ema_kl[d] = float(e)
            if isinstance(n, (int, float)):
                state.normalized_kl[d] = float(n)

        locked = [d for d in domain_map if state.initial_kl.get(d) is not None]
        if locked:
            logger.info(
                "WriterState (dynamic) seeded: initial_kl locked for %s, "
                "last_update_step=%s, mixture_ratio=%s",
                locked, state.last_update_step,
                {d: f"{r*100:.1f}%" for d, r in state.mixture_ratio.items()},
            )
        return state

    # ---------- dynamic-composite mode (M5e-composite) ----------

    def step_dynamic_composite(
        self,
        domain_kl: dict[str, list[tuple[int, float]]],
        ema_window: int,
        seed_points: int,
        update_every_steps: int,
        temperature: float,
        ratio_floor: float,
        progress_smooth: bool = False,
        progress_smooth_threshold: float = 0.02,
        progress_smooth_scale: float = 0.02,
        progress_rolling_k: int = 1,
        velocity_floor: float = 0.0,
        abs_kl_floor: float = 0.0,
        rehearsal_domains: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """One poll's dynamic-composite-mode work; returns diagnostic dict.

        Combines dynamic-gap's initial_kl lock with dynamic-delta's ema_earlier
        computation. Signal = normalized × progress, max-normalized before softmax.

        If ``progress_rolling_k > 1``: progress is averaged over K non-overlapping
        (W-point) windows of Δ. Fixes M5e's over-sensitivity to a single
        transient KL bump — one noisy window is diluted by the K-1 prior
        windows. Needs ≥ (K+1)*W history points before first update.
        Falls back to single-window Δ if history is short.

        Warmup: needs ≥ max(seed_points, (max(K,1)+1)*ema_window) points.
        """
        diag: dict[str, Any] = {
            "updated": False, "reason": None, "latest_step": None,
            "normalized": {}, "progress": {}, "raw_signal": {}, "signal": {}, "ratio": {},
            "rolling_deltas": {},
        }
        # Phase 1: one-shot initial_kl lock (same as dynamic-gap)
        for domain in self.domain_map:
            if self.initial_kl.get(domain) is None:
                series = domain_kl.get(domain, [])
                if len(series) >= seed_points:
                    seed_vals = [v for _, v in series[:seed_points]]
                    self.initial_kl[domain] = sum(seed_vals) / len(seed_vals)
                    logger.warning(
                        "DOMAIN %s initial_kl LOCKED = %.6g (mean of first %d points, steps %s..%s)",
                        domain, self.initial_kl[domain], seed_points,
                        series[0][0], series[seed_points - 1][0],
                    )

        # Phase 2: all-locked + cadence check
        if any(self.initial_kl.get(d) is None for d in self.domain_map):
            unlocked = [d for d in self.domain_map if self.initial_kl.get(d) is None]
            diag["reason"] = f"awaiting_initial_kl_lock_for_{unlocked}"
            return diag

        latest_step = max(
            (series[-1][0] for series in domain_kl.values() if series),
            default=-1,
        )
        diag["latest_step"] = latest_step
        if latest_step < 0:
            diag["reason"] = "no_data_yet"
            return diag
        if (
            self.last_update_step is not None
            and latest_step < self.last_update_step + update_every_steps
        ):
            need = self.last_update_step + update_every_steps - latest_step
            diag["reason"] = f"waiting_{need}_more_steps_(next_update_step≥{self.last_update_step + update_every_steps})"
            return diag

        # Phase 3: per-domain EMA_now (last W) and EMA_earlier (prev W)
        # Plus (if progress_rolling_k > 1): K-1 additional older EMA snapshots
        # to compute K non-overlapping Δs and average.
        K = max(1, int(progress_rolling_k))
        need_min = 2 * ema_window                     # single-window minimum
        need_full = (K + 1) * ema_window              # full rolling coverage
        ema_now_map: dict[str, float] = {}
        ema_earlier_map: dict[str, float] = {}
        rolling_deltas_map: dict[str, list[float]] = {}
        for domain in self.domain_map:
            series = domain_kl.get(domain, [])
            if len(series) < need_min:
                diag["reason"] = (
                    f"insufficient_data_for_2W={need_min}_in_{domain}_(n={len(series)})"
                )
                return diag
            recent = [v for _, v in series[-ema_window:]]
            earlier = [v for _, v in series[-2 * ema_window:-ema_window]]
            ema_now_map[domain] = ema(recent)
            ema_earlier_map[domain] = ema(earlier)

            if K > 1:
                # Try to fit K non-overlapping windows of W points each,
                # then compute K deltas (each = adjacent window pair).
                # Falls back gracefully to fewer deltas if history is short.
                snapshots: list[float] = []
                for k in range(K + 1):
                    end = len(series) - k * ema_window
                    start = end - ema_window
                    if start < 0:
                        break
                    window_vals = [v for _, v in series[start:end]]
                    snapshots.append(ema(window_vals))
                # snapshots is newest→oldest; need at least 2 to form 1 delta
                deltas = []
                for i in range(len(snapshots) - 1):
                    e_new = snapshots[i]
                    e_old = snapshots[i + 1]
                    d = (e_new - e_old) / max(abs(e_old), 1e-8)
                    deltas.append(d)
                rolling_deltas_map[domain] = deltas

        composite_kwargs = dict(
            ema_now_per_domain=ema_now_map,
            ema_earlier_per_domain=ema_earlier_map,
            initial_kl_per_domain={d: float(v) for d, v in self.initial_kl.items()},
            temperature=temperature,
            ratio_floor=ratio_floor,
            progress_smooth=progress_smooth,
            progress_smooth_threshold=progress_smooth_threshold,
            progress_smooth_scale=progress_smooth_scale,
            velocity_floor=velocity_floor,
            abs_kl_floor=abs_kl_floor,
            rehearsal_domains=rehearsal_domains,
        )
        if K > 1 and rolling_deltas_map:
            composite_kwargs["rolling_deltas_per_domain"] = rolling_deltas_map

        result = compute_composite_ratios(**composite_kwargs)
        new_ratios = {d: r["ratio"] for d, r in result.items()}
        new_normalized = {d: r["normalized"] for d, r in result.items()}
        new_progress = {d: r["progress"] for d, r in result.items()}
        new_raw = {d: r["raw_signal"] for d, r in result.items()}
        new_signals = {d: r["signal"] for d, r in result.items()}
        new_is_rehearsal = {d: r["is_rehearsal"] for d, r in result.items()}

        # Commit — reuse fields from both dyn-gap and dyn-delta writer state.
        self.mixture_ratio = new_ratios
        self.current_ema_kl = ema_now_map
        self.ema_earlier_per_domain = ema_earlier_map
        self.normalized_kl = new_normalized
        self.signal_per_domain = new_signals
        self.is_rehearsal_per_domain = new_is_rehearsal
        self.last_update_step = latest_step

        diag["updated"] = True
        diag["normalized"] = new_normalized
        diag["progress"] = new_progress
        diag["raw_signal"] = new_raw
        diag["signal"] = new_signals
        diag["ratio"] = new_ratios
        diag["is_rehearsal"] = new_is_rehearsal
        return diag

    def to_status_dict_dynamic_composite(self, exp_name: str) -> dict[str, Any]:
        return {
            "exp_name": exp_name,
            "mode": "dynamic-composite",
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "last_update_step": self.last_update_step,
            "initial_kl_per_domain": {
                d: float(v) for d, v in self.initial_kl.items() if v is not None
            },
            "domains": {
                domain: {
                    "mixture_ratio": self.mixture_ratio.get(domain),
                    "ema_now": self.current_ema_kl.get(domain),
                    "ema_earlier": self.ema_earlier_per_domain.get(domain),
                    "normalized_kl": self.normalized_kl.get(domain),
                    "signal": self.signal_per_domain.get(domain),
                    # Patch B observability: was this domain classified as
                    # rehearsal on the last update? Only present when either
                    # the explicit flag or abs_kl_floor is in use; absent /
                    # False otherwise. Kept in status JSON so operators can
                    # eyeball the effect without opening the watcher log.
                    "is_rehearsal": self.is_rehearsal_per_domain.get(domain, False),
                    "data_sources": self.domain_map[domain],
                }
                for domain in self.domain_map
            },
        }

    # ---------- dynamic-delta mode (velocity-only signal variant) ----------

    def step_dynamic_delta(
        self,
        domain_kl: dict[str, list[tuple[int, float]]],
        ema_window: int,
        update_every_steps: int,
        temperature: float,
        ratio_floor: float,
        signal_cap: float,
        normalize_signal: bool = False,
    ) -> dict[str, Any]:
        """One poll's dynamic-delta-mode work; returns diagnostic dict.

        Unlike dynamic-gap, no one-shot initial_kl lock — every update recomputes
        EMA_now (last W points) and EMA_earlier (prev W points before that).
        Domains need ≥ 2W history points before they can contribute.

        - Phase 1: cadence check (latest_step ≥ last_update_step + update_every_steps)
        - Phase 2: per-domain EMA over recent W + prev W windows
        - Phase 3: softmax(max(0, -ΔEMA) clipped to signal_cap) + floor
        """
        diag: dict[str, Any] = {
            "updated": False, "reason": None, "latest_step": None,
            "signed_delta": {}, "signal": {}, "ratio": {},
        }
        latest_step = max(
            (series[-1][0] for series in domain_kl.values() if series),
            default=-1,
        )
        diag["latest_step"] = latest_step
        if latest_step < 0:
            diag["reason"] = "no_data_yet"
            return diag
        if (
            self.last_update_step is not None
            and latest_step < self.last_update_step + update_every_steps
        ):
            need = self.last_update_step + update_every_steps - latest_step
            diag["reason"] = f"waiting_{need}_more_steps_(next_update_step≥{self.last_update_step + update_every_steps})"
            return diag

        # Compute EMA_now and EMA_earlier per domain; both need ≥ ema_window points
        ema_now_map: dict[str, float] = {}
        ema_earlier_map: dict[str, float] = {}
        need_2w = 2 * ema_window
        for domain in self.domain_map:
            series = domain_kl.get(domain, [])
            if len(series) < need_2w:
                diag["reason"] = (
                    f"insufficient_data_for_2W={need_2w}_in_{domain}_(n={len(series)})"
                )
                return diag
            recent = [v for _, v in series[-ema_window:]]
            earlier = [v for _, v in series[-2 * ema_window:-ema_window]]
            ema_now_map[domain] = ema(recent)
            ema_earlier_map[domain] = ema(earlier)

        # Compute new ratios + intermediate stats
        result = compute_delta_softmax_ratios(
            ema_now_map, ema_earlier_map,
            temperature=temperature,
            ratio_floor=ratio_floor,
            signal_cap=signal_cap,
            normalize_signal=normalize_signal,
        )
        new_ratios = {d: r["ratio"] for d, r in result.items()}
        new_signed = {d: r["signed_delta"] for d, r in result.items()}
        new_signals = {d: r["signal"] for d, r in result.items()}

        # Commit
        self.mixture_ratio = new_ratios
        self.current_ema_kl = ema_now_map
        self.ema_earlier_per_domain = ema_earlier_map
        self.signed_delta_per_domain = new_signed
        self.signal_per_domain = new_signals
        self.last_update_step = latest_step

        diag["updated"] = True
        diag["signed_delta"] = new_signed
        diag["signal"] = new_signals
        diag["ratio"] = new_ratios
        return diag

    def to_status_dict_dynamic_delta(self, exp_name: str) -> dict[str, Any]:
        return {
            "exp_name": exp_name,
            "mode": "dynamic-delta",
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "last_update_step": self.last_update_step,
            "domains": {
                domain: {
                    "mixture_ratio": self.mixture_ratio.get(domain),
                    "ema_now": self.current_ema_kl.get(domain),
                    "ema_earlier": self.ema_earlier_per_domain.get(domain),
                    "signed_delta": self.signed_delta_per_domain.get(domain),
                    "signal": self.signal_per_domain.get(domain),
                    "data_sources": self.domain_map[domain],
                }
                for domain in self.domain_map
            },
        }

    @classmethod
    def from_status_file_dynamic_delta(
        cls,
        path: Path,
        domain_map: dict[str, list[str]],
    ) -> "WriterState":
        """Construct a dynamic-delta WriterState, restoring ratios + last_update_step.

        No initial_kl to restore (dynamic-delta doesn't use it). Last EMA values
        restored so a watcher restart doesn't need to wait for a fresh 2W window.
        """
        state = cls(patience_K=0, rehearsal_rate=0.0, domain_map=domain_map)
        try:
            with path.open() as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return state

        if data.get("mode") != "dynamic-delta":
            logger.warning(
                "from_status_file_dynamic_delta: existing status file %s has mode=%r "
                "(not 'dynamic-delta'); ignoring it and starting fresh.",
                path, data.get("mode"),
            )
            return state

        lus = data.get("last_update_step")
        if isinstance(lus, int):
            state.last_update_step = lus

        domains_in = data.get("domains") or {}
        for d, cfg in domains_in.items():
            if not isinstance(cfg, dict) or d not in domain_map:
                continue
            r = cfg.get("mixture_ratio")
            en = cfg.get("ema_now")
            ee = cfg.get("ema_earlier")
            sd = cfg.get("signed_delta")
            sg = cfg.get("signal")
            if isinstance(r, (int, float)):
                state.mixture_ratio[d] = float(r)
            if isinstance(en, (int, float)):
                state.current_ema_kl[d] = float(en)
            if isinstance(ee, (int, float)):
                state.ema_earlier_per_domain[d] = float(ee)
            if isinstance(sd, (int, float)):
                state.signed_delta_per_domain[d] = float(sd)
            if isinstance(sg, (int, float)):
                state.signal_per_domain[d] = float(sg)

        if state.mixture_ratio:
            logger.info(
                "WriterState (dynamic-delta) seeded: last_update_step=%s, mixture_ratio=%s",
                state.last_update_step,
                {d: f"{r*100:.1f}%" for d, r in state.mixture_ratio.items()},
            )
        return state


def write_status_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


# ---------- main loop ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-group", required=True,
                   help="EXP_NAME of the training run (slime sets --wandb-group=EXP_NAME).")
    p.add_argument("--wandb-run-id", default=None,
                   help="Specific run_id; if unset, picks most recent run in (project, group).")
    p.add_argument("--domain-map", default=None,
                   help="``math=ds1,ds2|code=ds3,ds4|if=ds5,ds6`` mapping. "
                        "Falls back to ``$D3MOPD_DOMAIN_MAP`` env if unset. "
                        "Required so the status JSON's ``data_sources`` field "
                        "(read by the D3 filter) is filled correctly.")
    p.add_argument("--status-path", required=True)
    p.add_argument("--write-status", action="store_true",
                   help="Off by default (observer mode); on = actually write status.")
    p.add_argument("--poll-interval", type=int, default=300)
    p.add_argument("--ema-window", type=int, default=5, help="W")
    p.add_argument("--kl-eps", type=float, default=0.03, help="ε")
    p.add_argument("--patience", type=int, default=3, help="K")
    p.add_argument("--rehearsal-rate", type=float, default=0.05)
    p.add_argument("--reward-source", default="domain_score",
                   choices=["domain_score", "neg_kl", "none"],
                   help="Convergence-gate reward source. "
                        "'domain_score' (default): pull rollout/domain_score/{domain} "
                        "from wandb — appropriate when a real per-domain task reward "
                        "is being logged. "
                        "'neg_kl': synthesize reward = -KL so 'new high' = 'KL hit new low' "
                        "— for pure-OPD-KL paths where no external task reward exists. "
                        "'none': skip the reward gate entirely; converged = KL plateau only.")
    p.add_argument("--api-key-env", default="WANDB_API_KEY")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--max-polls", type=int, default=0,
                   help="Stop after N polls (0 = run forever).")
    p.add_argument("--enable-all-down-stop", action="store_true",
                   help="When all domains are downsampled, trigger early-stop. "
                        "Avoids the pathological all-DOWN regime where the filter "
                        "rejects the vast majority of every rollout group (teacher "
                        "load spike + pace slowdown). Only effective in writer mode.")
    p.add_argument("--all-down-stop-cmd", default=None,
                   help="Optional shell command to run when all-DOWN is detected. "
                        "Executed via subprocess with shell=True. Leave unset to just "
                        "log a warning and exit the watcher (operator stops training "
                        "manually). Example: --all-down-stop-cmd 'my_scheduler stop 12345'.")
    p.add_argument("--all-down-stop-grace-polls", type=int, default=2,
                   help="Require N consecutive polls of all-DOWN before triggering "
                        "stop (avoids spurious trigger from a single flicker).")
    # ---------- dynamic modes ----------
    p.add_argument("--mode", choices=["static", "static-ratio", "dynamic", "dynamic-delta", "dynamic-composite"], default="static",
                   help="static (default): binary DOWN/UP per domain + Bernoulli "
                        "downsample via filter. dynamic (gap signal): continuous per-domain "
                        "mixture ratio via initial-KL-normalized softmax, consumed by "
                        "DynamicRatioStratifiedRolloutDataSourceWithBuffer. "
                        "dynamic-delta (velocity signal): same data source, but signal "
                        "is recent KL-drop speed (high Δ = active learner → more ratio; "
                        "plateau → only floor). No initial_kl lock phase. "
                        "dynamic-composite: normalized-gap × progress-velocity product signal.")
    p.add_argument("--update-every-steps", type=int, default=10,
                   help="dynamic / dynamic-delta only. Recompute mixture_ratio once latest "
                        "rollout step advances by ≥ N since last update (uses wandb rollout/step).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="dynamic / dynamic-delta only. Softmax temperature T. "
                        "Lower = sharper allocation toward the high-signal domain(s).")
    p.add_argument("--ratio-floor", type=float, default=0.05,
                   help="dynamic / dynamic-delta only. Minimum mixture ratio guaranteed to "
                        "each domain (catastrophic-forgetting safety net). "
                        "Must satisfy n_domains*floor < 1.")
    p.add_argument("--initial-kl-seed-points", type=int, default=5,
                   help="dynamic only (NOT dynamic-delta). Average the first N rollout-step "
                        "KL values to lock the normalization denominator initial_kl[d]. "
                        "Larger = less noise, but delays first ratio update by N rollout steps.")
    p.add_argument("--signal-cap", type=float, default=1.0,
                   help="dynamic-delta only. Cap the per-domain drop signal "
                        "max(0, -ΔEMA/EMA_earlier) to this value before softmax. "
                        "Defaults to 1.0 (= 100% relative drop in one window, which is rarely "
                        "exceeded). Prevents one noise spike from monopolizing the next batch.")
    # ---------- dynamic mapping (softmax vs sigmoid) ----------
    p.add_argument("--mapping", choices=["softmax", "sigmoid"], default="softmax",
                   help="dynamic only (NOT dynamic-delta). How to map normalized_kl → "
                        "per-domain weight. softmax (default): relative comparison via "
                        "exp(normalized / T), tunable by --temperature. sigmoid: absolute "
                        "plateau threshold via 1/(1+exp(-(normalized - center)/slope)); "
                        "domains with normalized < center sharply drop to floor "
                        "(= differentiable plateau detector).")
    p.add_argument("--sigmoid-center", type=float, default=0.20,
                   help="dynamic + mapping=sigmoid only. normalized_kl threshold below which "
                        "the domain is considered plateau (ratio drops sharply toward floor).")
    p.add_argument("--sigmoid-slope", type=float, default=0.03,
                   help="dynamic + mapping=sigmoid only. Transition sharpness of the sigmoid; "
                        "smaller = sharper. 0.03 gives ~95% transition over normalized_kl "
                        "range [center-0.09, center+0.09].")
    # ---------- signal max-normalization ----------
    p.add_argument("--normalize-signal", action="store_true",
                   help="Max-normalize the per-domain signal to [0,1] before softmax. "
                        "Makes T semantics scale-invariant across gap / delta / composite modes "
                        "(different raw signal magnitudes → without this flag, same T means "
                        "different sharpness across modes). Composite mode always max-norms "
                        "regardless; this flag adds the same treatment to gap and delta modes.")
    # ---------- composite: progress smoothing ----------
    p.add_argument("--progress-smooth", action="store_true",
                   help="dynamic-composite only. Replace hard max(0, -Δ) progress with "
                        "sigmoid((-Δ - threshold) / scale). Reduces noise sensitivity: "
                        "when Δ is in the noise region (near 0), progress goes to moderate "
                        "value (~0.3-0.5) instead of hard-zeroed. Fixes M5e's over-sensitivity "
                        "to short-term KL bumps (see Q19).")
    p.add_argument("--progress-smooth-threshold", type=float, default=0.02,
                   help="dynamic-composite + progress-smooth only. Delta_rel threshold where "
                        "sigmoid crosses 0.5. Default 0.02 (progress ≈ 0.5 when KL is dropping "
                        "by 2% per window). Smaller → more lenient (noise region gets higher "
                        "progress); larger → stricter.")
    p.add_argument("--progress-smooth-scale", type=float, default=0.02,
                   help="dynamic-composite + progress-smooth only. Sigmoid width. Default 0.02: "
                        "sigmoid transition covers ±2% around threshold. Smaller = sharper "
                        "transition (closer to hard clip); larger = softer.")
    p.add_argument("--progress-rolling-k", type=int, default=1,
                   help="dynamic-composite only. K = number of non-overlapping W-point windows "
                        "over which to average Δ before applying the max(0, -Δ) clip. Default 1 "
                        "reproduces M5e single-window behavior. K=3 dilutes a single transient "
                        "KL bump (M5g rolling-avg fix). Needs (K+1)*ema_window points of history "
                        "before rolling activates; falls back to single-window otherwise.")
    # ---------- composite Patch A: velocity floor ----------
    p.add_argument("--progress-floor", type=float, default=0.0,
                   help="dynamic-composite only. Velocity floor v_min ∈ [0, 1]; when > 0, remap "
                        "progress from [0, 1] into [v_min, 1] so a KL-rebounding domain with "
                        "large remaining gap keeps a gap-only fallback signal instead of getting "
                        "zeroed. 0.0 (default) reproduces the paper composite bit-exact. 0.2 = "
                        "20%% gap-only fallback when velocity ≤ 0. Composes with --progress-smooth "
                        "and --progress-rolling-k (applied after those).")
    # ---------- composite Patch B: rehearsal / low-abs-KL gate ----------
    p.add_argument("--abs-kl-floor", type=float, default=0.0,
                   help="dynamic-composite only. Auto-detect rehearsal domains by absolute "
                        "initial_kl: any domain with initial_kl < abs_kl_floor has its "
                        "raw_signal forced to 0 → softmax uniform → ratio = ratio_floor. "
                        "For merge/anti-forgetting scenarios where a 'teacher' is the frozen "
                        "student itself (initial_kl ≈ 0, normalized gap explodes on noise, "
                        "domain would otherwise hog the mixture). 0.0 (default) disables "
                        "auto-detect. Typical value: 1e-3 (1-2 orders of magnitude below the "
                        "smallest normal-training initial_kl).")
    p.add_argument("--rehearsal-domains", default="",
                   help="dynamic-composite only. Comma-separated list of domain names to "
                        "explicitly tag as rehearsal (forces raw_signal = 0 → ratio_floor share). "
                        "Falls back to $D3MOPD_REHEARSAL_DOMAINS env if unset. Empty (default) "
                        "disables the explicit-tag path; auto-detect via --abs-kl-floor still "
                        "applies. Use this when the wrapper knows a priori which domains are "
                        "self-distilled and does not want to rely on the abs-KL threshold.")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    raw_map = args.domain_map or os.environ.get("D3MOPD_DOMAIN_MAP", "")
    domain_map = parse_domain_map(raw_map)
    if not domain_map:
        logger.error(
            "no domain map: pass --domain-map or set $D3MOPD_DOMAIN_MAP "
            "(format: math=ds1,ds2|code=ds3,ds4|if=ds5,ds6)"
        )
        sys.exit(1)

    # Composite Patch B: parse explicit rehearsal-domain tags (CLI wins, env fallback).
    raw_rehearsal = args.rehearsal_domains or os.environ.get("D3MOPD_REHEARSAL_DOMAINS", "")
    rehearsal_domains = frozenset(
        d.strip() for d in raw_rehearsal.split(",") if d.strip()
    )
    unknown_rehearsal = rehearsal_domains - domain_map.keys()
    if unknown_rehearsal:
        logger.warning(
            "rehearsal-domains contains names not in domain_map (%s); "
            "they will be silently ignored (known domains: %s)",
            sorted(unknown_rehearsal), sorted(domain_map.keys()),
        )

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        logger.error("env %s not set; can't auth wandb", args.api_key_env)
        sys.exit(1)
    os.environ["WANDB_API_KEY"] = api_key

    entity_project = (
        f"{args.wandb_entity}/{args.wandb_project}"
        if args.wandb_entity else args.wandb_project
    )

    status_path = Path(args.status_path)
    # Seed from existing status file (handles watcher restart without losing
    # already-downsampled domains / locked initial_kl / mixture_ratio; no-op
    # if file absent).
    if args.mode == "dynamic":
        state = WriterState.from_status_file_dynamic(status_path, domain_map=domain_map)
    elif args.mode == "dynamic-delta":
        state = WriterState.from_status_file_dynamic_delta(status_path, domain_map=domain_map)
    elif args.mode == "dynamic-composite":
        # Composite reuses dynamic-gap's status schema (initial_kl locked + normalized_kl).
        state = WriterState.from_status_file_dynamic(status_path, domain_map=domain_map)
    else:
        state = WriterState.from_status_file(
            status_path,
            patience_K=args.patience,
            rehearsal_rate=args.rehearsal_rate,
            domain_map=domain_map,
        )

    if args.mode == "dynamic":
        logger.info(
            "watcher (DYNAMIC-gap) write_status=%s poll=%ds W=%d update_every_steps=%d "
            "T=%.2f floor=%.2f seed_points=%d",
            args.write_status, args.poll_interval, args.ema_window,
            args.update_every_steps, args.temperature, args.ratio_floor,
            args.initial_kl_seed_points,
        )
    elif args.mode == "dynamic-delta":
        logger.info(
            "watcher (DYNAMIC-DELTA) write_status=%s poll=%ds W=%d update_every_steps=%d "
            "T=%.2f floor=%.2f signal_cap=%.2f",
            args.write_status, args.poll_interval, args.ema_window,
            args.update_every_steps, args.temperature, args.ratio_floor,
            args.signal_cap,
        )
    elif args.mode == "dynamic-composite":
        logger.info(
            "watcher (DYNAMIC-COMPOSITE) write_status=%s poll=%ds W=%d update_every_steps=%d "
            "T=%.2f floor=%.2f seed_points=%d (signal = normalized × progress, max-normalized) "
            "progress_floor=%.2f abs_kl_floor=%.4g rehearsal_domains=%s",
            args.write_status, args.poll_interval, args.ema_window,
            args.update_every_steps, args.temperature, args.ratio_floor,
            args.initial_kl_seed_points,
            args.progress_floor, args.abs_kl_floor,
            sorted(rehearsal_domains) if rehearsal_domains else "[]",
        )
    else:
        logger.info(
            "watcher (STATIC) write_status=%s poll=%ds W=%d eps=%.3f K=%d rehearsal=%.2f reward_source=%s",
            args.write_status, args.poll_interval, args.ema_window,
            args.kl_eps, args.patience, args.rehearsal_rate, args.reward_source,
        )
    logger.info("status_path=%s entity_project=%s group=%s",
                status_path, entity_project, args.wandb_group)
    logger.info("domains: %s", {d: len(ds) for d, ds in domain_map.items()})

    # Training side now emits per-domain metrics directly, so just fetch the
    # 3 domain keys (not the 6 fine-grained data_source keys we used to roll up).
    kl_keys = [f"rollout/opd_reverse_kl/{d}" for d in domain_map]
    # Reward signal only applies in static mode; both dynamic modes look at KL only.
    if args.mode in ("dynamic", "dynamic-delta", "dynamic-composite"):
        reward_keys: list[str] = []
        require_reward_gate = False
    else:
        # Only fetch reward keys when the user asks for them; saves wandb roundtrips
        # in modes that ignore reward (none / neg_kl synthesize from KL).
        reward_keys = (
            [f"rollout/domain_score/{d}" for d in domain_map]
            if args.reward_source == "domain_score" else []
        )
        require_reward_gate = args.reward_source != "none"

    poll_count = 0
    all_down_streak = 0
    while True:
        try:
            # Re-create the wandb.Api each poll so we get fresh run state /
            # newly-created runs. The Api object caches Runs result sets, which
            # otherwise sticks us on a stale failed run when a fresh trial
            # (same wandb_group prefix) starts mid-watcher.
            api = wandb.Api(timeout=60)
            run = pick_run(api, entity_project, args.wandb_group, args.wandb_run_id)
            logger.info("poll %d run=%s id=%s state=%s",
                        poll_count, run.name, run.id, run.state)

            domain_kl_raw = fetch_metrics(run, kl_keys)
            domain_reward_raw = fetch_metrics(run, reward_keys) if reward_keys else {}
            # Also pull per-batch ratio + count so the watcher log surfaces
            # the actual D3 effect (how the filter is reshaping the batch
            # over time) without needing the wandb UI.
            ratio_keys = [f"rollout/domain_ratio/{d}" for d in domain_map]
            count_keys = [f"rollout/domain_count/{d}" for d in domain_map]
            domain_ratio_raw = fetch_metrics(run, ratio_keys)
            domain_count_raw = fetch_metrics(run, count_keys)
            # Strip the "rollout/..." prefix so series keys are just domain names.
            domain_kl = {d: domain_kl_raw.get(f"rollout/opd_reverse_kl/{d}", []) for d in domain_map}
            domain_ratio = {d: domain_ratio_raw.get(f"rollout/domain_ratio/{d}", []) for d in domain_map}
            domain_count = {d: domain_count_raw.get(f"rollout/domain_count/{d}", []) for d in domain_map}
            # Derive reward series based on --reward-source:
            #   domain_score → use wandb-fetched rollout/domain_score
            #   neg_kl       → synthesize -KL (new high == KL new low)
            #   none         → empty; judge_domain skips the reward gate
            if args.reward_source == "domain_score":
                domain_reward = {d: domain_reward_raw.get(f"rollout/domain_score/{d}", []) for d in domain_map}
            elif args.reward_source == "neg_kl":
                domain_reward = {d: [(s, -v) for s, v in domain_kl[d]] for d in domain_map}
            else:  # none
                domain_reward = {d: [] for d in domain_map}

            if args.mode == "dynamic":
                diag = state.step_dynamic(
                    domain_kl,
                    ema_window=args.ema_window,
                    seed_points=args.initial_kl_seed_points,
                    update_every_steps=args.update_every_steps,
                    temperature=args.temperature,
                    ratio_floor=args.ratio_floor,
                    mapping=args.mapping,
                    sigmoid_center=args.sigmoid_center,
                    sigmoid_slope=args.sigmoid_slope,
                    normalize_signal=args.normalize_signal,
                )
                for domain in domain_map:
                    kl_n = len(domain_kl.get(domain, []))
                    last_count = domain_count.get(domain, [])
                    last_ratio = domain_ratio.get(domain, [])
                    count_str = f"{int(last_count[-1][1])}" if last_count else "—"
                    actual_ratio_str = f"{last_ratio[-1][1] * 100:.1f}%" if last_ratio else "—"
                    ikl = state.initial_kl.get(domain)
                    cur_ema = state.current_ema_kl.get(domain)
                    nrm = state.normalized_kl.get(domain)
                    target_ratio = state.mixture_ratio.get(domain)
                    logger.info(
                        "%-4s | KL n=%-3d | initial=%s | EMA=%s | normalized=%s | target_ratio=%s | actual batch ratio=%s count=%s",
                        domain, kl_n,
                        f"{ikl:.4g}" if ikl is not None else "—",
                        f"{cur_ema:.4g}" if cur_ema is not None else "—",
                        f"{nrm:.3f}" if nrm is not None else "—",
                        f"{target_ratio * 100:.1f}%" if target_ratio is not None else "—",
                        actual_ratio_str, count_str,
                    )
                if diag["updated"]:
                    logger.warning(
                        "ratios UPDATED at step %s: %s",
                        diag["latest_step"],
                        {d: f"{r * 100:.1f}%" for d, r in diag["ratio"].items()},
                    )
                else:
                    logger.info(
                        "ratios held (no update): %s | latest_step=%s last_update_step=%s",
                        diag["reason"], diag["latest_step"], state.last_update_step,
                    )

                if args.write_status:
                    status_dict = state.to_status_dict_dynamic(exp_name=args.wandb_group)
                    write_status_atomic(status_path, status_dict)
                    logger.info(
                        "status written (dynamic) ratio=%s",
                        {d: f"{(status_dict['domains'][d]['mixture_ratio'] or 0) * 100:.1f}%" for d in domain_map},
                    )
                else:
                    dryrun = state.to_status_dict_dynamic(exp_name=args.wandb_group)
                    logger.info(
                        "DRY-RUN status (dynamic) ratio=%s",
                        {d: f"{(dryrun['domains'][d]['mixture_ratio'] or 0) * 100:.1f}%" for d in domain_map},
                    )
            elif args.mode == "dynamic-delta":
                diag = state.step_dynamic_delta(
                    domain_kl,
                    ema_window=args.ema_window,
                    update_every_steps=args.update_every_steps,
                    temperature=args.temperature,
                    ratio_floor=args.ratio_floor,
                    signal_cap=args.signal_cap,
                    normalize_signal=args.normalize_signal,
                )
                for domain in domain_map:
                    kl_n = len(domain_kl.get(domain, []))
                    last_count = domain_count.get(domain, [])
                    last_ratio = domain_ratio.get(domain, [])
                    count_str = f"{int(last_count[-1][1])}" if last_count else "—"
                    actual_ratio_str = f"{last_ratio[-1][1] * 100:.1f}%" if last_ratio else "—"
                    ema_now = state.current_ema_kl.get(domain)
                    ema_earlier = state.ema_earlier_per_domain.get(domain)
                    sd = state.signed_delta_per_domain.get(domain)
                    sg = state.signal_per_domain.get(domain)
                    target_ratio = state.mixture_ratio.get(domain)
                    logger.info(
                        "%-4s | KL n=%-3d | EMA_now=%s | EMA_earlier=%s | signed_Δ=%s | signal=%s | target_ratio=%s | actual batch ratio=%s count=%s",
                        domain, kl_n,
                        f"{ema_now:.4g}" if ema_now is not None else "—",
                        f"{ema_earlier:.4g}" if ema_earlier is not None else "—",
                        f"{sd*100:+.1f}%" if sd is not None else "—",
                        f"{sg:.3f}" if sg is not None else "—",
                        f"{target_ratio * 100:.1f}%" if target_ratio is not None else "—",
                        actual_ratio_str, count_str,
                    )
                if diag["updated"]:
                    logger.warning(
                        "ratios UPDATED at step %s: signal=%s ratio=%s",
                        diag["latest_step"],
                        {d: f"{s:.3f}" for d, s in diag["signal"].items()},
                        {d: f"{r * 100:.1f}%" for d, r in diag["ratio"].items()},
                    )
                else:
                    logger.info(
                        "ratios held (no update): %s | latest_step=%s last_update_step=%s",
                        diag["reason"], diag["latest_step"], state.last_update_step,
                    )

                if args.write_status:
                    status_dict = state.to_status_dict_dynamic_delta(exp_name=args.wandb_group)
                    write_status_atomic(status_path, status_dict)
                    logger.info(
                        "status written (dynamic-delta) ratio=%s",
                        {d: f"{(status_dict['domains'][d]['mixture_ratio'] or 0) * 100:.1f}%" for d in domain_map},
                    )
                else:
                    dryrun = state.to_status_dict_dynamic_delta(exp_name=args.wandb_group)
                    logger.info(
                        "DRY-RUN status (dynamic-delta) ratio=%s",
                        {d: f"{(dryrun['domains'][d]['mixture_ratio'] or 0) * 100:.1f}%" for d in domain_map},
                    )
            elif args.mode == "dynamic-composite":
                diag = state.step_dynamic_composite(
                    domain_kl,
                    ema_window=args.ema_window,
                    seed_points=args.initial_kl_seed_points,
                    update_every_steps=args.update_every_steps,
                    temperature=args.temperature,
                    ratio_floor=args.ratio_floor,
                    progress_smooth=args.progress_smooth,
                    progress_smooth_threshold=args.progress_smooth_threshold,
                    progress_smooth_scale=args.progress_smooth_scale,
                    progress_rolling_k=args.progress_rolling_k,
                    velocity_floor=args.progress_floor,
                    abs_kl_floor=args.abs_kl_floor,
                    rehearsal_domains=rehearsal_domains,
                )
                for domain in domain_map:
                    kl_n = len(domain_kl.get(domain, []))
                    last_count = domain_count.get(domain, [])
                    last_ratio = domain_ratio.get(domain, [])
                    count_str = f"{int(last_count[-1][1])}" if last_count else "—"
                    actual_ratio_str = f"{last_ratio[-1][1] * 100:.1f}%" if last_ratio else "—"
                    ikl = state.initial_kl.get(domain)
                    ema_now = state.current_ema_kl.get(domain)
                    ema_earlier = state.ema_earlier_per_domain.get(domain)
                    nrm = state.normalized_kl.get(domain)
                    sg = state.signal_per_domain.get(domain)
                    target_ratio = state.mixture_ratio.get(domain)
                    logger.info(
                        "%-4s | KL n=%-3d | initial=%s | EMA_now=%s | EMA_earlier=%s | normalized=%s | signal=%s | target_ratio=%s | actual batch ratio=%s count=%s",
                        domain, kl_n,
                        f"{ikl:.4g}" if ikl is not None else "—",
                        f"{ema_now:.4g}" if ema_now is not None else "—",
                        f"{ema_earlier:.4g}" if ema_earlier is not None else "—",
                        f"{nrm:.3f}" if nrm is not None else "—",
                        f"{sg:.3f}" if sg is not None else "—",
                        f"{target_ratio * 100:.1f}%" if target_ratio is not None else "—",
                        actual_ratio_str, count_str,
                    )
                if diag["updated"]:
                    logger.warning(
                        "ratios UPDATED at step %s: signal=%s ratio=%s",
                        diag["latest_step"],
                        {d: f"{s:.3f}" for d, s in diag["signal"].items()},
                        {d: f"{r * 100:.1f}%" for d, r in diag["ratio"].items()},
                    )
                else:
                    logger.info(
                        "ratios held (no update): %s | latest_step=%s last_update_step=%s",
                        diag["reason"], diag["latest_step"], state.last_update_step,
                    )

                if args.write_status:
                    status_dict = state.to_status_dict_dynamic_composite(exp_name=args.wandb_group)
                    write_status_atomic(status_path, status_dict)
                    logger.info(
                        "status written (dynamic-composite) ratio=%s",
                        {d: f"{(status_dict['domains'][d]['mixture_ratio'] or 0) * 100:.1f}%" for d in domain_map},
                    )
                else:
                    dryrun = state.to_status_dict_dynamic_composite(exp_name=args.wandb_group)
                    logger.info(
                        "DRY-RUN status (dynamic-composite) ratio=%s",
                        {d: f"{(dryrun['domains'][d]['mixture_ratio'] or 0) * 100:.1f}%" for d in domain_map},
                    )
            else:
                judgments = {}
                for domain in domain_map:
                    judgments[domain] = judge_domain(
                        domain_kl.get(domain, []), domain_reward.get(domain, []),
                        window_W=args.ema_window, kl_eps=args.kl_eps,
                        require_reward_gate=require_reward_gate,
                    )

                for domain in domain_map:
                    j = judgments[domain]
                    kl_n = len(domain_kl.get(domain, []))
                    status_str = "DOWN" if state.downsampled[domain] else f"pat{state.patience[domain]}/{args.patience}"
                    # Latest per-batch composition (count + ratio) — handy as a
                    # quick "is the filter actually downsampling?" check without
                    # opening wandb.
                    last_count = domain_count.get(domain, [])
                    last_ratio = domain_ratio.get(domain, [])
                    count_str = f"{int(last_count[-1][1])}" if last_count else "—"
                    ratio_str = f"{last_ratio[-1][1] * 100:.1f}%" if last_ratio else "—"
                    if args.reward_source == "none":
                        reward_str = "[KL-only mode]"
                    else:
                        r_n = len(domain_reward.get(domain, []))
                        stagnant = j.get("reward_stagnant")
                        src_label = "neg_kl" if args.reward_source == "neg_kl" else "reward"
                        reward_str = f"{src_label} n={r_n:<3d} stagnant={stagnant}"
                    logger.info(
                        "%-4s | step=%s | batch count=%-4s ratio=%-6s | KL n=%-3d EMA=%s Δ=%s plateau=%s | %s | converged=%s | %s",
                        domain,
                        j.get("step"),
                        count_str,
                        ratio_str,
                        kl_n,
                        f"{j['kl_ema']:.4g}" if j.get("kl_ema") is not None else "—",
                        f"{j['kl_rel_change']*100:.1f}%" if j.get("kl_rel_change") is not None else "—",
                        j.get("kl_plateau"),
                        reward_str,
                        j.get("converged"),
                        status_str,
                    )

                state.step(judgments, domain_reward)

                if args.mode == "static-ratio":
                    if args.write_status:
                        status_dict = state.to_status_dict_static_ratio(
                            exp_name=args.wandb_group, ratio_floor=args.ratio_floor,
                        )
                        write_status_atomic(status_path, status_dict)
                        logger.info(
                            "status written (static-ratio, n_converged=%d): %s",
                            status_dict["n_converged"],
                            {d: f"{status_dict['domains'][d]['mixture_ratio']*100:.1f}%" for d in domain_map},
                        )
                    else:
                        dryrun = state.to_status_dict_static_ratio(
                            exp_name=args.wandb_group, ratio_floor=args.ratio_floor,
                        )
                        logger.info(
                            "DRY-RUN status (static-ratio, n_converged=%d): %s",
                            dryrun["n_converged"],
                            {d: f"{dryrun['domains'][d]['mixture_ratio']*100:.1f}%" for d in domain_map},
                        )
                else:
                    if args.write_status:
                        status_dict = state.to_status_dict(exp_name=args.wandb_group)
                        write_status_atomic(status_path, status_dict)
                        logger.info("status written: %s",
                                    {d: s["downsampled"] for d, s in status_dict["domains"].items()})
                    else:
                        dryrun = state.to_status_dict(exp_name=args.wandb_group)
                        logger.info("DRY-RUN status: %s",
                                    {d: s["downsampled"] for d, s in dryrun["domains"].items()})

            if args.mode == "static" and args.enable_all_down_stop and args.write_status:
                all_down = all(state.downsampled[d] for d in domain_map)
                if all_down:
                    all_down_streak += 1
                    logger.warning(
                        "ALL-DOWN detected (streak=%d/%d)",
                        all_down_streak, args.all_down_stop_grace_polls,
                    )
                    if all_down_streak >= args.all_down_stop_grace_polls:
                        logger.warning(
                            "ALL-DOWN early-stop triggered: all %d domains downsampled "
                            "for %d consecutive polls. Filter would reject the vast "
                            "majority of every rollout group → teacher-load spike + "
                            "significant pace slowdown.",
                            len(domain_map), all_down_streak,
                        )
                        if args.all_down_stop_cmd:
                            try:
                                r = subprocess.run(
                                    args.all_down_stop_cmd,
                                    shell=True,
                                    capture_output=True, text=True, timeout=120,
                                )
                                logger.warning(
                                    "all-down-stop cmd %r → rc=%d stdout=%s stderr=%s",
                                    args.all_down_stop_cmd, r.returncode,
                                    r.stdout.strip()[:500], r.stderr.strip()[:500],
                                )
                            except Exception as exc:
                                logger.error("all-down-stop cmd failed: %s", exc)
                        else:
                            logger.warning(
                                "--all-down-stop-cmd not set; watcher exits but "
                                "operator must stop the training job manually.",
                            )
                        logger.warning("watcher exiting due to all-DOWN early-stop")
                        break
                else:
                    all_down_streak = 0

        except Exception as exc:
            logger.exception("poll failed: %s", exc)

        poll_count += 1
        if args.max_polls and poll_count >= args.max_polls:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()

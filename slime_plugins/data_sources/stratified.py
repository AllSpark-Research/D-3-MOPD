"""Stratified per-domain RolloutDataSource for D³-MOPD.

Wire via ``--data-source-path
slime_plugins.data_sources.stratified.StratifiedRolloutDataSourceWithBuffer``.

Default slime behavior is graceful: only experiments that pass this CLI flag
get stratified sampling; everything else (any baseline that does not pass the
flag) uses the stock random shuffle.

Semantics
---------
For each ``get_samples(N)`` call (N = number of prompts):
- Reads ``D3MOPD_DOMAIN_MAP`` env (already exported by the base scripts).
- Splits the merged ``--prompt-data`` jsonl by ``metadata.data_source`` into
  per-domain pools.
- Returns **exactly** ``N // n_domains`` prompts per domain, with the remainder
  going to lex-first domains. E.g. N=256, n_domains=3 → quotas math=86,
  code=85, if=85.
- Each per-domain pool is independently shuffled per epoch (different seed per
  domain so they don't lock-step).
- After fetching, prompts within the batch are re-shuffled so the order isn't
  sorted-by-domain (some downstream code may key off order).
- ``add_samples`` (called by slime to push aborted rollouts back) is accepted
  but the buffer is **never read** in stratified mode — preserving strict
  per-batch ratio is more important than re-using aborted samples.

Graceful fallback
-----------------
If ``D3MOPD_DOMAIN_MAP`` is unset, malformed, or no sample has
``metadata.data_source`` matching any known domain → falls back to
``RolloutDataSourceWithBuffer`` parent (= stock slime behavior). This means
the class is **safe to use as a default** in any wrapper — runs without a
domain map just get parent semantics.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from slime.rollout.data_source import RolloutDataSourceWithBuffer

if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = [
    "StratifiedRolloutDataSourceWithBuffer",
    "DynamicRatioStratifiedRolloutDataSourceWithBuffer",
]


def _parse_domain_map(raw: str) -> dict[str, list[str]]:
    """Parse ``math=ds1,ds2|code=ds3,ds4|if=ds5,ds6`` into ``{domain: [ds, ...]}``."""
    out: dict[str, list[str]] = {}
    for entry in (raw or "").split("|"):
        if "=" not in entry:
            continue
        domain, ds_csv = entry.split("=", 1)
        domain = domain.strip()
        if domain:
            out[domain] = [s.strip() for s in ds_csv.split(",") if s.strip()]
    return out


class StratifiedRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
    """Round-robin per-domain prompt fetcher with strict per-batch quotas."""

    def __init__(self, args) -> None:
        super().__init__(args)

        self._domain_map = _parse_domain_map(os.environ.get("D3MOPD_DOMAIN_MAP", ""))
        self._per_domain_pools: dict[str, list[Sample]] | None = None

        if not self._domain_map or self.dataset is None:
            logger.warning(
                "Stratified DataSource: domain map empty or dataset None; "
                "falling back to RolloutDataSourceWithBuffer behavior."
            )
            return

        # ds -> domain reverse lookup
        ds_to_domain: dict[str, str] = {}
        for domain, dss in self._domain_map.items():
            for ds in dss:
                ds_to_domain[ds] = domain

        pools: dict[str, list] = defaultdict(list)
        unmatched = 0
        for sample in self.dataset.samples:
            md = sample.metadata if isinstance(sample.metadata, dict) else {}
            ds = md.get("data_source")
            domain = ds_to_domain.get(ds)
            if domain:
                pools[domain].append(sample)
            else:
                unmatched += 1

        if not pools:
            logger.warning(
                "Stratified DataSource: no sample matched any domain in %s (%d samples unmatched); "
                "falling back to parent.",
                sorted(self._domain_map),
                unmatched,
            )
            return

        self._per_domain_pools = dict(pools)
        self._per_domain_offset: dict[str, int] = {d: 0 for d in self._per_domain_pools}
        self._per_domain_epoch: dict[str, int] = {d: 0 for d in self._per_domain_pools}

        base_seed = getattr(self.args, "rollout_seed", 42) or 42
        if self.args.rollout_shuffle:
            for i, domain in enumerate(sorted(self._per_domain_pools)):
                random.Random(base_seed + i * 1009).shuffle(self._per_domain_pools[domain])

        logger.info(
            "Stratified DataSource ready: domains=%s, pool sizes=%s, unmatched samples=%d",
            sorted(self._per_domain_pools),
            {d: len(p) for d, p in self._per_domain_pools.items()},
            unmatched,
        )

    def get_samples(self, num_samples: int):
        if self._per_domain_pools is None:
            # Graceful fallback when env / dataset / matching missing
            return super().get_samples(num_samples)

        prompt_samples = self._fetch_stratified_prompts(num_samples)

        # Optional within-batch shuffle so order isn't sorted-by-domain
        if self.args.rollout_shuffle:
            seed = (getattr(self.args, "rollout_seed", 42) or 42) + self.sample_group_index
            random.Random(seed).shuffle(prompt_samples)

        # Expand each prompt into a group of n_samples_per_prompt rollouts.
        # Logic mirrored from RolloutDataSource.get_samples post-fetch block.
        samples: list[list] = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                s = copy.deepcopy(prompt_sample)
                s.group_index = self.sample_group_index
                s.index = self.sample_index
                self.sample_index += 1
                group.append(s)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def _fetch_stratified_prompts(self, num_samples: int) -> list:
        """Pull N prompts with strict per-domain quotas; handles per-domain wrap-around."""
        domains_sorted = sorted(self._per_domain_pools)
        n_domains = len(domains_sorted)
        base = num_samples // n_domains
        remainder = num_samples % n_domains
        quotas = {d: base + (1 if i < remainder else 0) for i, d in enumerate(domains_sorted)}

        prompt_samples: list = []
        for domain in domains_sorted:
            quota = quotas[domain]
            pool = self._per_domain_pools[domain]
            offset = self._per_domain_offset[domain]

            if offset + quota <= len(pool):
                prompt_samples += pool[offset:offset + quota]
                self._per_domain_offset[domain] = offset + quota
            else:
                # Wrap around: take tail of current epoch, re-shuffle for new epoch, take head
                taken = pool[offset:]
                still_needed = quota - len(taken)
                self._per_domain_epoch[domain] += 1
                if self.args.rollout_shuffle:
                    seed = (
                        (getattr(self.args, "rollout_seed", 42) or 42)
                        + self._per_domain_epoch[domain] * 1009
                    )
                    random.Random(seed).shuffle(pool)
                prompt_samples += taken + pool[:still_needed]
                self._per_domain_offset[domain] = still_needed
        return prompt_samples

    def add_samples(self, samples) -> None:
        # Stratified mode: silently discard aborted samples instead of buffering.
        # Re-emitting them via the buffer would break strict per-batch ratios.
        # In steady state aborts are rare; if you see many, investigate teacher
        # health rather than try to recycle them.
        if samples:
            logger.debug(
                "Stratified DataSource: discarding %d aborted sample group(s) "
                "(buffer disabled to preserve per-batch quotas).",
                len(samples),
            )


class DynamicRatioStratifiedRolloutDataSourceWithBuffer(StratifiedRolloutDataSourceWithBuffer):
    """Stratified per-domain prompt fetcher with **dynamic** per-batch ratios.

    Wire via ``--data-source-path
    slime_plugins.data_sources.stratified.DynamicRatioStratifiedRolloutDataSourceWithBuffer``
    plus ``D3MOPD_STATUS_PATH`` env pointing at a status JSON written by either:
    - watcher ``--mode dynamic`` (writes ``mode: "dynamic"``, dynamic-gap algorithm)
    - watcher ``--mode dynamic-delta`` (writes ``mode: "dynamic-delta"``, dynamic-delta algorithm)
    - watcher ``--mode dynamic-composite`` (writes ``mode: "dynamic-composite"``, composite signal, v27+)

    All modes produce per-domain ``mixture_ratio`` fields, consumed identically here.

    Per ``get_samples(N)`` call:
    - Read status JSON; pull per-domain ``mixture_ratio``.
    - Convert ratios to integer per-domain quotas via largest-remainder rounding
      so quotas sum **exactly** to N.
    - Pull from per-domain pools with same wrap-around / per-epoch reshuffle
      logic as the parent.

    Graceful fallback (any of):
    - ``D3MOPD_STATUS_PATH`` env unset
    - status file missing / unreadable / malformed JSON
    - status JSON has ``mode`` not in {"dynamic", "dynamic-delta"} (e.g. an old
      static-mode file lying around, or a future-mode file from a different code version)
    - any pool-domain missing a positive ``mixture_ratio`` in the JSON

    → falls back to ``StratifiedRolloutDataSourceWithBuffer`` parent semantics
    (strict 1:1:1 equal quotas). The class is safe to ship as the default
    data-source path even on non-dynamic runs.
    """

    _ACCEPTED_MODES = ("dynamic", "dynamic-delta", "dynamic-composite", "static-ratio")

    @staticmethod
    def _read_jitter_strength() -> float:
        """Read D3MOPD_RATIO_JITTER env (default 0 = no jitter, backward compatible).

        jitter_strength α ∈ [0, 1) — per-batch mixture_ratio[d] is multiplied
        by (1 + U(-α, +α)) then renormalized. Expectation preserved (long-run
        mean ratio unchanged), only batch-level variance added.

        Use α=0.3 to recover roughly the same batch-level σ as random shuffle's
        hypergeometric variance over 3 domains.
        """
        raw = os.environ.get("D3MOPD_RATIO_JITTER", "0")
        try:
            v = float(raw)
        except ValueError:
            logger.warning(
                "D3MOPD_RATIO_JITTER=%r not parseable as float; using 0.", raw,
            )
            return 0.0
        if v < 0 or v >= 1:
            logger.warning(
                "D3MOPD_RATIO_JITTER=%s out of [0, 1) range; clamping to 0.", v,
            )
            return 0.0
        return v

    def _jitter_ratios(self, ratios: dict[str, float]) -> dict[str, float]:
        """Apply ±strength multiplicative jitter to each ratio, re-normalize to sum=1.

        Per-batch deterministic seed = base_seed + sample_group_index, so a given
        batch index produces the same jittered ratios across re-runs (useful for
        ablation reproducibility).
        """
        strength = self._read_jitter_strength()
        if strength <= 0:
            return ratios
        seed = (getattr(self.args, "rollout_seed", 42) or 42) + self.sample_group_index * 7919
        rng = random.Random(seed)
        jittered = {
            d: max(r * (1.0 + rng.uniform(-strength, strength)), 1e-6)
            for d, r in ratios.items()
        }
        total = sum(jittered.values())
        out = {d: r / total for d, r in jittered.items()}
        logger.info(
            "DynamicRatio jitter α=%.2f (batch=%d): base=%s -> jittered=%s",
            strength, self.sample_group_index,
            {d: f"{r * 100:.1f}%" for d, r in ratios.items()},
            {d: f"{r * 100:.1f}%" for d, r in out.items()},
        )
        return out

    def _read_ratios_from_status(self) -> dict[str, float] | None:
        status_path_str = os.environ.get("D3MOPD_STATUS_PATH")
        if not status_path_str:
            return None
        status_path = Path(status_path_str)
        try:
            with status_path.open() as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.debug(
                "DynamicRatio DataSource: cannot read %s (%s); falling back to equal quotas.",
                status_path, exc,
            )
            return None
        if payload.get("mode") not in self._ACCEPTED_MODES:
            logger.debug(
                "DynamicRatio DataSource: status JSON mode=%r not in %s; equal quotas.",
                payload.get("mode"), self._ACCEPTED_MODES,
            )
            return None
        domains_in = payload.get("domains")
        if not isinstance(domains_in, dict):
            return None
        ratios: dict[str, float] = {}
        for d, cfg in domains_in.items():
            if d not in (self._per_domain_pools or {}) or not isinstance(cfg, dict):
                continue
            r = cfg.get("mixture_ratio")
            if isinstance(r, (int, float)) and r > 0:
                ratios[d] = float(r)
        if not self._per_domain_pools or set(ratios) != set(self._per_domain_pools):
            missing = set(self._per_domain_pools or {}) - set(ratios)
            logger.warning(
                "DynamicRatio DataSource: status JSON missing mixture_ratio for %s; equal quotas.",
                sorted(missing),
            )
            return None
        total = sum(ratios.values())
        if total <= 0:
            return None
        return {d: r / total for d, r in ratios.items()}

    @staticmethod
    def _largest_remainder_quotas(
        ratios: dict[str, float], num_samples: int,
    ) -> dict[str, int]:
        """Hare-quota integer rounding: floors + give +1 to top-(N-Σfloor) by remainder.

        Output ``quotas`` satisfies ``sum(quotas.values()) == num_samples``.
        """
        domains = sorted(ratios)
        raw = {d: ratios[d] * num_samples for d in domains}
        floors = {d: int(raw[d]) for d in domains}
        leftover = num_samples - sum(floors.values())
        if leftover <= 0:
            return floors
        by_remainder = sorted(domains, key=lambda d: raw[d] - floors[d], reverse=True)
        quotas = dict(floors)
        for d in by_remainder[:leftover]:
            quotas[d] += 1
        return quotas

    def _fetch_stratified_prompts(self, num_samples: int) -> list:
        ratios = self._read_ratios_from_status()
        if ratios is None:
            # Warmup / no valid status: fall back to uniform per-domain quotas.
            # Still apply jitter env (if set > 0) so batch variance is present
            # from step 0 (continuous-jitter regime), rather than a "no jitter
            # until watcher writes first ratio" gap.
            # If jitter env is 0, this reduces to the parent's strict quotas.
            n = len(self._per_domain_pools or {})
            if n == 0:
                return super()._fetch_stratified_prompts(num_samples)
            ratios = {d: 1.0 / n for d in self._per_domain_pools}

        ratios = self._jitter_ratios(ratios)
        quotas = self._largest_remainder_quotas(ratios, num_samples)
        logger.info(
            "DynamicRatio quotas (N=%d): %s (ratios=%s)",
            num_samples, quotas,
            {d: f"{r * 100:.1f}%" for d, r in ratios.items()},
        )

        prompt_samples: list = []
        for domain in sorted(self._per_domain_pools):
            quota = quotas.get(domain, 0)
            if quota <= 0:
                continue
            pool = self._per_domain_pools[domain]
            offset = self._per_domain_offset[domain]
            if offset + quota <= len(pool):
                prompt_samples += pool[offset:offset + quota]
                self._per_domain_offset[domain] = offset + quota
            else:
                # Wrap around: tail of current epoch + (re-shuffle) + head of next epoch.
                taken = pool[offset:]
                still_needed = quota - len(taken)
                self._per_domain_epoch[domain] += 1
                if self.args.rollout_shuffle:
                    seed = (
                        (getattr(self.args, "rollout_seed", 42) or 42)
                        + self._per_domain_epoch[domain] * 1009
                    )
                    random.Random(seed).shuffle(pool)
                prompt_samples += taken + pool[:still_needed]
                self._per_domain_offset[domain] = still_needed
        return prompt_samples

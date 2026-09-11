"""D³-MOPD dynamic-domain-downsample filter.

Wire via ``--dynamic-sampling-filter-path
slime_plugins.filters.d3mopd_downsample_filter.d3mopd_downsample_filter``.

State file path is read from the ``D3MOPD_STATUS_PATH`` env so we do not need
to touch ``slime/utils/arguments.py``. Runs that do not set the env or do not
pass this filter are unaffected — the filter has no effect when the env is
unset, when the status file is missing or unreadable, or when the sample's
data_source is not listed in any domain.

Status file format (written by the external watcher):

    {
      "exp_name": "...",
      "updated_at": "...",
      "domains": {
        "math": {"downsampled": true, "rehearsal_rate": 0.05,
                  "data_sources": ["math_dapo", "acereason_math"]},
        "code": {"downsampled": false, "rehearsal_rate": 1.0,
                  "data_sources": [...]},
        "if":   {"downsampled": false, "rehearsal_rate": 1.0,
                  "data_sources": [...]}
      }
    }

Each filter call receives one group (n_samples_per_prompt clones of the same
prompt), so every sample in the group shares the same data_source.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = ["d3mopd_downsample_filter"]


def _load_status() -> dict[str, dict[str, Any]] | None:
    status_path_str = os.environ.get("D3MOPD_STATUS_PATH")
    if not status_path_str:
        return None
    status_path = Path(status_path_str)
    try:
        with status_path.open() as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.debug("d3mopd_downsample_filter: could not read %s (%s); keeping group", status_path, exc)
        return None
    domains = payload.get("domains")
    if not isinstance(domains, dict):
        return None
    return domains


def d3mopd_downsample_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    domains = _load_status()
    if domains is None:
        return DynamicFilterOutput(keep=True)

    # Build data_source -> (domain, downsampled, rehearsal_rate) reverse lookup
    ds_lookup: dict[str, tuple[str, bool, float]] = {}
    for domain, cfg in domains.items():
        if not isinstance(cfg, dict):
            continue
        downsampled = bool(cfg.get("downsampled", False))
        rate = float(cfg.get("rehearsal_rate", 1.0))
        for ds in cfg.get("data_sources", []):
            ds_lookup[ds] = (domain, downsampled, rate)

    if not samples:
        return DynamicFilterOutput(keep=True)

    first = samples[0]
    metadata = first.metadata if isinstance(first.metadata, dict) else {}
    data_source = metadata.get("data_source")
    if not data_source or data_source not in ds_lookup:
        return DynamicFilterOutput(keep=True)

    domain, downsampled, rate = ds_lookup[data_source]
    if not downsampled:
        return DynamicFilterOutput(keep=True)

    # Domain is downsampled: keep with probability ``rehearsal_rate``.
    if random.random() < rate:
        return DynamicFilterOutput(keep=True, reason=f"d3_rehearsal_{domain}")
    return DynamicFilterOutput(keep=False, reason=f"d3_downsampled_{domain}")

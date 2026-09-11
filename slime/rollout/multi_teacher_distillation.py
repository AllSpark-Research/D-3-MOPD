"""Multi-teacher source-routed on-policy distillation reward functions.

Core reward path for D³-MOPD (Domain-adaptive Dynamic Distillation with
Multi-teacher On-Policy Distillation).

Each sample is routed to the teacher whose domain matches the sample's
`metadata["data_source"]` (or any other configured route key). Unlike the
stock single-teacher `slime.rollout.on_policy_distillation`, this module
reads a domain→URL mapping from env `OPD_DOMAIN_TEACHER_ROUTES` (format:
`domain_a=http://host:port/generate|domain_b=http://...|...`).

Uses stock `post_process_rewards` from `slime.rollout.on_policy_distillation`
unchanged — it consumes the sglang logprob response and writes
`sample.teacher_log_probs`.
"""

from __future__ import annotations

import os

import aiohttp

from slime.rollout.on_policy_distillation import post_process_rewards  # noqa: F401  re-export
from slime.utils.processing_utils import encode_image_for_rollout_engine


def _load_domain_routes() -> dict[str, str]:
    raw = os.environ.get("OPD_DOMAIN_TEACHER_ROUTES", "").strip()
    if not raw:
        return {}
    routes: dict[str, str] = {}
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"OPD_DOMAIN_TEACHER_ROUTES entries must be `domain=url`, got {part!r}"
            )
        domain, url = part.split("=", 1)
        routes[domain.strip()] = url.strip()
    return routes


_ROUTE_KEY = os.environ.get("OPD_TEACHER_ROUTE_KEY", "data_source")
_DOMAIN_ROUTES = _load_domain_routes()
_SINGLE_TEACHER_URL = os.environ.get("OPD_TEACHER_URL", "").strip()


def _resolve_teacher_url(args, sample) -> str:
    """Pick teacher URL for this sample.

    Priority:
      1. OPD_TEACHER_URL env (single-teacher override, useful for debugging).
      2. OPD_DOMAIN_TEACHER_ROUTES env mapping by sample.metadata[route_key].
      3. args.rm_url (slime's stock single-teacher fallback).
    """
    if _SINGLE_TEACHER_URL:
        return _SINGLE_TEACHER_URL
    if _DOMAIN_ROUTES:
        meta = sample.metadata or {}
        key = meta.get(_ROUTE_KEY)
        if key is None:
            raise KeyError(
                f"sample.metadata has no `{_ROUTE_KEY}` (needed for multi-teacher routing); "
                f"metadata keys={list(meta)}"
            )
        url = _DOMAIN_ROUTES.get(key)
        if url is None:
            raise KeyError(
                f"no teacher route for {_ROUTE_KEY}={key!r}; configured routes={list(_DOMAIN_ROUTES)}"
            )
        return url
    if getattr(args, "rm_url", None):
        return args.rm_url
    raise RuntimeError(
        "no teacher URL resolvable — set OPD_DOMAIN_TEACHER_ROUTES, OPD_TEACHER_URL, or --rm-url"
    )


async def reward_func(args, sample, **kwargs):
    """Async per-sample teacher call. Routes to the per-domain teacher URL."""
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    teacher_url = _resolve_teacher_url(args, sample)
    async with aiohttp.ClientSession() as session:
        async with session.post(teacher_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

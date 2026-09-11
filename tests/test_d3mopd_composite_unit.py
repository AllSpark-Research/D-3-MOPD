"""Unit tests for D³-MOPD dynamic-composite mode robustness knobs.

Covers the two backward-compatible Patches added on top of the paper's
composite = normalized_gap × velocity formulation:

  • Patch A — ``velocity_floor`` (aka ``--progress-floor``): retains a
    gap-only fallback when a domain's KL rebounds (velocity ≤ 0). Default
    0.0 must be bit-exact with the paper formulation.
  • Patch B — ``abs_kl_floor`` / ``rehearsal_domains`` (aka
    ``--abs-kl-floor`` / ``--rehearsal-domains``): forces a rehearsal-style
    domain (frozen-student teacher, initial_kl ≈ 0) onto the ``ratio_floor``
    share instead of letting its exploded normalized gap hog the mixture.
    Defaults (empty set, 0.0) must be no-ops.

Run from the repo root:
    python -m pytest tests/test_d3mopd_composite_unit.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


# watcher.py is a script (no __init__.py in tools/d3mopd/); load by file path.
_watcher_path = PKG_ROOT / "tools" / "d3mopd" / "watcher.py"
_spec = importlib.util.spec_from_file_location("d3mopd_watcher", _watcher_path)
_watcher_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_watcher_module)
compute_composite_ratios = _watcher_module.compute_composite_ratios


# ---------------- shared toy setup ----------------
# 3 domains, 2 in normal learning territory + 1 near-zero baseline that
# stands in for a frozen-student "rehearsal" teacher.
INITIAL_KL_NORMAL = {"math": 0.10, "code": 0.10, "if": 0.10}
EMA_NOW_NORMAL = {"math": 0.05, "code": 0.02, "if": 0.09}   # math half, code learned a lot, if still full
EMA_EARLIER_NORMAL = {"math": 0.07, "code": 0.03, "if": 0.10}  # code+math falling, if flat


# ---------------- Patch A: velocity_floor ----------------


def test_velocity_floor_default_is_bit_exact():
    """velocity_floor=0.0 must reproduce the paper formula exactly."""
    baseline = compute_composite_ratios(
        ema_now_per_domain=EMA_NOW_NORMAL,
        ema_earlier_per_domain=EMA_EARLIER_NORMAL,
        initial_kl_per_domain=INITIAL_KL_NORMAL,
        temperature=0.5,
        ratio_floor=0.10,
    )
    with_zero_floor = compute_composite_ratios(
        ema_now_per_domain=EMA_NOW_NORMAL,
        ema_earlier_per_domain=EMA_EARLIER_NORMAL,
        initial_kl_per_domain=INITIAL_KL_NORMAL,
        temperature=0.5,
        ratio_floor=0.10,
        velocity_floor=0.0,
    )
    for d in baseline:
        for k in ("ratio", "normalized", "progress", "raw_signal", "signal"):
            assert baseline[d][k] == with_zero_floor[d][k], f"mismatch on {d}/{k}"


def test_velocity_floor_saves_a_rebounding_domain():
    """A domain whose KL rose (velocity < 0 → progress = 0 under default)
    but still has a large remaining gap should get > floor share when
    velocity_floor > 0."""
    # math: gap ~0.6 remaining but KL rebounded (ema_now > ema_earlier)
    # code: gap much smaller, actively descending
    ema_earlier = {"math": 0.05, "code": 0.03}
    ema_now = {"math": 0.06, "code": 0.02}
    initial_kl = {"math": 0.10, "code": 0.10}
    common = dict(
        ema_now_per_domain=ema_now,
        ema_earlier_per_domain=ema_earlier,
        initial_kl_per_domain=initial_kl,
        temperature=0.5,
        ratio_floor=0.10,
    )
    without = compute_composite_ratios(**common, velocity_floor=0.0)
    with_floor = compute_composite_ratios(**common, velocity_floor=0.3)

    # Under default, math's velocity is clipped at 0 → raw_signal=0 → math on floor.
    assert without["math"]["progress"] == 0.0
    assert without["math"]["ratio"] == pytest.approx(0.10, abs=1e-6) or \
           without["math"]["ratio"] < without["code"]["ratio"]

    # With floor, math's progress is ≥ 0.3 → raw_signal > 0 → math > floor.
    assert with_floor["math"]["progress"] >= 0.3 - 1e-9
    assert with_floor["math"]["ratio"] > 0.10 + 1e-6
    # Sum invariant still holds.
    assert abs(sum(r["ratio"] for r in with_floor.values()) - 1.0) < 1e-9


def test_velocity_floor_upper_bound_still_reaches_one():
    """velocity_floor=0.0 with a fast-descending domain → progress ≈ 1;
    velocity_floor=1.0 → progress ≡ 1 for everyone → shape is dictated
    purely by normalized_gap (equivalent to dyn-gap mode)."""
    ema_earlier = {"math": 0.10, "code": 0.10}
    ema_now = {"math": 0.01, "code": 0.05}    # both descending
    initial_kl = {"math": 0.10, "code": 0.10}
    result = compute_composite_ratios(
        ema_now_per_domain=ema_now, ema_earlier_per_domain=ema_earlier,
        initial_kl_per_domain=initial_kl, temperature=0.5, ratio_floor=0.10,
        velocity_floor=1.0,
    )
    for d in result:
        assert result[d]["progress"] == pytest.approx(1.0, abs=1e-9)


# ---------------- Patch B: rehearsal / abs_kl_floor ----------------


def test_rehearsal_default_is_bit_exact():
    """No rehearsal_domains and abs_kl_floor=0 must be a no-op."""
    baseline = compute_composite_ratios(
        ema_now_per_domain=EMA_NOW_NORMAL,
        ema_earlier_per_domain=EMA_EARLIER_NORMAL,
        initial_kl_per_domain=INITIAL_KL_NORMAL,
        temperature=0.5,
        ratio_floor=0.10,
    )
    with_defaults = compute_composite_ratios(
        ema_now_per_domain=EMA_NOW_NORMAL,
        ema_earlier_per_domain=EMA_EARLIER_NORMAL,
        initial_kl_per_domain=INITIAL_KL_NORMAL,
        temperature=0.5,
        ratio_floor=0.10,
        abs_kl_floor=0.0,
        rehearsal_domains=frozenset(),
    )
    for d in baseline:
        assert baseline[d]["ratio"] == with_defaults[d]["ratio"]
        assert with_defaults[d]["is_rehearsal"] is False


def test_explicit_rehearsal_tag_forces_floor():
    """A domain explicitly tagged as rehearsal lands on ratio_floor even if
    its normalized gap would otherwise dominate the mixture."""
    # Simulate a self-teacher setup: if's initial_kl ≈ 0 → normalized explodes.
    initial_kl = {"math": 0.10, "code": 0.10, "if": 1e-4}
    ema_earlier = {"math": 0.09, "code": 0.09, "if": 5e-4}
    ema_now = {"math": 0.07, "code": 0.06, "if": 3e-4}    # if "learning" against near-zero baseline

    unfixed = compute_composite_ratios(
        ema_now_per_domain=ema_now, ema_earlier_per_domain=ema_earlier,
        initial_kl_per_domain=initial_kl, temperature=0.5, ratio_floor=0.10,
    )
    # Confirm the pathology: `if` hogs the mixture because normalized ≈ 3.
    assert unfixed["if"]["ratio"] > unfixed["math"]["ratio"]
    assert unfixed["if"]["ratio"] > unfixed["code"]["ratio"]

    fixed = compute_composite_ratios(
        ema_now_per_domain=ema_now, ema_earlier_per_domain=ema_earlier,
        initial_kl_per_domain=initial_kl, temperature=0.5, ratio_floor=0.10,
        rehearsal_domains=frozenset({"if"}),
    )
    # `if` is on floor; the other two split the remainder.
    assert fixed["if"]["is_rehearsal"] is True
    assert fixed["if"]["ratio"] == pytest.approx(0.10, abs=1e-6)
    assert fixed["math"]["is_rehearsal"] is False
    assert fixed["code"]["is_rehearsal"] is False
    assert fixed["math"]["ratio"] > 0.10
    assert fixed["code"]["ratio"] > 0.10
    assert abs(sum(r["ratio"] for r in fixed.values()) - 1.0) < 1e-9


def test_abs_kl_floor_auto_detects_rehearsal():
    """A domain with initial_kl below abs_kl_floor should be treated the
    same as if it were explicitly tagged."""
    initial_kl = {"math": 0.10, "code": 0.10, "if": 5e-4}
    ema_earlier = {"math": 0.09, "code": 0.09, "if": 8e-4}
    ema_now = {"math": 0.07, "code": 0.06, "if": 6e-4}

    result = compute_composite_ratios(
        ema_now_per_domain=ema_now, ema_earlier_per_domain=ema_earlier,
        initial_kl_per_domain=initial_kl, temperature=0.5, ratio_floor=0.10,
        abs_kl_floor=1e-3,
    )
    assert result["if"]["is_rehearsal"] is True
    assert result["if"]["ratio"] == pytest.approx(0.10, abs=1e-6)
    assert result["math"]["is_rehearsal"] is False
    assert result["code"]["is_rehearsal"] is False


def test_all_rehearsal_degrades_to_uniform_floor():
    """If every domain is rehearsal-gated, softmax collapses to uniform and
    every domain still ends up on the floor share (equal split of what floor
    guarantees)."""
    result = compute_composite_ratios(
        ema_now_per_domain=EMA_NOW_NORMAL,
        ema_earlier_per_domain=EMA_EARLIER_NORMAL,
        initial_kl_per_domain=INITIAL_KL_NORMAL,
        temperature=0.5,
        ratio_floor=0.10,
        rehearsal_domains=frozenset({"math", "code", "if"}),
    )
    # All raw_signal = 0 → max_raw ≈ 0 → signal_norm = 0 → softmax uniform
    # → ratio = floor + (1 - 3*floor) / 3 = 1/3 each.
    for d, r in result.items():
        assert r["is_rehearsal"] is True
        assert r["ratio"] == pytest.approx(1.0 / 3, abs=1e-6)


def test_patches_compose():
    """velocity_floor and rehearsal_domains stack cleanly: gated domain sits
    on floor irrespective of velocity_floor value."""
    initial_kl = {"math": 0.10, "code": 0.10, "if": 1e-4}
    ema_earlier = {"math": 0.05, "code": 0.03, "if": 5e-4}
    ema_now = {"math": 0.06, "code": 0.02, "if": 3e-4}   # math rebounding

    result = compute_composite_ratios(
        ema_now_per_domain=ema_now, ema_earlier_per_domain=ema_earlier,
        initial_kl_per_domain=initial_kl, temperature=0.5, ratio_floor=0.10,
        velocity_floor=0.3,
        rehearsal_domains=frozenset({"if"}),
    )
    # `if` sits on floor despite the velocity_floor boost applying to progress.
    assert result["if"]["ratio"] == pytest.approx(0.10, abs=1e-6)
    # math (rebounding) still benefits from the velocity_floor patch.
    assert result["math"]["ratio"] > 0.10 + 1e-6

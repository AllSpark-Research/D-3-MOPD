"""Unit tests for D³-MOPD dynamic-mode ratio computation and data source.

Pure-function tests — no wandb, no slime args, no training infra needed.
Run from the repo root:
    python -m pytest tests/test_d3mopd_dynamic_unit.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# INNER slime dir (contains the `slime/` and `slime_plugins/` packages).
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


# ---------------- watcher.py is a script (no __init__.py in tools/d3mopd/) ----------------
# Load it directly from its file path via importlib.
_watcher_path = PKG_ROOT / "tools" / "d3mopd" / "watcher.py"
_spec = importlib.util.spec_from_file_location("d3mopd_watcher", _watcher_path)
_watcher_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_watcher_module)
compute_dynamic_ratios = _watcher_module.compute_dynamic_ratios
WriterState = _watcher_module.WriterState
write_status_atomic = _watcher_module.write_status_atomic


def test_compute_dynamic_ratios_sum_equals_one():
    ratios = compute_dynamic_ratios(
        ema_kl_per_domain={"math": 0.5, "code": 0.3, "if": 0.7},
        initial_kl_per_domain={"math": 1.0, "code": 1.0, "if": 1.0},
        temperature=1.0,
        ratio_floor=0.05,
    )
    assert abs(sum(ratios.values()) - 1.0) < 1e-9


def test_compute_dynamic_ratios_floor_enforced():
    # One domain learned perfectly (current=0); without floor it'd get ~0.
    ratios = compute_dynamic_ratios(
        ema_kl_per_domain={"math": 0.0, "code": 1.0, "if": 1.0},
        initial_kl_per_domain={"math": 1.0, "code": 1.0, "if": 1.0},
        temperature=1.0,
        ratio_floor=0.05,
    )
    for d, r in ratios.items():
        assert r >= 0.05 - 1e-9, f"{d}={r} below floor"
    assert abs(sum(ratios.values()) - 1.0) < 1e-9


def test_compute_dynamic_ratios_equal_inputs_give_equal_ratios():
    ratios = compute_dynamic_ratios(
        ema_kl_per_domain={"math": 0.5, "code": 0.5, "if": 0.5},
        initial_kl_per_domain={"math": 1.0, "code": 1.0, "if": 1.0},
        temperature=1.0,
        ratio_floor=0.05,
    )
    for r in ratios.values():
        assert abs(r - 1 / 3) < 1e-9


def test_compute_dynamic_ratios_laggard_gets_more():
    # if-domain has barely learned (normalized=0.9); math/code at 0.4.
    ratios = compute_dynamic_ratios(
        ema_kl_per_domain={"math": 0.4, "code": 0.4, "if": 0.9},
        initial_kl_per_domain={"math": 1.0, "code": 1.0, "if": 1.0},
        temperature=1.0,
        ratio_floor=0.05,
    )
    assert ratios["if"] > ratios["math"]
    assert ratios["if"] > ratios["code"]
    assert abs(ratios["math"] - ratios["code"]) < 1e-9  # symmetric


def test_compute_dynamic_ratios_lower_temp_sharper():
    args = dict(
        ema_kl_per_domain={"math": 0.4, "code": 0.7, "if": 0.9},
        initial_kl_per_domain={"math": 1.0, "code": 1.0, "if": 1.0},
        ratio_floor=0.05,
    )
    r_warm = compute_dynamic_ratios(temperature=2.0, **args)
    r_cool = compute_dynamic_ratios(temperature=0.5, **args)
    # Lower temperature → sharper allocation toward laggard (if)
    assert r_cool["if"] > r_warm["if"]
    # And less toward leader (math)
    assert r_cool["math"] < r_warm["math"]


def test_compute_dynamic_ratios_initial_kl_normalizes_scales():
    # Two scenarios where the SHAPE of progress is identical but absolute
    # KL is at very different scales — ratios should be the same.
    r_small = compute_dynamic_ratios(
        ema_kl_per_domain={"math": 0.04, "code": 0.07, "if": 0.09},
        initial_kl_per_domain={"math": 0.1, "code": 0.1, "if": 0.1},
        temperature=1.0,
        ratio_floor=0.05,
    )
    r_big = compute_dynamic_ratios(
        ema_kl_per_domain={"math": 4.0, "code": 7.0, "if": 9.0},
        initial_kl_per_domain={"math": 10.0, "code": 10.0, "if": 10.0},
        temperature=1.0,
        ratio_floor=0.05,
    )
    for d in ("math", "code", "if"):
        assert abs(r_small[d] - r_big[d]) < 1e-9


def test_compute_dynamic_ratios_infeasible_floor():
    with pytest.raises(ValueError):
        compute_dynamic_ratios(
            ema_kl_per_domain={"math": 1, "code": 1, "if": 1},
            initial_kl_per_domain={"math": 1, "code": 1, "if": 1},
            temperature=1.0,
            ratio_floor=0.4,  # 3 * 0.4 = 1.2 > 1
        )


# ---------------- P0': WriterState.step_dynamic ----------------


def _kl_series(values: list[float], start_step: int = 0):
    return [(start_step + i, v) for i, v in enumerate(values)]


def test_step_dynamic_waits_for_seed_points():
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": [], "code": [], "if": []},
    )
    diag = state.step_dynamic(
        domain_kl={
            "math": _kl_series([1.0, 0.9]),  # only 2 points, seed_points=5 required
            "code": _kl_series([1.0, 0.9]),
            "if": _kl_series([1.0, 0.9]),
        },
        ema_window=5, seed_points=5, update_every_steps=10,
        temperature=1.0, ratio_floor=0.05,
    )
    assert not diag["updated"]
    assert "awaiting_initial_kl_lock" in diag["reason"]
    # initial_kl still None
    assert all(state.initial_kl[d] is None for d in state.domain_map)


def test_step_dynamic_locks_initial_kl_and_updates():
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": [], "code": [], "if": []},
    )
    # 20 KL points per domain: math learns well (1.0 → 0.4), code OK (1.0 → 0.6),
    # if barely learns (1.0 → 0.85).
    domain_kl = {
        "math": _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.4] * 15),
        "code": _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.6] * 15),
        "if":   _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.85] * 15),
    }
    diag = state.step_dynamic(
        domain_kl=domain_kl,
        ema_window=5, seed_points=5, update_every_steps=10,
        temperature=1.0, ratio_floor=0.05,
    )
    assert diag["updated"], diag["reason"]
    # initial_kl = mean of first 5 = (1.0+0.95+0.9+0.85+0.8)/5 = 0.9
    for d in state.domain_map:
        assert abs(state.initial_kl[d] - 0.9) < 1e-9
    # Ratios: math < code < if (laggard gets more)
    r = diag["ratio"]
    assert r["math"] < r["code"] < r["if"]
    assert abs(sum(r.values()) - 1.0) < 1e-9


def test_step_dynamic_respects_update_cadence():
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": [], "code": [], "if": []},
    )
    domain_kl = {
        d: _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.5] * 15) for d in ("math", "code", "if")
    }
    # First call: lock initial_kl + compute ratios at latest_step=19.
    diag1 = state.step_dynamic(
        domain_kl=domain_kl, ema_window=5, seed_points=5,
        update_every_steps=10, temperature=1.0, ratio_floor=0.05,
    )
    assert diag1["updated"]
    assert state.last_update_step == 19

    # Second call: data unchanged (still latest_step=19). Next update_step
    # is 29; 19 < 29 → should NOT update.
    diag2 = state.step_dynamic(
        domain_kl=domain_kl, ema_window=5, seed_points=5,
        update_every_steps=10, temperature=1.0, ratio_floor=0.05,
    )
    assert not diag2["updated"]
    assert "waiting" in diag2["reason"]


def test_step_dynamic_persistence_roundtrip(tmp_path):
    """Lock initial_kl, write status, load it back — initial_kl preserved."""
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": ["m_ds"], "code": ["c_ds"], "if": ["i_ds"]},
    )
    domain_kl = {
        d: _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.5] * 15) for d in ("math", "code", "if")
    }
    state.step_dynamic(
        domain_kl=domain_kl, ema_window=5, seed_points=5,
        update_every_steps=10, temperature=1.0, ratio_floor=0.05,
    )
    status_path = tmp_path / "status.json"
    write_status_atomic(status_path, state.to_status_dict_dynamic(exp_name="exp_x"))

    state2 = WriterState.from_status_file_dynamic(
        status_path, domain_map={"math": ["m_ds"], "code": ["c_ds"], "if": ["i_ds"]},
    )
    for d in ("math", "code", "if"):
        assert state2.initial_kl[d] == state.initial_kl[d]
        assert state2.mixture_ratio[d] == state.mixture_ratio[d]
    assert state2.last_update_step == state.last_update_step


def test_from_status_file_dynamic_ignores_static_mode_json(tmp_path):
    """Old static-mode status JSON should NOT be re-interpreted as dynamic."""
    static_payload = {
        "exp_name": "old",
        "domains": {
            "math": {"downsampled": True, "rehearsal_rate": 0.05, "data_sources": ["m"]},
        },
    }
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(static_payload))
    state = WriterState.from_status_file_dynamic(
        status_path, domain_map={"math": ["m"]},
    )
    # Should start fresh — initial_kl still None
    assert state.initial_kl["math"] is None
    assert state.mixture_ratio == {}


# ---------------- P2: DynamicRatioStratifiedRolloutDataSourceWithBuffer ----------------

from slime_plugins.data_sources.stratified import (
    DynamicRatioStratifiedRolloutDataSourceWithBuffer as Dyn,
)


def test_largest_remainder_quotas_sum_equals_N():
    quotas = Dyn._largest_remainder_quotas(
        ratios={"math": 0.40, "code": 0.27, "if": 0.33}, num_samples=256,
    )
    assert sum(quotas.values()) == 256


def test_largest_remainder_quotas_extreme_ratio():
    quotas = Dyn._largest_remainder_quotas(
        ratios={"math": 0.05, "code": 0.05, "if": 0.90}, num_samples=256,
    )
    assert sum(quotas.values()) == 256
    assert quotas["if"] > quotas["math"]
    assert quotas["if"] > quotas["code"]


def test_largest_remainder_quotas_equal_split():
    quotas = Dyn._largest_remainder_quotas(
        ratios={"math": 1/3, "code": 1/3, "if": 1/3}, num_samples=256,
    )
    assert sum(quotas.values()) == 256
    # Each ~85; absolute diff between max and min is at most 1
    assert max(quotas.values()) - min(quotas.values()) <= 1


def test_largest_remainder_quotas_handles_small_N():
    quotas = Dyn._largest_remainder_quotas(
        ratios={"math": 0.40, "code": 0.27, "if": 0.33}, num_samples=10,
    )
    assert sum(quotas.values()) == 10
    # All non-negative
    for v in quotas.values():
        assert v >= 0


def test_largest_remainder_quotas_2_domains():
    quotas = Dyn._largest_remainder_quotas(
        ratios={"math": 0.7, "code": 0.3}, num_samples=100,
    )
    assert sum(quotas.values()) == 100
    assert quotas["math"] == 70 and quotas["code"] == 30


def test_read_ratios_from_status_env_unset(monkeypatch, tmp_path):
    """No D3MOPD_STATUS_PATH → returns None (caller falls back to equal)."""
    monkeypatch.delenv("D3MOPD_STATUS_PATH", raising=False)
    instance = Dyn.__new__(Dyn)  # bypass parent __init__
    instance._per_domain_pools = {"math": [], "code": [], "if": []}
    assert instance._read_ratios_from_status() is None


def test_read_ratios_from_status_static_mode_file(monkeypatch, tmp_path):
    """Status file in static mode → returns None (don't misuse old file)."""
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"domains": {"math": {"downsampled": True}}}))
    monkeypatch.setenv("D3MOPD_STATUS_PATH", str(status_path))
    instance = Dyn.__new__(Dyn)
    instance._per_domain_pools = {"math": []}
    assert instance._read_ratios_from_status() is None


def test_read_ratios_from_status_valid(monkeypatch, tmp_path):
    payload = {
        "mode": "dynamic",
        "domains": {
            "math": {"mixture_ratio": 0.4, "data_sources": ["m"]},
            "code": {"mixture_ratio": 0.27, "data_sources": ["c"]},
            "if":   {"mixture_ratio": 0.33, "data_sources": ["i"]},
        },
    }
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(payload))
    monkeypatch.setenv("D3MOPD_STATUS_PATH", str(status_path))
    instance = Dyn.__new__(Dyn)
    instance._per_domain_pools = {"math": [], "code": [], "if": []}
    ratios = instance._read_ratios_from_status()
    assert ratios is not None
    assert abs(sum(ratios.values()) - 1.0) < 1e-9
    # Order-independent comparison
    assert abs(ratios["math"] - 0.40) < 1e-9
    assert abs(ratios["if"] - 0.33) < 1e-9


def test_read_ratios_from_status_accepts_dynamic_delta_mode(monkeypatch, tmp_path):
    """Regression: dynamic-delta writes mode='dynamic-delta' but an earlier
    version of the data source only accepted mode='dynamic', silently falling
    back to equal 1:1:1. This test guards: both 'dynamic' and 'dynamic-delta'
    modes are accepted.
    """
    payload = {
        "mode": "dynamic-delta",  # ← the previously-rejected mode
        "domains": {
            "math": {"mixture_ratio": 0.4, "data_sources": ["m"]},
            "code": {"mixture_ratio": 0.3, "data_sources": ["c"]},
            "if":   {"mixture_ratio": 0.3, "data_sources": ["i"]},
        },
    }
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(payload))
    monkeypatch.setenv("D3MOPD_STATUS_PATH", str(status_path))
    instance = Dyn.__new__(Dyn)
    instance._per_domain_pools = {"math": [], "code": [], "if": []}
    ratios = instance._read_ratios_from_status()
    assert ratios is not None, "dynamic-delta mode must be accepted"
    assert abs(sum(ratios.values()) - 1.0) < 1e-9
    assert abs(ratios["math"] - 0.40) < 1e-9


def test_read_ratios_from_status_rejects_unknown_mode(monkeypatch, tmp_path):
    """Future/unknown mode → safe fallback to equal quotas (not crash)."""
    payload = {
        "mode": "dynamic-future-experimental",
        "domains": {"math": {"mixture_ratio": 1.0}},
    }
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(payload))
    monkeypatch.setenv("D3MOPD_STATUS_PATH", str(status_path))
    instance = Dyn.__new__(Dyn)
    instance._per_domain_pools = {"math": []}
    assert instance._read_ratios_from_status() is None


def test_read_ratios_from_status_missing_one_domain(monkeypatch, tmp_path):
    """Status file missing one of our pool domains → returns None (safe fallback)."""
    payload = {
        "mode": "dynamic",
        "domains": {
            "math": {"mixture_ratio": 0.5},
            # 'code' missing
            "if":   {"mixture_ratio": 0.5},
        },
    }
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(payload))
    monkeypatch.setenv("D3MOPD_STATUS_PATH", str(status_path))
    instance = Dyn.__new__(Dyn)
    instance._per_domain_pools = {"math": [], "code": [], "if": []}
    assert instance._read_ratios_from_status() is None

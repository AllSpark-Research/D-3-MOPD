"""Unit tests for D³-MOPD dynamic-delta mode (velocity-only signal variant)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Load watcher (script, not package member)
_watcher_path = PKG_ROOT / "tools" / "d3mopd" / "watcher.py"
_spec = importlib.util.spec_from_file_location("d3mopd_watcher", _watcher_path)
_watcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_watcher)
compute_delta_softmax_ratios = _watcher.compute_delta_softmax_ratios
WriterState = _watcher.WriterState
write_status_atomic = _watcher.write_status_atomic


def _kl_series(values: list[float], start_step: int = 0):
    return [(start_step + i, v) for i, v in enumerate(values)]


# ---------------- compute_delta_softmax_ratios ----------------


def test_delta_ratios_sum_equals_one_and_floor_enforced():
    r = compute_delta_softmax_ratios(
        ema_now_per_domain={"math": 0.5, "code": 0.4, "if": 0.6},
        ema_earlier_per_domain={"math": 0.6, "code": 0.7, "if": 0.65},
        temperature=1.0, ratio_floor=0.05,
    )
    ratios = {d: v["ratio"] for d, v in r.items()}
    assert abs(sum(ratios.values()) - 1.0) < 1e-9
    for v in ratios.values():
        assert v >= 0.05 - 1e-9


def test_delta_all_plateau_degrades_to_uniform():
    """When all signed_delta ≈ 0, signal=0, softmax(0)=uniform → all get 1/n."""
    r = compute_delta_softmax_ratios(
        ema_now_per_domain={"math": 0.50, "code": 0.50, "if": 0.50},
        ema_earlier_per_domain={"math": 0.50, "code": 0.50, "if": 0.50},
        temperature=1.0, ratio_floor=0.05,
    )
    for d, v in r.items():
        assert abs(v["ratio"] - 1 / 3) < 1e-9
        assert v["signal"] == 0.0
        assert v["signed_delta"] == 0.0


def test_delta_kl_rising_truncates_signal_to_zero():
    """KL rising in a domain (positive signed_delta) should give signal=0."""
    r = compute_delta_softmax_ratios(
        ema_now_per_domain={"math": 0.30, "code": 0.60, "if": 0.40},
        ema_earlier_per_domain={"math": 0.40, "code": 0.50, "if": 0.50},  # code went UP
        temperature=1.0, ratio_floor=0.05,
    )
    # code's signed_delta is positive (rising) → signal=0
    assert r["code"]["signed_delta"] > 0
    assert r["code"]["signal"] == 0.0
    # math is dropping fastest → highest signal
    assert r["math"]["signal"] > r["if"]["signal"]
    assert r["math"]["ratio"] > r["if"]["ratio"] > r["code"]["ratio"]


def test_delta_fast_dropper_gets_more_ratio():
    """Direction A core property: higher drop speed → higher ratio."""
    r = compute_delta_softmax_ratios(
        ema_now_per_domain={"math": 0.30, "code": 0.70, "if": 0.95},
        ema_earlier_per_domain={"math": 1.00, "code": 1.00, "if": 1.00},
        # math dropped 70%, code 30%, if only 5%
        temperature=1.0, ratio_floor=0.05,
    )
    # math should get the most (fastest dropper)
    ratios = {d: v["ratio"] for d, v in r.items()}
    assert ratios["math"] > ratios["code"] > ratios["if"]
    assert abs(sum(ratios.values()) - 1.0) < 1e-9


def test_delta_signal_cap():
    """signal should be clipped to signal_cap."""
    r = compute_delta_softmax_ratios(
        # math: EMA_earlier=1, EMA_now=0 → signed_delta=-1, signal=1 (capped from 1.0)
        ema_now_per_domain={"math": 0.0, "code": 0.5, "if": 0.6},
        ema_earlier_per_domain={"math": 1.0, "code": 1.0, "if": 1.0},
        temperature=1.0, ratio_floor=0.05,
        signal_cap=0.5,  # cap at 0.5
    )
    assert r["math"]["signal"] == 0.5  # capped
    assert r["code"]["signal"] == 0.5  # also at cap (signed_delta=-0.5, signal=0.5)
    # if's signal should also be 0.4 (uncapped, within cap)
    assert r["if"]["signal"] == pytest.approx(0.4, abs=1e-9)


def test_delta_lower_temp_sharper():
    args = dict(
        ema_now_per_domain={"math": 0.30, "code": 0.70, "if": 0.95},
        ema_earlier_per_domain={"math": 1.00, "code": 1.00, "if": 1.00},
        ratio_floor=0.05,
    )
    r_warm = compute_delta_softmax_ratios(temperature=2.0, **args)
    r_cool = compute_delta_softmax_ratios(temperature=0.5, **args)
    # Lower T → sharper toward winner (math)
    assert r_cool["math"]["ratio"] > r_warm["math"]["ratio"]
    assert r_cool["if"]["ratio"] < r_warm["if"]["ratio"]


def test_delta_infeasible_floor():
    with pytest.raises(ValueError):
        compute_delta_softmax_ratios(
            ema_now_per_domain={"math": 1, "code": 1, "if": 1},
            ema_earlier_per_domain={"math": 1, "code": 1, "if": 1},
            temperature=1.0, ratio_floor=0.5,  # 3*0.5 > 1
        )


def test_delta_invalid_signal_cap():
    with pytest.raises(ValueError):
        compute_delta_softmax_ratios(
            ema_now_per_domain={"math": 1, "code": 1, "if": 1},
            ema_earlier_per_domain={"math": 1, "code": 1, "if": 1},
            temperature=1.0, ratio_floor=0.05, signal_cap=0,
        )


def test_delta_zero_ema_earlier_no_div_zero():
    """When EMA_earlier=0 (extreme edge), eps guards division."""
    r = compute_delta_softmax_ratios(
        ema_now_per_domain={"math": 0.5, "code": 0.4, "if": 0.6},
        ema_earlier_per_domain={"math": 0.0, "code": 0.7, "if": 0.65},
        temperature=1.0, ratio_floor=0.05,
    )
    # Should not crash; math's signed_delta will be huge but signal clipped
    assert all(v["ratio"] >= 0.05 - 1e-9 for v in r.values())
    assert abs(sum(v["ratio"] for v in r.values()) - 1.0) < 1e-9


# ---------------- WriterState.step_dynamic_delta ----------------


def test_step_delta_waits_for_2W():
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": [], "code": [], "if": []},
    )
    # Only 5 points, W=5 → need 10. Not enough.
    domain_kl = {d: _kl_series([1.0, 0.9, 0.85, 0.8, 0.75]) for d in ("math", "code", "if")}
    diag = state.step_dynamic_delta(
        domain_kl, ema_window=5, update_every_steps=10,
        temperature=1.0, ratio_floor=0.05, signal_cap=1.0,
    )
    assert not diag["updated"]
    assert "insufficient_data_for_2W" in diag["reason"]


def test_step_delta_first_update_computes_ratios():
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": [], "code": [], "if": []},
    )
    # 10 KL points each (= 2W with W=5). Per-domain:
    #   prev 5 (positions 0-4)  → EMA_earlier
    #   last 5 (positions 5-9)  → EMA_now
    # Construct so that math drops 60%, code drops 20%, if drops 4%.
    domain_kl = {
        "math": _kl_series([0.5] * 5 + [0.2] * 5),
        "code": _kl_series([0.5] * 5 + [0.4] * 5),
        "if":   _kl_series([0.5] * 5 + [0.48] * 5),
    }
    diag = state.step_dynamic_delta(
        domain_kl, ema_window=5, update_every_steps=10,
        temperature=1.0, ratio_floor=0.05, signal_cap=1.0,
    )
    assert diag["updated"], diag["reason"]
    # Signals: math 0.6, code 0.2, if 0.04 — fast dropper gets more
    sig = diag["signal"]
    assert sig["math"] > sig["code"] > sig["if"]
    r = diag["ratio"]
    assert r["math"] > r["code"] > r["if"]
    assert abs(sum(r.values()) - 1.0) < 1e-9
    for v in r.values():
        assert v >= 0.05 - 1e-9


def test_step_delta_respects_update_cadence():
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": [], "code": [], "if": []},
    )
    domain_kl = {
        d: _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.5] * 15)
        for d in ("math", "code", "if")
    }
    diag1 = state.step_dynamic_delta(
        domain_kl, ema_window=5, update_every_steps=10,
        temperature=1.0, ratio_floor=0.05, signal_cap=1.0,
    )
    assert diag1["updated"]
    assert state.last_update_step == 19

    # Same data → next update is at step 29, but latest_step still 19 → wait.
    diag2 = state.step_dynamic_delta(
        domain_kl, ema_window=5, update_every_steps=10,
        temperature=1.0, ratio_floor=0.05, signal_cap=1.0,
    )
    assert not diag2["updated"]
    assert "waiting" in diag2["reason"]


def test_step_delta_roundtrip(tmp_path):
    """Compute ratios, write status JSON, load back, ratios preserved."""
    state = WriterState(
        patience_K=0, rehearsal_rate=0.0,
        domain_map={"math": ["m_ds"], "code": ["c_ds"], "if": ["i_ds"]},
    )
    domain_kl = {
        "math": _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.3] * 15),
        "code": _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.6] * 15),
        "if":   _kl_series([1.0, 0.95, 0.9, 0.85, 0.8] + [0.85] * 15),
    }
    state.step_dynamic_delta(
        domain_kl, ema_window=5, update_every_steps=10,
        temperature=1.0, ratio_floor=0.05, signal_cap=1.0,
    )
    status_path = tmp_path / "status.json"
    write_status_atomic(status_path, state.to_status_dict_dynamic_delta(exp_name="exp_x"))

    state2 = WriterState.from_status_file_dynamic_delta(
        status_path, domain_map={"math": ["m_ds"], "code": ["c_ds"], "if": ["i_ds"]},
    )
    for d in ("math", "code", "if"):
        assert state2.mixture_ratio[d] == state.mixture_ratio[d]
        assert state2.signed_delta_per_domain[d] == state.signed_delta_per_domain[d]
        assert state2.signal_per_domain[d] == state.signal_per_domain[d]
    assert state2.last_update_step == state.last_update_step


def test_step_delta_ignores_dynamic_gap_mode_json(tmp_path):
    """Old dynamic-gap status JSON should NOT be loaded into dynamic-delta state."""
    payload = {
        "mode": "dynamic",  # the gap mode
        "domains": {
            "math": {"mixture_ratio": 0.5, "initial_kl": 0.1, "normalized_kl": 0.3},
        },
    }
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(payload))
    state = WriterState.from_status_file_dynamic_delta(
        status_path, domain_map={"math": ["m"]},
    )
    # Should start fresh
    assert state.mixture_ratio == {}
    assert state.last_update_step is None

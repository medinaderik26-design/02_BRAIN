"""Tests for GX-012-A.1 dynamic controller response."""

from gx012_dynamic_response import run_dynamic_sequence


def test_dynamic_trace_is_deterministic_and_bounded():
    a = run_dynamic_sequence().rows
    b = run_dynamic_sequence().rows
    assert a == b
    assert len(a) == 25
    for row in a:
        assert 0.0 <= row["input_aperture"] <= 1.0
        assert 0.0 <= row["output_aperture"] <= 1.0
        assert 0.0 <= row["input_target"] <= 1.0
        assert 0.0 <= row["output_target"] <= 1.0


def test_pressure_spike_reduces_input_aperture():
    rows = run_dynamic_sequence().rows
    baseline = rows[4]["input_aperture"]
    pressure_peak = rows[9]["input_aperture"]
    assert pressure_peak < baseline


def test_uncertainty_spike_increases_output_target():
    rows = run_dynamic_sequence().rows
    before = rows[14]["output_target"]
    spike = rows[19]["output_target"]
    assert spike > before


def test_controller_recovers_after_pressure():
    rows = run_dynamic_sequence().rows
    pressure_peak = rows[9]["input_aperture"]
    recovered = rows[14]["input_aperture"]
    assert recovered > pressure_peak


if __name__ == "__main__":
    test_dynamic_trace_is_deterministic_and_bounded()
    test_pressure_spike_reduces_input_aperture()
    test_uncertainty_spike_increases_output_target()
    test_controller_recovers_after_pressure()
    print("GX-012-A.1 dynamic response tests passed")

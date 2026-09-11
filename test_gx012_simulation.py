"""Deterministic tests for the GX-012-A simulation harness."""

from gx012_simulation import default_profiles, run_profile, run_suite


def test_suite_shape_and_bounds():
    rows = run_suite(steps=20, mu=.25)
    assert len(rows) == len(default_profiles())
    for row in rows:
        assert 0.0 <= row["final_input_aperture"] <= 1.0
        assert 0.0 <= row["final_output_aperture"] <= 1.0
        assert 0.0 <= row["final_raw_fraction"] <= 1.0
        assert 0.0 <= row["final_lineage_fraction"] <= 1.0
        assert row["final_output_budget"] >= 1


def test_resource_pressure_reduces_apertures():
    rows = {row["task"]: row for row in run_suite()}
    constrained = rows["resource_constrained"]
    relevant = rows["high_memory_relevance"]
    assert constrained["final_input_aperture"] < relevant["final_input_aperture"]
    assert constrained["final_output_aperture"] < relevant["final_output_aperture"]


def test_repeated_runs_are_deterministic():
    assert run_suite() == run_suite()


def test_controller_converges_toward_targets():
    profile = default_profiles()[1]
    short = run_profile(profile, steps=1)
    long = run_profile(profile, steps=100)
    assert abs(long.final_input_aperture - .5) > abs(short.final_input_aperture - .5)
    assert long.final_input_aperture <= 1.0
    assert long.final_output_aperture <= 1.0


if __name__ == "__main__":
    test_suite_shape_and_bounds()
    test_resource_pressure_reduces_apertures()
    test_repeated_runs_are_deterministic()
    test_controller_converges_toward_targets()
    print("GX-012-A simulation tests passed")

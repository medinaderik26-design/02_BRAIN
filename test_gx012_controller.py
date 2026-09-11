"""Deterministic tests for GX-012-A Dual Cone Variator."""

from gx012_controller import DualConeVariator, GX012Signals


def test_apertures_stay_bounded_and_independent():
    controller = DualConeVariator(mu=0.25, output_min=16, output_max=128)
    allocation = controller.update(
        GX012Signals(
            relevance=1.0,
            cohesion=1.0,
            lineage_depth=1.0,
            uncertainty=0.8,
            task_complexity=0.2,
            resource_pressure=0.0,
        ),
        input_budget=1000,
    )
    assert 0.0 <= allocation.input_aperture <= 1.0
    assert 0.0 <= allocation.output_aperture <= 1.0
    assert abs(allocation.raw_fraction + allocation.lineage_fraction - 1.0) < 1e-12
    assert 16 <= allocation.output_budget <= 128


def test_resource_pressure_reduces_target_allocation():
    controller = DualConeVariator(mu=1.0)
    low_pressure = controller.target(GX012Signals(resource_pressure=0.0))
    high_pressure = controller.target(GX012Signals(resource_pressure=1.0))
    assert high_pressure[0] < low_pressure[0]
    assert high_pressure[1] < low_pressure[1]


def test_controller_converges_to_constant_target():
    controller = DualConeVariator(mu=0.5)
    signals = GX012Signals(
        relevance=0.9,
        cohesion=0.8,
        lineage_depth=0.7,
        uncertainty=0.2,
        task_complexity=0.6,
        contradiction=0.1,
        resource_pressure=0.1,
    )
    target_input, target_output = controller.target(signals)
    for _ in range(40):
        controller.update(signals, input_budget=2048)
    assert abs(controller.input_aperture - target_input) < 1e-6
    assert abs(controller.output_aperture - target_output) < 1e-6


def test_history_is_reproducible_and_serializable():
    controller = DualConeVariator(mu=0.25)
    signals = GX012Signals(relevance=0.8, uncertainty=0.7)
    controller.update(signals, input_budget=512)
    history = controller.export_history()
    assert len(history) == 1
    assert history[0]["step"] == 1
    assert set(history[0]["signals"]) == {
        "relevance",
        "cohesion",
        "lineage_depth",
        "uncertainty",
        "task_complexity",
        "contradiction",
        "resource_pressure",
    }


if __name__ == "__main__":
    test_apertures_stay_bounded_and_independent()
    test_resource_pressure_reduces_target_allocation()
    test_controller_converges_to_constant_target()
    test_history_is_reproducible_and_serializable()
    print("GX-012 controller tests: PASS")

"""Tests for reproducible experiment-run auditing."""
from glyphin_experiment import GlyphinExperiment
from glyphin_experiment_referee import referee_run


def operations():
    return [
        {"operation": "create", "name": "root", "cohesion": 0.4},
        {"operation": "create", "name": "child", "level": 1},
        {"operation": "create", "name": "leaf", "level": 2},
        {"operation": "link", "parent": "root", "child": "child"},
        {"operation": "link", "parent": "child", "child": "leaf"},
        {"operation": "reinforce", "name": "child", "kappa": 0.5},
        {"operation": "decay", "delta": 1.0},
    ]


def test_recorded_run_replays_exactly() -> None:
    run = GlyphinExperiment().run(operations())
    result = referee_run(run)
    assert result.exact
    assert result.replay_succeeded
    assert result.memory is not None
    assert result.memory.state_mismatches == ()


def test_recorded_final_state_tampering_is_detected() -> None:
    run = GlyphinExperiment().run(operations())
    tampered = run.to_dict()
    tampered["final_memory"]["states"]["child"]["frequency"] += 1
    from glyphin_experiment import ExperimentRun

    changed = ExperimentRun(
        version=tampered["version"],
        operations=tuple(tampered["operations"]),
        results=tuple(tampered["results"]),
        final_memory=tampered["final_memory"],
    )
    result = referee_run(changed)
    assert not result.exact
    assert result.memory is not None
    assert any(
        item["state"] == "child" and item["field"] == "frequency"
        for item in result.memory.state_mismatches
    )


def test_invalid_trace_is_reported_not_hidden() -> None:
    run = GlyphinExperiment().run([
        {"operation": "create", "name": "root"},
    ])
    tampered = run.to_dict()
    tampered["operations"].append({"operation": "link", "parent": "root", "child": "missing"})
    from glyphin_experiment import ExperimentRun

    changed = ExperimentRun(
        version=tampered["version"],
        operations=tuple(tampered["operations"]),
        results=tuple(tampered["results"]),
        final_memory=tampered["final_memory"],
    )
    result = referee_run(changed)
    assert not result.exact
    assert not result.replay_succeeded
    assert result.replay_error is not None


if __name__ == "__main__":
    tests = [
        test_recorded_run_replays_exactly,
        test_recorded_final_state_tampering_is_detected,
        test_invalid_trace_is_reported_not_hidden,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} experiment referee tests")

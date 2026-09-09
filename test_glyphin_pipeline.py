"""End-to-end tests for the deterministic Glyphin research pipeline."""
from glyphin_pipeline import analyze
from glyphin_research_core import GlyphinMemory


def _memory() -> GlyphinMemory:
    memory = GlyphinMemory()
    memory.add_state("root", created_at="2026-01-01T00:00:00+00:00")
    memory.add_state("child", level=1, created_at="2026-01-01T00:00:01+00:00")
    memory.add_state("leaf", level=2, created_at="2026-01-01T00:00:02+00:00")
    memory.link_state("root", "child")
    memory.link_state("child", "leaf")
    memory.reinforce("child", kappa=0.5)
    return memory


def test_pipeline_preserves_state_and_topology() -> None:
    report = analyze(_memory())
    assert report.persistence_referee.exact
    assert report.adaptation.lossless
    assert report.topology_referee.parse_ok
    assert report.topology_referee.exact_match
    assert report.candidate_search.exact_candidates


def test_pipeline_does_not_claim_token_measurement() -> None:
    report = analyze(_memory())
    assert report.compression.source_tokens is None
    assert report.compression.encoded_tokens is None
    assert report.compression.token_reduction_pct is None


def test_pipeline_reports_loss_at_memory_to_topology_boundary() -> None:
    memory = _memory()
    report = analyze(memory)
    assert "cohesion" in report.adaptation.lost_state_fields
    assert "resonance" in report.adaptation.lost_state_fields
    assert "frequency" in report.adaptation.lost_state_fields


if __name__ == "__main__":
    tests = [
        test_pipeline_preserves_state_and_topology,
        test_pipeline_does_not_claim_token_measurement,
        test_pipeline_reports_loss_at_memory_to_topology_boundary,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} pipeline tests")

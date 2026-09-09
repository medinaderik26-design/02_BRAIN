"""Tests for the user-to-model context gateway."""
from context_gateway import ContextGateway, ContextItem


def test_budget_and_determinism() -> None:
    gateway = ContextGateway(max_chars=60)
    gateway.add_many([
        ContextItem("core", "Glyphin topology and reconstruction", priority=5),
        ContextItem("noise", "unrelated historical material", priority=0),
        ContextItem("referee", "exact missing and extra edge comparison", priority=4),
    ])
    first = gateway.build("topology referee")
    second = gateway.build("topology referee")
    assert first.context == second.context
    assert first.packet_chars <= 60
    assert set(first.selected_keys).isdisjoint(first.excluded_keys)


def test_metrics_explicitly_leave_tokens_unmeasured() -> None:
    gateway = ContextGateway(max_chars=1000)
    gateway.add(ContextItem("a", "alpha beta gamma", priority=1))
    packet = gateway.build("alpha")
    metrics = gateway.metrics(packet)
    assert metrics.source_chars >= metrics.encoded_chars
    assert metrics.token_reduction_pct is None


def test_selection_is_budget_bounded() -> None:
    gateway = ContextGateway(max_chars=10)
    gateway.add(ContextItem("long", "12345678901234567890", priority=10))
    packet = gateway.build("long")
    assert packet.packet_chars <= 10
    assert packet.selected_keys == ()
    assert packet.excluded_keys == ("long",)

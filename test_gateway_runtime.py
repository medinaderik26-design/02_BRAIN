"""Integration tests for the user-to-model gateway runtime."""
from context_cache import ContextCache
from context_gateway import ContextGateway, ContextItem
from provider_adapter import DryRunAdapter
from gateway_runtime import GatewayRuntime


def make_runtime() -> GatewayRuntime:
    gateway = ContextGateway(max_chars=500)
    gateway.add_many([
        ContextItem("core", "Glyphin topology and reconstruction", priority=5),
        ContextItem("referee", "exact missing and extra edge comparison", priority=4),
        ContextItem("noise", "unrelated historical material", priority=0),
    ])
    return GatewayRuntime(gateway, ContextCache(), DryRunAdapter(), policy={"mode": "test"})


def test_identical_requests_hit_cache_despite_fresh_request_metadata() -> None:
    runtime = make_runtime()
    first = runtime.run("topology referee")
    second = runtime.run("topology referee")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert first.response.text == second.response.text
    assert runtime.cache.stats().requests == 2
    assert runtime.cache.stats().hits == 1
    assert runtime.cache.stats().misses == 1


def test_complete_request_fingerprint_remains_request_specific() -> None:
    runtime = make_runtime()
    first = runtime.prepare("topology")
    second = runtime.prepare("topology")

    assert first.fingerprint() != second.fingerprint()
    assert first.cache_key() == second.cache_key()


def test_changed_query_does_not_reuse_cached_response() -> None:
    runtime = make_runtime()
    first = runtime.run("topology")
    second = runtime.run("reconstruction")

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert first.cache_key != second.cache_key
    assert runtime.cache.stats().hits == 0
    assert runtime.cache.stats().misses == 2


def test_prepare_preserves_selected_and_excluded_context() -> None:
    runtime = make_runtime()
    request = runtime.prepare("topology referee")

    assert request.envelope.query == "topology referee"
    assert request.envelope.selected_keys
    assert set(request.envelope.selected_keys).isdisjoint(request.envelope.excluded_keys)
    assert request.envelope.context_sha256
    assert request.envelope.source_sha256

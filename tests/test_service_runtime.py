from __future__ import annotations

import pytest

from cap_harness.providers import service_runtime


def test_request_limit_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAP_HARNESS_SERVICE_CONCURRENCY", raising=False)
    assert service_runtime.request_limit() == 1
    assert service_runtime.request_limit(default=3) == 3
    monkeypatch.setenv("CAP_HARNESS_SERVICE_CONCURRENCY", "4")
    assert service_runtime.request_limit() == 4
    monkeypatch.setenv("CAP_HARNESS_SERVICE_CONCURRENCY", "0")
    with pytest.raises(ValueError):
        service_runtime.request_limit()
    monkeypatch.setenv("CAP_HARNESS_SERVICE_CONCURRENCY", "not-an-int")
    with pytest.raises(ValueError):
        service_runtime.request_limit()


def test_install_service_runtime_bounds_and_adds_health() -> None:
    fastapi = pytest.importorskip("fastapi")
    app = fastapi.FastAPI()
    service_runtime.install_service_runtime(app, limit=2)
    assert app.state.request_limit == 2
    assert any(getattr(route, "path", None) == "/healthz" for route in app.router.routes)


def test_every_provider_service_installs_the_runtime_guard() -> None:
    from pathlib import Path

    providers = Path(service_runtime.__file__).resolve().parent
    for service in providers.glob("*/service.py"):
        source = service.read_text(encoding="utf-8")
        assert "install_service_runtime(app)" in source, service

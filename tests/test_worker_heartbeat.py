import asyncio

import httpx

from orchestro_mesh import worker_heartbeat
from orchestro_mesh.models import Heartbeat, NodeStatus


def _patch_async_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original = worker_heartbeat.httpx.AsyncClient

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return original(transport=transport, **kwargs)

    monkeypatch.setattr(worker_heartbeat.httpx, "AsyncClient", factory)


def test_send_heartbeat_posts_bearer_token(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(200, json={"ok": True})

    _patch_async_client(monkeypatch, handler)
    asyncio.run(
        worker_heartbeat.send_heartbeat(
            "http://gateway:8765",
            token="secret",
            beat=Heartbeat(node_id="w1", status=NodeStatus.ONLINE, current_jobs=2, queue_depth=0),
        )
    )
    assert seen["url"].endswith("/mesh/heartbeat")
    assert seen["auth"] == "Bearer secret"
    assert b'"node_id":"w1"' in seen["body"]
    assert b'"current_jobs":2' in seen["body"]


def test_send_heartbeat_omits_auth_when_no_token(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    _patch_async_client(monkeypatch, handler)
    asyncio.run(
        worker_heartbeat.send_heartbeat("http://gateway:8765", token=None, beat=Heartbeat(node_id="w1"))
    )
    assert seen["auth"] is None


def test_heartbeat_loop_starts_and_stops_cleanly():
    async def run():
        loop = worker_heartbeat.HeartbeatLoop(
            gateway_url="http://nowhere",
            token=None,
            inventory_provider=lambda: None,  # tick is a no-op
            interval_s=0.05,
        )
        loop.start()
        await asyncio.sleep(0.1)
        await loop.stop()

    asyncio.run(run())

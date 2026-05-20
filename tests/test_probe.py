import asyncio

from orchestro_mesh import probe
from orchestro_mesh.models import (
    BackendEndpoint,
    BackendKind,
    ModelCapability,
    ModelState,
    NodeInventory,
)


def _node() -> NodeInventory:
    return NodeInventory(
        node_id="n1",
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id="b1", kind=BackendKind.OPENAI_COMPAT, base_url="http://x/v1")],
        models=[
            ModelCapability(id="present", backend_id="b1", state=ModelState.COLD),
            ModelCapability(id="missing", backend_id="b1", state=ModelState.WARM),
        ],
    )


def test_refresh_promotes_present_and_demotes_missing(monkeypatch):
    async def fake_probe(backend, timeout_s=10.0):
        return {"present"}

    monkeypatch.setattr(probe, "probe_backend_models", fake_probe)
    node = asyncio.run(probe.refresh_node_model_states(_node()))
    states = {m.id: m.state for m in node.models}
    assert states["present"] == ModelState.WARM
    assert states["missing"] == ModelState.ABSENT


def test_refresh_leaves_state_when_backend_unreachable(monkeypatch):
    async def boom(backend, timeout_s=10.0):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(probe, "probe_backend_models", boom)
    node = asyncio.run(probe.refresh_node_model_states(_node()))
    states = {m.id: m.state for m in node.models}
    assert states["present"] == ModelState.COLD
    assert states["missing"] == ModelState.WARM

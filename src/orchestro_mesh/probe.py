from __future__ import annotations

import logging

import httpx

from orchestro_mesh.models import BackendEndpoint, BackendKind, ModelState, NodeInventory

logger = logging.getLogger(__name__)


async def probe_backend_models(backend: BackendEndpoint, timeout_s: float = 10.0) -> set[str]:
    """Return the set of model IDs the backend currently exposes via /models."""
    if backend.kind == BackendKind.MOCK:
        return set()
    url = str(backend.base_url).rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=timeout_s, headers=backend.headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
    models = data.get("data") or []
    return {entry.get("id") for entry in models if isinstance(entry, dict) and entry.get("id")}


async def refresh_node_model_states(node: NodeInventory) -> NodeInventory:
    """Update model.state on a node based on what each backend currently exposes.

    - Model present in backend listing: mark WARM unless already ACTIVE/LOADING.
    - Model absent: mark ABSENT.
    - Backend unreachable: leave models as-is and log a warning.
    """
    for backend in node.backends:
        try:
            present = await probe_backend_models(backend)
        except Exception as exc:  # noqa: BLE001
            logger.warning("probe.backend_unreachable node=%s backend=%s error=%s", node.node_id, backend.id, exc)
            continue
        for model in node.models:
            if model.backend_id != backend.id:
                continue
            if model.id in present:
                if model.state in {ModelState.COLD, ModelState.ABSENT, ModelState.LOADING}:
                    model.state = ModelState.WARM
            else:
                model.state = ModelState.ABSENT
    return node

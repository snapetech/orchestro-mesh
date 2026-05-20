from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from orchestro_mesh.models import NodeInventory

RedactionMode = Literal["off", "log", "block"]


class MeshConfig(BaseModel):
    local_node_id: str | None = None
    store_path: str = ".orchestro-mesh/mesh.db"
    nodes: list[NodeInventory] = Field(default_factory=list)
    mesh_token: str | None = None
    api_tokens: dict[str, str] = Field(default_factory=dict)
    worker_token: str | None = None
    redaction_mode: RedactionMode = "log"
    node_ttl_seconds: int | None = 300
    quota_credits: dict[str, float] = Field(default_factory=dict)
    credits_per_1k_tokens: float = 1.0
    rate_limit_per_minute: dict[str, int] = Field(default_factory=dict)
    rate_limit_default_per_minute: int | None = None
    probe_interval_s: float | None = None


def load_config(path: str | Path) -> MeshConfig:
    config_path = Path(path)
    if not config_path.exists():
        return MeshConfig()
    data = yaml.safe_load(config_path.read_text()) or {}
    return MeshConfig.model_validate(data)


def apply_env_overrides(config: MeshConfig) -> MeshConfig:
    """Overlay environment variables onto a config. Env always wins when set."""
    if (token := os.environ.get("ORCHESTRO_MESH_TOKEN")):
        config.mesh_token = token
    if (worker_token := os.environ.get("ORCHESTRO_MESH_WORKER_TOKEN")):
        config.worker_token = worker_token
    if (store_path := os.environ.get("ORCHESTRO_MESH_STORE_PATH")):
        config.store_path = store_path
    if (mode := os.environ.get("ORCHESTRO_MESH_REDACTION_MODE")):
        if mode in {"off", "log", "block"}:
            config.redaction_mode = mode  # type: ignore[assignment]
    return config

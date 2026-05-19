from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from orchestro_mesh.models import NodeInventory


class MeshConfig(BaseModel):
    local_node_id: str | None = None
    store_path: str = ".orchestro-mesh/mesh.db"
    nodes: list[NodeInventory] = Field(default_factory=list)


def load_config(path: str | Path) -> MeshConfig:
    config_path = Path(path)
    if not config_path.exists():
        return MeshConfig()
    data = yaml.safe_load(config_path.read_text()) or {}
    return MeshConfig.model_validate(data)

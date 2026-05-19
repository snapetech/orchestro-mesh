from __future__ import annotations

from collections.abc import Iterable

from orchestro_mesh.models import ModelCapability, NodeInventory, TaskClass


class CapabilityMarket:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeInventory] = {}

    def upsert_node(self, node: NodeInventory) -> None:
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)

    def list_nodes(self) -> list[NodeInventory]:
        return list(self._nodes.values())

    def find_models(self, task_class: TaskClass | None = None, tags: Iterable[str] = ()) -> list[tuple[NodeInventory, ModelCapability]]:
        tag_set = set(tags)
        found: list[tuple[NodeInventory, ModelCapability]] = []
        for node in self._nodes.values():
            for model in node.models:
                if task_class and task_class not in model.task_classes:
                    continue
                if tag_set and not tag_set.issubset(set(model.tags)):
                    continue
                found.append((node, model))
        return found

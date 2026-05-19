from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from orchestro_mesh.benchmark import run_chat_benchmark
from orchestro_mesh.config import load_config
from orchestro_mesh.models import ChatMessage, InferenceRequest, TaskClass
from orchestro_mesh.openai_client import OpenAICompatClient
from orchestro_mesh.scheduler import Scheduler
from orchestro_mesh.store import MeshStore

app = typer.Typer(help="Private trusted inference mesh control plane.")


@app.command()
def init(path: Path = typer.Option(Path("mesh.yaml"), "--path")) -> None:
    if path.exists():
        raise typer.BadParameter(f"already exists: {path}")
    path.write_text("local_node_id: local\nstore_path: .orchestro-mesh/mesh.db\nnodes: []\n")
    print(f"[green]wrote[/green] {path}")


@app.command("nodes")
def nodes(config: Path = typer.Option(Path("mesh.yaml"), "--config")) -> None:
    cfg = load_config(config)
    store = MeshStore(cfg.store_path)
    for node in cfg.nodes:
        store.upsert_node(node)
    table = Table(title="Orchestro Mesh Nodes")
    table.add_column("node")
    table.add_column("owner")
    table.add_column("trust")
    table.add_column("status")
    table.add_column("models")
    for node in store.list_nodes():
        table.add_row(node.node_id, node.owner, node.trust_domain, node.status.value, str(len(node.models)))
    print(table)


@app.command("route")
def route(
    prompt: str,
    config: Path = typer.Option(Path("mesh.yaml"), "--config"),
    requester: str = typer.Option("local", "--requester"),
    task_class: TaskClass = typer.Option(TaskClass.CHAT, "--task"),
) -> None:
    cfg = load_config(config)
    scheduler = Scheduler(local_node_id=cfg.local_node_id)
    request = InferenceRequest(
        requester=requester,
        task_class=task_class,
        messages=[ChatMessage(role="user", content=prompt)],
    )
    result = scheduler.route(request, cfg.nodes)
    print_json(result.model_dump(mode="json"))


@app.command("benchmark")
def benchmark(
    node_id: str,
    model_id: str,
    config: Path = typer.Option(Path("mesh.yaml"), "--config"),
    task_class: TaskClass = typer.Option(TaskClass.CHAT, "--task"),
) -> None:
    cfg = load_config(config)
    node = next((item for item in cfg.nodes if item.node_id == node_id), None)
    if node is None:
        raise typer.BadParameter(f"node not found: {node_id}")
    model = node.model_by_id(model_id)
    if model is None:
        raise typer.BadParameter(f"model not found: {model_id}")
    backend = node.backend_by_id(model.backend_id)
    if backend is None:
        raise typer.BadParameter(f"backend not found: {model.backend_id}")
    sample = asyncio.run(run_chat_benchmark(OpenAICompatClient(backend), model_id=model.id, task_class=task_class))
    print_json(sample.model_dump(mode="json"))


def print_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    app()

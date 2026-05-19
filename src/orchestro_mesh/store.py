from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from orchestro_mesh.models import JobRecord, NodeInventory, UsageLedgerEntry

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    trust_domain TEXT NOT NULL,
    status TEXT NOT NULL,
    inventory_json TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    requester TEXT NOT NULL,
    node_id TEXT,
    model_id TEXT,
    backend_id TEXT,
    sensitivity TEXT NOT NULL,
    task_class TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_ms REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS usage_ledger (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    requester TEXT NOT NULL,
    node_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    gpu_ms_estimate REAL NOT NULL DEFAULT 0,
    credit_cost REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


class MeshStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def upsert_node(self, node: NodeInventory) -> None:
        payload = node.model_dump_json()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO nodes (node_id, owner, trust_domain, status, inventory_json, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    owner=excluded.owner,
                    trust_domain=excluded.trust_domain,
                    status=excluded.status,
                    inventory_json=excluded.inventory_json,
                    last_seen=excluded.last_seen
                """,
                (node.node_id, node.owner, node.trust_domain, node.status.value, payload, node.last_seen.isoformat()),
            )

    def list_nodes(self) -> list[NodeInventory]:
        with self.connect() as conn:
            rows = conn.execute("SELECT inventory_json FROM nodes ORDER BY node_id").fetchall()
        return [NodeInventory.model_validate_json(row["inventory_json"]) for row in rows]

    def create_job(self, job: JobRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, requester, node_id, model_id, backend_id, sensitivity, task_class, status,
                    created_at, started_at, finished_at, input_tokens, output_tokens, total_ms, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.requester,
                    job.node_id,
                    job.model_id,
                    job.backend_id,
                    job.sensitivity.value,
                    job.task_class.value,
                    job.status,
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.finished_at.isoformat() if job.finished_at else None,
                    job.input_tokens,
                    job.output_tokens,
                    job.total_ms,
                    job.error,
                ),
            )

    def add_ledger_entries(self, entries: Iterable[UsageLedgerEntry]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO usage_ledger (
                    id, job_id, requester, node_id, model_id, input_tokens, output_tokens,
                    gpu_ms_estimate, credit_cost, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        entry.id,
                        entry.job_id,
                        entry.requester,
                        entry.node_id,
                        entry.model_id,
                        entry.input_tokens,
                        entry.output_tokens,
                        entry.gpu_ms_estimate,
                        entry.credit_cost,
                        entry.created_at.isoformat(),
                    )
                    for entry in entries
                ],
            )

    def ledger_totals(self) -> dict[str, float]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT requester, SUM(credit_cost) AS credits FROM usage_ledger GROUP BY requester ORDER BY requester"
            ).fetchall()
        return {row["requester"]: float(row["credits"] or 0.0) for row in rows}

    def dump_json(self) -> str:
        return json.dumps({"nodes": [node.model_dump(mode="json") for node in self.list_nodes()]}, indent=2)

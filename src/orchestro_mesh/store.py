from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from orchestro_mesh.models import (
    FeedbackRating,
    JobRecord,
    NodeInventory,
    UsageLedgerEntry,
    now_utc,
)

logger = logging.getLogger(__name__)

SCHEMA = """
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
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    node_id TEXT,
    model_id TEXT,
    requester TEXT,
    verifier TEXT NOT NULL,
    rating REAL NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_node_model_created
    ON feedback (node_id, model_id, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created
    ON jobs (status, created_at);
"""

_BUSY_RETRY_MAX = 5
_BUSY_RETRY_BASE_S = 0.02


def _is_busy_error(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.OperationalError):
        msg = str(exc).lower()
        return "locked" in msg or "busy" in msg
    return False


class MeshStore:
    """SQLite-backed store with WAL, busy_timeout, and write retries.

    Single-process safe; a single gateway can scale to plenty of concurrent
    requests on one machine. For multi-replica deployments, swap this out for
    Postgres — the interface is narrow enough to make that practical.
    """

    def __init__(self, path: str | Path, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = busy_timeout_ms
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self._busy_timeout_ms / 1000.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def _execute_write(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> None:
        last_exc: BaseException | None = None
        for attempt in range(_BUSY_RETRY_MAX):
            try:
                with self.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        conn.execute(sql, params)
                        conn.execute("COMMIT")
                    except BaseException:
                        conn.execute("ROLLBACK")
                        raise
                return
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if not _is_busy_error(exc):
                    raise
                backoff = _BUSY_RETRY_BASE_S * (2 ** attempt) + random.uniform(0, _BUSY_RETRY_BASE_S)
                logger.warning("store.busy attempt=%d sleep=%.3f", attempt + 1, backoff)
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def _executemany_write(self, sql: str, params_iter: list[tuple[Any, ...]]) -> None:
        if not params_iter:
            return
        last_exc: BaseException | None = None
        for attempt in range(_BUSY_RETRY_MAX):
            try:
                with self.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        conn.executemany(sql, params_iter)
                        conn.execute("COMMIT")
                    except BaseException:
                        conn.execute("ROLLBACK")
                        raise
                return
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if not _is_busy_error(exc):
                    raise
                backoff = _BUSY_RETRY_BASE_S * (2 ** attempt) + random.uniform(0, _BUSY_RETRY_BASE_S)
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def upsert_node(self, node: NodeInventory) -> None:
        payload = node.model_dump_json()
        self._execute_write(
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

    def get_node(self, node_id: str) -> NodeInventory | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT inventory_json FROM nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
        if row is None:
            return None
        return NodeInventory.model_validate_json(row["inventory_json"])

    def delta_node_jobs(self, node_id: str, delta: int) -> None:
        """Atomically adjust a node's current_jobs counter under a write lock."""
        last_exc: BaseException | None = None
        for attempt in range(_BUSY_RETRY_MAX):
            try:
                with self.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        row = conn.execute(
                            "SELECT inventory_json FROM nodes WHERE node_id = ?",
                            (node_id,),
                        ).fetchone()
                        if row is None:
                            conn.execute("COMMIT")
                            return
                        node = NodeInventory.model_validate_json(row["inventory_json"])
                        node.current_jobs = max(0, node.current_jobs + delta)
                        conn.execute(
                            "UPDATE nodes SET inventory_json = ?, status = ?, last_seen = ? WHERE node_id = ?",
                            (
                                node.model_dump_json(),
                                node.status.value,
                                node.last_seen.isoformat(),
                                node_id,
                            ),
                        )
                        conn.execute("COMMIT")
                    except BaseException:
                        conn.execute("ROLLBACK")
                        raise
                return
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if not _is_busy_error(exc):
                    raise
                backoff = _BUSY_RETRY_BASE_S * (2 ** attempt) + random.uniform(0, _BUSY_RETRY_BASE_S)
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def create_job(self, job: JobRecord) -> None:
        self._execute_write(
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

    def update_job(self, job_id: str, **fields: object) -> None:
        if not fields:
            return
        allowed = {
            "status",
            "node_id",
            "model_id",
            "backend_id",
            "started_at",
            "finished_at",
            "input_tokens",
            "output_tokens",
            "total_ms",
            "error",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values: list[Any] = []
        for value in fields.values():
            if hasattr(value, "isoformat"):
                values.append(value.isoformat())  # type: ignore[attr-defined]
            else:
                values.append(value)
        values.append(job_id)
        self._execute_write(f"UPDATE jobs SET {assignments} WHERE id = ?", tuple(values))

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_ms: float | None = None,
        error: str | None = None,
        finished_at: object | None = None,
    ) -> None:
        self.update_job(
            job_id,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_ms=total_ms,
            error=error,
            finished_at=finished_at or now_utc(),
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobRecord.model_validate(dict(row))

    def add_ledger_entries(self, entries: Iterable[UsageLedgerEntry]) -> None:
        rows = [
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
        ]
        self._executemany_write(
            """
            INSERT INTO usage_ledger (
                id, job_id, requester, node_id, model_id, input_tokens, output_tokens,
                gpu_ms_estimate, credit_cost, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def ledger_totals(self) -> dict[str, float]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT requester, SUM(credit_cost) AS credits FROM usage_ledger GROUP BY requester ORDER BY requester"
            ).fetchall()
        return {row["requester"]: float(row["credits"] or 0.0) for row in rows}

    def add_feedback(self, rating: FeedbackRating) -> None:
        self._execute_write(
            """
            INSERT INTO feedback (id, job_id, node_id, model_id, requester, verifier, rating, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rating.id,
                rating.job_id,
                rating.node_id,
                rating.model_id,
                rating.requester,
                rating.verifier,
                rating.rating,
                rating.notes,
                rating.created_at.isoformat(),
            ),
        )

    def list_feedback(self, job_id: str) -> list[FeedbackRating]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE job_id = ? ORDER BY created_at",
                (job_id,),
            ).fetchall()
        return [FeedbackRating.model_validate(dict(row)) for row in rows]

    def recent_jobs(self, limit: int = 20) -> list[JobRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [JobRecord.model_validate(dict(row)) for row in rows]

    def recent_feedback(self, limit: int = 20) -> list[FeedbackRating]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [FeedbackRating.model_validate(dict(row)) for row in rows]

    def recent_rating_averages(self, window_n: int = 50) -> dict[tuple[str, str], float]:
        """Return mean rating per (node_id, model_id) over the last N entries per pair."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT node_id, model_id, rating, created_at
                FROM feedback
                WHERE node_id IS NOT NULL AND model_id IS NOT NULL
                ORDER BY created_at DESC
                """
            ).fetchall()
        buckets: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            key = (row["node_id"], row["model_id"])
            bucket = buckets.setdefault(key, [])
            if len(bucket) < window_n:
                bucket.append(float(row["rating"]))
        return {key: sum(values) / len(values) for key, values in buckets.items() if values}

    def dump_json(self) -> str:
        return json.dumps({"nodes": [node.model_dump(mode="json") for node in self.list_nodes()]}, indent=2)

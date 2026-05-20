from orchestro_mesh.models import (
    BackendEndpoint,
    JobRecord,
    ModelCapability,
    NodeInventory,
    Sensitivity,
    TaskClass,
    UsageLedgerEntry,
)
from orchestro_mesh.store import MeshStore


def test_store_round_trips_node(tmp_path):
    store = MeshStore(tmp_path / "mesh.db")
    node = NodeInventory(
        node_id="n1",
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id="b1", base_url="http://localhost:8080/v1")],
        models=[ModelCapability(id="m1", backend_id="b1")],
    )
    store.upsert_node(node)
    nodes = store.list_nodes()
    assert len(nodes) == 1
    assert nodes[0].node_id == "n1"
    assert nodes[0].models[0].id == "m1"


def test_delta_node_jobs_increments_and_floors(tmp_path):
    store = MeshStore(tmp_path / "mesh.db")
    node = NodeInventory(
        node_id="n1",
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id="b1", base_url="http://localhost:8080/v1")],
        models=[ModelCapability(id="m1", backend_id="b1")],
    )
    store.upsert_node(node)
    store.delta_node_jobs("n1", 1)
    store.delta_node_jobs("n1", 1)
    assert store.get_node("n1").current_jobs == 2
    store.delta_node_jobs("n1", -5)
    assert store.get_node("n1").current_jobs == 0


def test_finish_job_writes_status_and_tokens(tmp_path):
    store = MeshStore(tmp_path / "mesh.db")
    job = JobRecord(
        id="j1",
        requester="keith",
        sensitivity=Sensitivity.PUBLIC,
        task_class=TaskClass.CHAT,
    )
    store.create_job(job)
    store.finish_job("j1", status="completed", input_tokens=10, output_tokens=20, total_ms=123.4)
    with store.connect() as conn:
        row = conn.execute(
            "SELECT status, input_tokens, output_tokens, total_ms, finished_at FROM jobs WHERE id = 'j1'"
        ).fetchone()
    assert row["status"] == "completed"
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 20
    assert row["total_ms"] == 123.4
    assert row["finished_at"] is not None


def test_ledger_totals(tmp_path):
    store = MeshStore(tmp_path / "mesh.db")
    store.add_ledger_entries(
        [
            UsageLedgerEntry(job_id="j1", requester="keith", node_id="n1", model_id="m1", credit_cost=1.5),
            UsageLedgerEntry(job_id="j2", requester="keith", node_id="n1", model_id="m1", credit_cost=2.5),
        ]
    )
    assert store.ledger_totals()["keith"] == 4.0

from orchestro_mesh.models import BackendEndpoint, ModelCapability, NodeInventory, UsageLedgerEntry
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


def test_ledger_totals(tmp_path):
    store = MeshStore(tmp_path / "mesh.db")
    store.add_ledger_entries(
        [
            UsageLedgerEntry(job_id="j1", requester="keith", node_id="n1", model_id="m1", credit_cost=1.5),
            UsageLedgerEntry(job_id="j2", requester="keith", node_id="n1", model_id="m1", credit_cost=2.5),
        ]
    )
    assert store.ledger_totals()["keith"] == 4.0

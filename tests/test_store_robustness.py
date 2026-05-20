import threading

from orchestro_mesh.models import (
    BackendEndpoint,
    ModelCapability,
    NodeInventory,
)
from orchestro_mesh.store import MeshStore


def test_delta_node_jobs_is_atomic_under_threads(tmp_path):
    store = MeshStore(tmp_path / "mesh.db")
    store.upsert_node(
        NodeInventory(
            node_id="n",
            owner="keith",
            trust_domain="local",
            backends=[BackendEndpoint(id="b", base_url="http://x/v1")],
            models=[ModelCapability(id="m", backend_id="b")],
        )
    )

    def bump():
        for _ in range(50):
            store.delta_node_jobs("n", 1)

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    node = store.get_node("n")
    assert node is not None
    # 4 threads * 50 increments = 200. Atomic counter must produce exactly this.
    assert node.current_jobs == 200


def test_delta_node_jobs_floors_at_zero(tmp_path):
    store = MeshStore(tmp_path / "mesh.db")
    store.upsert_node(
        NodeInventory(
            node_id="n",
            owner="keith",
            trust_domain="local",
            backends=[BackendEndpoint(id="b", base_url="http://x/v1")],
            models=[ModelCapability(id="m", backend_id="b")],
        )
    )
    store.delta_node_jobs("n", -5)
    assert store.get_node("n").current_jobs == 0

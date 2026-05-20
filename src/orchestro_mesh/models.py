from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


class Sensitivity(str, Enum):
    PUBLIC = "public"
    FRIENDS_PRIVATE = "friends-private"
    REPO_PRIVATE = "repo-private"
    SECRET_LOCAL = "secret-local"
    TOOL_LOCAL = "tool-local"


class TaskClass(str, Enum):
    CHAT = "chat"
    CODING = "coding"
    ANALYSIS = "analysis"
    TOOL_USE = "tool-use"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    BENCHMARK = "benchmark"
    BATCH = "batch"


class BackendKind(str, Enum):
    OPENAI_COMPAT = "openai-compat"
    LLAMA_CPP = "llama.cpp"
    OLLAMA = "ollama"
    VLLM = "vllm"
    TABBYAPI = "tabbyapi"
    SGLANG = "sglang"
    MLX = "mlx"
    MOCK = "mock"


class NodeStatus(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    DRAINING = "draining"
    OFFLINE = "offline"


class ModelState(str, Enum):
    ABSENT = "absent"
    COLD = "cold"
    LOADING = "loading"
    WARM = "warm"
    ACTIVE = "active"
    UNLOADING = "unloading"


class RouteDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    DEFER = "defer"


class BenchmarkSample(BaseModel):
    prompt_id: str = "default"
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None
    total_ms: float | None = None
    prompt_tps: float | None = None
    decode_tps: float | None = None
    end_to_end_tps: float | None = None
    tool_roundtrip_tps: float | None = None
    measured_at: datetime = Field(default_factory=now_utc)
    measured_by: str | None = None
    notes: str | None = None


class BenchmarkProfile(BaseModel):
    expected_prompt_tps: float | None = None
    expected_decode_tps: float | None = None
    expected_end_to_end_tps: float | None = None
    expected_tool_output_tps: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    samples: list[BenchmarkSample] = Field(default_factory=list)

    def best_decode_tps(self) -> float:
        observed = [s.decode_tps for s in self.samples if s.decode_tps and s.decode_tps > 0]
        if observed:
            return max(observed)
        return float(self.expected_decode_tps or 0.0)

    def best_tool_output_tps(self) -> float:
        observed = [s.tool_roundtrip_tps for s in self.samples if s.tool_roundtrip_tps and s.tool_roundtrip_tps > 0]
        if observed:
            return max(observed)
        return float(self.expected_tool_output_tps or self.expected_end_to_end_tps or self.expected_decode_tps or 0.0)


class GpuInfo(BaseModel):
    vendor: str
    model: str
    vram_gb: float
    driver: str | None = None
    backend: str | None = None
    compute_capability: str | None = None


class HardwareInfo(BaseModel):
    cpu_model: str | None = None
    cpu_threads: int | None = None
    ram_gb: float | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class BackendEndpoint(BaseModel):
    id: str
    kind: BackendKind = BackendKind.OPENAI_COMPAT
    base_url: HttpUrl
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_embeddings: bool = False
    supports_batching: bool = False
    timeout_s: float = 120.0
    headers: dict[str, str] = Field(default_factory=dict)


class ModelCapability(BaseModel):
    id: str
    backend_id: str
    display_name: str | None = None
    artifact_hash: str | None = None
    quantization: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    required_vram_gb: float | None = None
    required_ram_gb: float | None = None
    state: ModelState = ModelState.COLD
    tags: list[str] = Field(default_factory=list)
    task_classes: list[TaskClass] = Field(default_factory=lambda: [TaskClass.CHAT])
    benchmark: BenchmarkProfile = Field(default_factory=BenchmarkProfile)
    warm_since: datetime | None = None

    def supports_task(self, task_class: TaskClass) -> bool:
        return task_class in self.task_classes or (TaskClass.CHAT in self.task_classes and task_class == TaskClass.CHAT)


class NodePolicy(BaseModel):
    allowed_users: list[str] = Field(default_factory=list)
    allowed_sensitivities: list[Sensitivity] = Field(default_factory=lambda: [Sensitivity.PUBLIC])
    denied_sensitivities: list[Sensitivity] = Field(default_factory=lambda: [Sensitivity.SECRET_LOCAL, Sensitivity.TOOL_LOCAL])
    max_concurrent_jobs: int = Field(default=1, ge=1)
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    max_watts: int | None = None
    allow_remote_tool_authority: bool = False
    tags: list[str] = Field(default_factory=list)


class NodeInventory(BaseModel):
    node_id: str
    owner: str
    trust_domain: str = "local"
    status: NodeStatus = NodeStatus.ONLINE
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    backends: list[BackendEndpoint] = Field(default_factory=list)
    models: list[ModelCapability] = Field(default_factory=list)
    policy: NodePolicy = Field(default_factory=NodePolicy)
    current_jobs: int = 0
    queue_depth: int = 0
    last_seen: datetime = Field(default_factory=now_utc)
    advertised_at: datetime = Field(default_factory=now_utc)

    def model_by_id(self, model_id: str) -> ModelCapability | None:
        return next((model for model in self.models if model.id == model_id), None)

    def backend_by_id(self, backend_id: str) -> BackendEndpoint | None:
        return next((backend for backend in self.backends if backend.id == backend_id), None)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class InferenceRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    requester: str = "anonymous"
    sensitivity: Sensitivity = Sensitivity.PUBLIC
    task_class: TaskClass = TaskClass.CHAT
    messages: list[ChatMessage]
    model: str | None = None
    required_tags: list[str] = Field(default_factory=list)
    forbidden_tags: list[str] = Field(default_factory=list)
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    tool_authority_required: bool = False
    deadline_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def non_empty_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value


class RouteRejection(BaseModel):
    node_id: str
    model_id: str | None = None
    reason: str


class RouteCandidate(BaseModel):
    node: NodeInventory
    model: ModelCapability
    backend: BackendEndpoint
    score: float
    reasons: list[str] = Field(default_factory=list)


class RouteResult(BaseModel):
    decision: RouteDecision
    request_id: str
    selected: RouteCandidate | None = None
    candidates: list[RouteCandidate] = Field(default_factory=list)
    rejections: list[RouteRejection] = Field(default_factory=list)


class Heartbeat(BaseModel):
    node_id: str
    status: NodeStatus | None = None
    current_jobs: int | None = None
    queue_depth: int | None = None


class FeedbackRating(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    node_id: str | None = None
    model_id: str | None = None
    requester: str | None = None
    verifier: str = "human"
    rating: float = Field(ge=0.0, le=1.0)
    notes: str | None = None
    created_at: datetime = Field(default_factory=now_utc)


class JobRecord(BaseModel):
    id: str
    requester: str
    node_id: str | None = None
    model_id: str | None = None
    backend_id: str | None = None
    sensitivity: Sensitivity
    task_class: TaskClass
    status: str = "queued"
    created_at: datetime = Field(default_factory=now_utc)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_ms: float | None = None
    error: str | None = None


class UsageLedgerEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    requester: str
    node_id: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    gpu_ms_estimate: float = 0.0
    credit_cost: float = 0.0
    created_at: datetime = Field(default_factory=now_utc)

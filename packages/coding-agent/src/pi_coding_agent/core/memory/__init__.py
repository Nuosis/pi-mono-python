"""Project-local atomic memory for tau (design doc §8). P0: store + scaffolding."""
from .embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    cosine,
    embedding_provider_from_env,
)
from .models import (
    ConversationTurn,
    MemoryHit,
    MemoryStatus,
    MemoryType,
    Scope,
    ScopeType,
    SemanticMemory,
)
from .curator import CommitDecision, Curator, Evidence
from .dream import (
    DreamProposal,
    DreamResult,
    ScheduledDreamResult,
    dream_memory,
    last_dream_at,
    next_dream_at,
    parse_since,
    run_scheduled_dream,
)
from .recall import build_recall_block, latest_user_query
from .store import MemoryStore
from .working_context import (
    CtxBlock,
    WorkingContextConfig,
    profile_for,
)

__all__ = [
    "MemoryStore", "SemanticMemory", "MemoryHit", "ConversationTurn", "Scope",
    "MemoryType", "ScopeType", "MemoryStatus", "EmbeddingProvider",
    "OllamaEmbeddingProvider", "DeterministicEmbeddingProvider",
    "embedding_provider_from_env", "cosine",
    "Curator", "Evidence", "CommitDecision",
    "DreamProposal", "DreamResult", "ScheduledDreamResult", "dream_memory",
    "last_dream_at", "next_dream_at", "parse_since", "run_scheduled_dream",
    "build_recall_block", "latest_user_query",
    "WorkingContextConfig", "CtxBlock", "profile_for",
]

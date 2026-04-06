"""Memory tier modules — imports are lazy to allow partial usage
without all dependencies installed."""

from .episodic_buffer import EpisodicBuffer, TickObservation

__all__ = [
    "EpisodicBuffer", "TickObservation",
]

# Lazy imports for modules that require external deps (redis, qdrant)
def __getattr__(name):
    if name in ("AgentState", "StateStore"):
        from .state_store import AgentState, StateStore
        return {"AgentState": AgentState, "StateStore": StateStore}[name]
    if name in ("SemanticMemory", "EventRecord"):
        from .semantic_memory import SemanticMemory, EventRecord
        return {"SemanticMemory": SemanticMemory, "EventRecord": EventRecord}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

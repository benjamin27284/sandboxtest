"""Tier 2 — Semantic Memory (Vector DB via Qdrant).

Stores embeddings of past macroeconomic events and domain-specific knowledge.
When an exogenous shock is injected, agents perform RAG to retrieve the
top-K most relevant historical precedents, preventing hallucination.

Each agent can have its own collection or share a global one with
agent-specific metadata filtering.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "market_events"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


@dataclass
class EventRecord:
    """A stored event with its embedding and metadata."""
    event_id: str
    text: str
    category: str       # "monetary_policy", "earnings", "geopolitical", ...
    severity: float     # 0.0 – 1.0
    tick: int
    embedding: list[float]


class SemanticMemory:
    """Qdrant-backed semantic memory for RAG over historical events."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6334,
        embed_fn=None,
    ) -> None:
        self._client: Optional[AsyncQdrantClient] = None
        self._host = host
        self._port = port
        # embed_fn: async callable(text: str) -> list[float]
        # Use sentence-transformers locally or an embedding API.
        self._embed_fn = embed_fn

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(
            host=self._host, port=self._port, prefer_grpc=True
        )
        # Ensure collection exists
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in names:
            await self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def store_event(self, record: EventRecord) -> None:
        """Upsert an event embedding into the vector store."""
        point_id = self._deterministic_id(record.event_id)
        await self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=record.embedding,
                    payload={
                        "event_id": record.event_id,
                        "text": record.text,
                        "category": record.category,
                        "severity": record.severity,
                        "tick": record.tick,
                    },
                )
            ],
        )

    async def retrieve_relevant(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """RAG retrieval: embed the query and return top-K similar events.

        Returns list of dicts with keys: text, category, severity, tick, score.
        """
        if self._embed_fn is None:
            raise RuntimeError("No embedding function configured")

        query_vector = await self._embed_fn(query_text)

        search_filter = None
        if category_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category_filter),
                    )
                ]
            )

        results = await self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
        )

        return [
            {
                "text": hit.payload["text"],
                "category": hit.payload["category"],
                "severity": hit.payload["severity"],
                "tick": hit.payload["tick"],
                "score": hit.score,
            }
            for hit in results
        ]

    async def retrieve_as_context(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> str:
        """Convenience method: returns a formatted string for LLM context injection."""
        events = await self.retrieve_relevant(query_text, top_k=top_k)
        if not events:
            return "No relevant historical precedents found."

        lines = ["Relevant historical precedents (from memory):"]
        for i, ev in enumerate(events, 1):
            lines.append(
                f"  {i}. [tick {ev['tick']}, severity={ev['severity']:.1f}] "
                f"{ev['text']} (relevance={ev['score']:.2f})"
            )
        return "\n".join(lines)

    @staticmethod
    def _deterministic_id(event_id: str) -> str:
        """Hash event_id to a Qdrant-compatible UUID-like hex string."""
        return hashlib.md5(event_id.encode()).hexdigest()

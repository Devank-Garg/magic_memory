"""
archival.py  —  Layer 4: Archival Memory (Vector Store)

Long-term semantic memory. Every conversation turn is embedded and stored
in ChromaDB. At query time, semantically similar past messages are retrieved
and injected into context.

This is the "infinite" part — the full conversation history lives here,
searchable by meaning, not just recency.

Uses: sentence-transformers (local, no API key needed)
"""

import logging
import time

from agent_memory.config import MemoryConfig
from agent_memory.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_cfg = MemoryConfig()
_chroma = ChromaStore(
    chroma_path=_cfg.chroma_path,
    similarity_threshold=_cfg.archival_similarity_threshold,
    embedder_model=_cfg.embedder_model,
    archival_top_k=_cfg.archival_top_k,
)


def archive_message(user_id: str, role: str, content: str, message_id: int) -> None:
    """Embed and store a message in the vector store.

    Non-fatal: any ChromaDB or embedding failure is logged and swallowed
    so a vector-store outage never kills the main message pipeline (Issue 4).
    Uses upsert() instead of get()-then-add() to eliminate the redundant
    round-trip (Issue 7).
    """
    if len(content.strip()) < 10:
        return

    try:
        embedder   = _chroma.get_embedder()
        collection = _chroma.get_collection(user_id)
        doc_id     = f"{user_id}_{message_id}"
        embedding  = embedder.encode(content).tolist()

        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "role":       role,
                "timestamp":  time.time(),
                "message_id": message_id,
            }],
        )
    except Exception as e:
        logger.warning("archive_message failed for user=%s msg=%s: %s", user_id, message_id, e)


def search(user_id: str, query: str, n_results: int = 3) -> list[dict]:
    """Semantic search over archived messages. Returns most relevant past context."""
    try:
        embedder   = _chroma.get_embedder()
        collection = _chroma.get_collection(user_id)

        count = collection.count()
        if count == 0:
            return []

        n_results      = min(n_results, count)
        query_embedding = embedder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist < _chroma.similarity_threshold:
                retrieved.append({
                    "role":      meta["role"],
                    "content":   doc,
                    "relevance": round(1 - dist, 3),
                })

        return retrieved

    except Exception as e:
        logger.warning("archival search failed for user=%s: %s", user_id, e)
        return []


def render_for_prompt(user_id: str, query: str) -> str:
    """Retrieve relevant memories and format for system prompt."""
    results = search(user_id, query, n_results=_chroma.archival_top_k)
    if not results:
        return ""

    lines = ["## RETRIEVED MEMORIES (semantically relevant past context)"]
    for r in results:
        lines.append(f"  [{r['role'].upper()} | relevance: {r['relevance']}]: {r['content'][:300]}")

    return "\n".join(lines)

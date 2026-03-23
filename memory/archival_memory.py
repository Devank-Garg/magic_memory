"""
archival_memory.py  —  Layer 4: Archival Memory (Vector Store)

Long-term semantic memory. Every conversation turn is embedded and stored
in ChromaDB. At query time, semantically similar past messages are retrieved
and injected into context.

This is the "infinite" part — the full conversation history lives here,
searchable by meaning, not just recency.

Uses: sentence-transformers (local, no API key needed)
"""

import hashlib
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "chroma"


def _get_client():
    import chromadb
    return chromadb.PersistentClient(path=str(DB_PATH))


def _get_collection(user_id: str):
    client = _get_client()
    safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
    return client.get_or_create_collection(
        name=f"memory_{safe_id}",
        metadata={"hnsw:space": "cosine"}
    )


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    # Small, fast model — good for local use
    return SentenceTransformer("all-MiniLM-L6-v2")


_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = _get_embedder()
    return _embedder


def archive_message(user_id: str, role: str, content: str, message_id: int):
    """Embed and store a message in the vector store."""
    if len(content.strip()) < 10:
        return  # skip very short messages
    
    embedder = get_embedder()
    collection = _get_collection(user_id)
    
    # Use message_id as unique doc id
    doc_id = f"{user_id}_{message_id}"
    
    # Check if already archived
    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        return
    
    embedding = embedder.encode(content).tolist()
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "role": role,
            "timestamp": time.time(),
            "message_id": message_id
        }]
    )


def search(user_id: str, query: str, n_results: int = 3) -> list[dict]:
    """Semantic search over archived messages. Returns most relevant past context."""
    embedder = get_embedder()
    collection = _get_collection(user_id)
    
    count = collection.count()
    if count == 0:
        return []
    
    n_results = min(n_results, count)
    
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # Only include if reasonably similar (cosine distance < 0.7)
        if dist < 0.7:
            retrieved.append({
                "role": meta["role"],
                "content": doc,
                "relevance": round(1 - dist, 3)
            })
    
    return retrieved


def render_for_prompt(user_id: str, query: str) -> str:
    """Retrieve relevant memories and format for system prompt."""
    results = search(user_id, query, n_results=3)
    if not results:
        return ""
    
    lines = ["## RETRIEVED MEMORIES (semantically relevant past context)"]
    for r in results:
        lines.append(f"  [{r['role'].upper()} | relevance: {r['relevance']}]: {r['content'][:300]}")
    
    return "\n".join(lines)

"""
chroma_store.py  —  ChromaDB wrapper with error boundaries

Consolidates all ChromaDB and sentence-transformer access that was
previously scattered inline in archival.py. Provides a lazy-loaded
embedder singleton and safe collection access.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(
        self,
        chroma_path: str | Path,
        similarity_threshold: float = 0.7,
        embedder_model: str = "all-MiniLM-L6-v2",
        archival_top_k: int = 3,
    ):
        self.chroma_path = Path(chroma_path)
        self.similarity_threshold = similarity_threshold
        self.embedder_model = embedder_model
        self.archival_top_k = archival_top_k
        self._client = None
        self._embedder = None

    @staticmethod
    def _safe(user_id: str) -> str:
        """Sanitize user_id for use as a ChromaDB collection name suffix.

        Raises ValueError for empty/blank user_ids, matching the behaviour of
        SQLiteStore._safe to catch the error at the call site.
        """
        if not user_id or not user_id.strip():
            raise ValueError(f"user_id must not be empty, got {user_id!r}")
        return "".join(c if c.isalnum() else "_" for c in user_id)

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        return self._client

    def get_embedder(self):
        """Lazy-load the sentence-transformer model (singleton per instance)."""
        if self._embedder is None:
            import os
            import warnings

            # Silence HuggingFace hub auth nag and transformers weight-load chatter
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

            print("🧠 Preparing magic memory potion... (loading semantic model)", flush=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Suppress transformers / sentence-transformers logging
                import logging as _logging
                for _noisy in ("sentence_transformers", "transformers", "huggingface_hub"):
                    _logging.getLogger(_noisy).setLevel(_logging.ERROR)

                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    self.embedder_model,
                    tokenizer_kwargs={"clean_up_tokenization_spaces": True},
                )

            print("✨ Semantic memory ready.", flush=True)

        return self._embedder

    def get_collection(self, user_id: str):
        """Return (or create) the ChromaDB collection for this user."""
        client = self._get_client()
        return client.get_or_create_collection(
            name=f"memory_{self._safe(user_id)}",
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, user_id: str) -> None:
        """Delete the vector store collection for a user. Silent on failure."""
        try:
            client = self._get_client()
            client.delete_collection(f"memory_{self._safe(user_id)}")
        except Exception as e:
            logger.warning("delete_collection failed for %s: %s", user_id, e)

"""
============================================================
app/services/embedding_service.py — OpenAI Embedding Engine
============================================================
PURPOSE:
    Wraps the OpenAI text-embedding-3-small API to convert
    text strings into 1536-dimensional float vectors.

    This module is the bridge between raw text and the
    mathematical vector space that pgvector operates in.

WHY text-embedding-3-small?
    - 1536 dimensions (same as ada-002 but much better quality)
    - ~6x cheaper than text-embedding-3-large
    - Fast latency (~200ms per call)
    - Excellent for wellness/psychological text similarity

CONNECTED TO:
    Phase 1  → retrieval_engine.py: embeds incoming user message
    Phase 7  → consolidation.py: embeds extracted memory insights
    Phase 7  → dedup engine: embeds new facts to compare vs. existing
    Phase 9  → benchmark: embeds test queries for precision/recall testing
============================================================
"""
from openai import AsyncOpenAI
import structlog
from typing import Optional

from app.core.config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

# -------------------------------------------------------
# Shared AsyncOpenAI client (created once at module load)
# AsyncOpenAI uses an internal connection pool — one global
# instance is optimal for throughput.
# -------------------------------------------------------
_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    """Returns (or initializes) the shared AsyncOpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


async def embed_text(text: str) -> list[float]:
    """
    Converts a text string into a 1536-dimensional embedding vector
    using OpenAI's text-embedding-3-small model.

    The returned vector is used for:
    1. Cosine similarity search against Layer 3 semantic_memories
    2. Phase 7 deduplication checks (new fact vs. existing memories)

    Args:
        text : The text string to embed (max ~8191 tokens)

    Returns:
        A list of 1536 float values representing the semantic
        meaning of the input text in vector space.

    Raises:
        openai.APIError  : On API failure (network, rate limit, auth)
        ValueError       : If text is empty
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    # Truncate very long texts to avoid token limit errors
    # text-embedding-3-small supports up to 8191 tokens (~32KB)
    text = text.strip()[:10000]

    client = get_openai_client()

    try:
        response = await client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,  # text-embedding-3-small
            input=text,
            encoding_format="float",  # Return raw floats (not base64)
        )
        vector = response.data[0].embedding
        log.debug(
            "Text embedded",
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimensions=len(vector),
            text_preview=text[:80]
        )
        return vector

    except Exception as e:
        log.error("Embedding generation failed", error=str(e), text_preview=text[:80])
        raise


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of text strings in a single API call.
    OpenAI supports batch embedding — much more efficient than
    calling embed_text() in a loop.

    Used by:
        Phase 7 → Embed all extracted insights in one API call
        Phase 9 → Batch embed test queries for benchmark suite

    Args:
        texts : List of text strings (max 2048 items per batch)

    Returns:
        List of 1536-dimensional embedding vectors,
        in the same order as the input texts.
    """
    if not texts:
        return []

    # Clean and truncate
    cleaned = [t.strip()[:10000] for t in texts if t and t.strip()]

    client = get_openai_client()

    try:
        response = await client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=cleaned,
            encoding_format="float",
        )
        # Sort by index to maintain input order (API may reorder)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        vectors = [item.embedding for item in sorted_data]

        log.debug("Batch embedded", count=len(vectors))
        return vectors

    except Exception as e:
        log.error("Batch embedding failed", error=str(e), count=len(texts))
        raise

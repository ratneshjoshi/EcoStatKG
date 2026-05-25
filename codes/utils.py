"""
Shared Utilities
================
Common functions used across pipeline components.
"""

import time
import math
import logging
from dataclasses import dataclass, field

from tqdm import tqdm
from openai import OpenAI

from config import (
    API_BASE_URL, API_KEY, HTTP_CLIENT,
    EMBEDDING_MODEL,
    TEMPERATURE, MAX_TOKENS, TOP_P,
)

log = logging.getLogger("ecostats.utils")


def create_client() -> OpenAI:
    """Create OpenAI-compatible client for LLM API."""
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        http_client=HTTP_CLIENT,
    )


def embed_query(client: OpenAI, query: str) -> list[float]:
    """Embed a single query string."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    top_p: float = TOP_P,
    max_retries: int = 3,
    **kwargs,
):
    """
    Wrapper around client.chat.completions.create() that handles:
    1. Thinking-model quirks (qwen-27b returns content=None)
    2. Automatic retry with exponential backoff on connection/rate errors

    Returns the full response object with guaranteed non-None content.
    """
    extra_body = kwargs.pop("extra_body", None) or {}

    # qwen-3.6 is a thinking model — disable thinking for direct content
    if "qwen-3.6" in model or "qwen3" in model.lower():
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    # glm-5 and gpt-oss are reasoning models whose chain-of-thought counts
    # toward max_tokens; bump the limit so the actual answer isn't truncated
    if "glm" in model.lower() or "gpt-oss" in model.lower():
        max_tokens = max(max_tokens, 1024)
    else:
        # non-reasoning models: use 512 to reduce truncation in verbose answers
        max_tokens = max(max_tokens, 512)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body=extra_body if extra_body else None,
                **kwargs,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                log.warning("API call failed (attempt %d/%d): %s -- retrying in %ds",
                            attempt + 1, max_retries, e, wait)
                time.sleep(wait)
            else:
                log.error("API call failed after %d retries: %s", max_retries, e)
                raise last_error

    # Fallback: if content is still None, try reasoning fields
    choice = response.choices[0]
    if choice.message.content is None:
        reasoning = (
            getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
        )
        if reasoning:
            choice.message.content = reasoning
        else:
            choice.message.content = ""

    return response


def embed_batch(
    client: OpenAI,
    texts: list[str],
    batch_size: int = 20,
    max_calls_per_minute: int = 30,
) -> list[list[float]]:
    """Embed a list of texts, respecting rate limits (30 calls/min)."""
    all_embeddings = []
    call_count = 0
    window_start = time.time()
    total_batches = math.ceil(len(texts) / batch_size)

    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Embedding", unit="batch"):
        batch = texts[i:i + batch_size]
        call_count += 1
        if call_count >= max_calls_per_minute:
            elapsed = time.time() - window_start
            if elapsed < 60.0:
                wait = 60.0 - elapsed
                tqdm.write(f"  Rate limit: waiting {wait:.0f}s...")
                time.sleep(wait)
            call_count = 0
            window_start = time.time()

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])

    return all_embeddings


@dataclass
class RetrievalResult:
    """Container for retrieved triples with scores."""
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)

    def to_numbered_facts(self) -> str:
        lines = []
        for i, (doc, _meta) in enumerate(zip(self.documents, self.metadatas), 1):
            lines.append(f"[{i}] {doc}")
        return "\n".join(lines)


@dataclass
class GenerationResult:
    """Container for the full pipeline output."""
    query: str = ""
    model: str = ""
    method: str = ""
    answer: str = ""
    retrieved_triples: list[dict] = field(default_factory=list)
    top_k: int = 0
    granularity: str = "fine"
    latency_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "model": self.model,
            "method": self.method,
            "answer": self.answer,
            "retrieved_triples": self.retrieved_triples,
            "top_k": self.top_k,
            "granularity": self.granularity,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
        }

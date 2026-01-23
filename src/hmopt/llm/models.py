"""LLM + embedding builders for LlamaIndex indexing."""

from __future__ import annotations

import os

from hmopt.core.config import AppConfig
from hmopt.llm.openai_like import OpenAILike, OpenAIEmbeddingLike


def build_llama_models(config: AppConfig) -> tuple[OpenAILike, OpenAIEmbeddingLike]:
    if not config.llm.api_key:
        raise RuntimeError("LLM API key is required for LlamaIndex indexing")
    if config.llm.api_key:
        os.environ.setdefault("OPENAI_API_KEY", config.llm.api_key)
    if config.llm.base_url:
        os.environ.setdefault("OPENAI_API_BASE", config.llm.base_url)
    llm = OpenAILike(
        model=config.llm.model,
        api_key=config.llm.api_key,
        api_base=config.llm.base_url,
        timeout=120,
        max_tokens=2048,
        temperature=0,
    )
    embed = OpenAIEmbeddingLike(
        model=config.llm.embedding_model,
        api_base=config.llm.base_url,
        api_key=config.llm.api_key,
    )
    return llm, embed

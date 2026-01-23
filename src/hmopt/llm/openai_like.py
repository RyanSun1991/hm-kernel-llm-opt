"""OpenAI-compatible wrappers for LlamaIndex without strict model name validation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai.base import (
    _QUERY_MODE_MODEL_DICT,
    _TEXT_MODE_MODEL_DICT,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
    get_engine,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai import utils as openai_utils

DEFAULT_CONTEXT_WINDOW = 8192


class OpenAILike(OpenAI):
    """OpenAI-compatible LLM wrapper that tolerates unknown model names."""

    @property
    def _tokenizer(self):  # type: ignore[override]
        try:
            return super()._tokenizer
        except Exception:  # pragma: no cover - depends on model name
            return None

    @property
    def metadata(self) -> LLMMetadata:
        model_name = self._get_model_name()
        try:
            context_window = openai_utils.openai_modelname_to_contextsize(model_name)
        except ValueError:
            context_window = DEFAULT_CONTEXT_WINDOW

        is_chat_model = openai_utils.is_chat_model(model_name)
        if model_name not in openai_utils.ALL_AVAILABLE_MODELS:
            is_chat_model = True

        return LLMMetadata(
            context_window=context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=is_chat_model,
            is_function_calling_model=openai_utils.is_function_calling_model(model_name),
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


class OpenAIEmbeddingLike(OpenAIEmbedding):
    """OpenAI-compatible embedding wrapper that tolerates unknown model names."""

    def __init__(
        self,
        mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
        model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
        embed_batch_size: int = 100,
        dimensions: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        if dimensions is not None:
            additional_kwargs["dimensions"] = dimensions

        api_key, api_base, api_version = self._resolve_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

        try:
            query_engine = get_engine(mode, model, _QUERY_MODE_MODEL_DICT)
            text_engine = get_engine(mode, model, _TEXT_MODE_MODEL_DICT)
        except ValueError:
            query_engine = model
            text_engine = model

        if "model_name" in kwargs:
            model_name = kwargs.pop("model_name")
            query_engine = text_engine = model_name
        else:
            model_name = model

        super().__init__(
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            model_name=model_name,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            default_headers=default_headers,
            num_workers=num_workers,
            **kwargs,
        )
        self._query_engine = query_engine
        self._text_engine = text_engine

        self._client = None
        self._aclient = None
        self._http_client = http_client
        self._async_http_client = async_http_client

"""OpenAI-like wrapper for LlamaIndex without model name validation."""

from __future__ import annotations

from llama_index.core.base.llms.types import LLMMetadata, MessageRole
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

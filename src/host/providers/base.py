"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] | None
    thinking: str | None
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


@dataclass
class StreamChunk:
    type: str  # "text", "thinking", "tool_use", "done", "error"
    content: str = ""
    tool_call: dict | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract LLM provider. Implement this to add a new provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'anthropic', 'openai', 'litellm')"""

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Return True if this provider handles the given model string."""

    @abstractmethod
    async def complete(self, params: dict[str, Any]) -> LLMResponse:
        """Non-streaming completion. params uses OpenAI message format."""

    @abstractmethod
    async def stream(self, params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Streaming completion. Yields StreamChunk objects."""

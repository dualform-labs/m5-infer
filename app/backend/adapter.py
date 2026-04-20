"""Abstract backend protocol for the M5 MLX Inference Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol


@dataclass
class GenerationChunk:
    """A single chunk of generated output."""

    text: str
    token: int
    finish_reason: str | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    logprobs: Any = None


class BackendAdapter(Protocol):
    """Protocol that all inference backends must implement."""

    def load_model(self, path: str, **kwargs) -> None: ...

    def is_loaded(self) -> bool: ...

    def unload_model(self) -> None: ...

    def generate_stream(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Iterator[GenerationChunk]: ...

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, tokens: list[int]) -> str: ...

    def get_model_name(self) -> str: ...

    def get_memory_usage(self) -> dict[str, float]: ...

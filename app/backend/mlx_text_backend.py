"""MLX text inference backend wrapping mlx_lm."""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx
from mlx_lm import load as mlx_load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from app.backend.adapter import GenerationChunk
from app.core.logging import get_logger

logger = get_logger(__name__)


class MLXTextBackend:
    """MLX text inference backend wrapping mlx_lm."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_path: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self, path: str, **kwargs) -> None:
        """Load an MLX model from *path* and run a warmup pass."""
        self._model, self._tokenizer = mlx_load(path, **kwargs)
        self._model_path = path
        self._warmup()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_path = ""
        mx.clear_cache()

    def get_tokenizer(self):
        """Return the tokenizer instance (for prompt construction)."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        return self._tokenizer

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Iterator[GenerationChunk]:
        """Yield :class:`GenerationChunk` objects from ``mlx_lm.stream_generate``.

        Note: The bare ``return`` for empty prompt_tokens is intentional —
        in a generator function, ``return`` without a value terminates the
        iterator cleanly (yields nothing), which is the correct behaviour
        when there is nothing to generate.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        if not prompt_tokens:
            return

        prompt_array = mx.array(prompt_tokens)

        # Build sampler and logits processors for mlx_lm 0.31+ API
        sampler = make_sampler(temp=temperature, top_p=top_p)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
        )

        for response in stream_generate(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt_array,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            **kwargs,
        ):
            yield GenerationChunk(
                text=response.text,
                token=response.token,
                finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                generation_tokens=response.generation_tokens,
                prompt_tps=response.prompt_tps,
                generation_tps=response.generation_tps,
                peak_memory_gb=response.peak_memory,
            )

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        return self._tokenizer.encode(text)

    def detokenize(self, tokens: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        return self._tokenizer.decode(tokens)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return self._model_path.split("/")[-1] if self._model_path else ""

    def get_memory_usage(self) -> dict[str, float]:
        return {
            "active_gb": mx.get_active_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
            "cache_gb": mx.get_cache_memory() / 1e9,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Run a minimal inference to compile Metal kernels.

        Warmup failure is non-fatal (the model still works for first-real-
        request cost) but MUST be surfaced so memory-exhaustion-at-load is
        not silently masked.
        """
        if self._model is None or self._tokenizer is None:
            return
        try:
            # Use the actual chat template so warmup exercises the real path.
            try:
                prompt = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": "hi"}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt = "hello"
            for _ in stream_generate(
                self._model, self._tokenizer, prompt, max_tokens=2,
            ):
                pass
        except Exception:
            logger.exception("Model warmup failed (non-fatal)")

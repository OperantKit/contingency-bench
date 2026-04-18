"""LLM-as-subject client interfaces.

The benchmark drives an "LLM subject" through a concurrent VI-VI session and
fits a generalized matching law to the resulting choice allocation. Two client
flavors are provided:

- :class:`MockLLMClient` — deterministic, dependency-free; uses the inlined
  :class:`~contingency_bench.policy_compression.PolicyCompressionAgent` so
  benchmarks can run hermetically (CI, smoke tests, beta sweeps).
- :class:`APILLMClient` — talks to the Anthropic Claude API. The ``anthropic``
  SDK is imported **lazily**: importing this module never requires the SDK;
  only constructing an :class:`APILLMClient` does. Install via::

      pip install "contingency-bench[llm]"

The original v0 driver supported multiple vendors via ``litellm``. This port
adopts the single-vendor Anthropic SDK to match the new repository's standard
LLM dependency. Re-introducing multi-vendor support is a follow-up.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from contingency_bench.policy_compression import PolicyCompressionAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """Abstract LLM-subject interface."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_action(self, prompt: str) -> int:
        """Return a discrete action index for the given prompt."""
        raise NotImplementedError


class APILLMClient(BaseLLMClient):
    """Live Anthropic API client.

    The ``anthropic`` package is imported lazily on first construction so that
    importing :mod:`contingency_bench.llm_client` does not require the optional
    ``[llm]`` extra to be installed.
    """

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "APILLMClient requires the 'anthropic' SDK. "
                    "Install with: pip install 'contingency-bench[llm]'"
                ) from exc
            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=resolved_key)

    def get_action(self, prompt: str) -> int:
        full_prompt = (
            "あなたは心理学実験の参加者です。"
            f"{prompt}\n答えは必ず数字の '1' か '2' だけで答えてください。"
        )
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}],
            )
            text = self._extract_text(response).strip()
        except Exception as exc:
            logger.warning("API error (%s): %s. Using fallback choice.", self.model_name, exc)
            return int(np.random.choice([0, 1]))

        if "1" in text and "2" not in text:
            return 0
        if "2" in text and "1" not in text:
            return 1
        return int(np.random.choice([0, 1]))

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the first text block from an Anthropic ``Message`` response."""
        content = getattr(response, "content", None)
        if not content:
            return ""
        first = content[0]
        text_attr = getattr(first, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        if isinstance(first, dict):
            value = first.get("text", "")
            return value if isinstance(value, str) else ""
        return ""


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client backed by a :class:`PolicyCompressionAgent`."""

    def __init__(self, model_name: str, inherent_beta: float = 0.2) -> None:
        super().__init__(model_name)
        self.agent = PolicyCompressionAgent(n_actions=2, beta=inherent_beta, learning_rate=0.1)

    def get_action(self, prompt: str) -> int:  # noqa: ARG002 - prompt unused for mock
        return self.agent.choose_action()

    def update(self, action: int, reward: float) -> None:
        self.agent.update(action, reward)

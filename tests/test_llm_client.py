"""LLM client tests, including the lazy ``anthropic`` import contract."""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest

from contingency_bench.llm_client import APILLMClient, BaseLLMClient, MockLLMClient


def test_base_get_action_is_abstract() -> None:
    base = BaseLLMClient("dummy")
    with pytest.raises(NotImplementedError):
        base.get_action("anything")


def test_mock_client_returns_valid_action() -> None:
    client = MockLLMClient("mock", inherent_beta=0.2)
    action = client.get_action("any prompt")
    assert action in (0, 1)


def test_mock_client_update_does_not_raise() -> None:
    client = MockLLMClient("mock", inherent_beta=0.2)
    client.update(action=0, reward=1.0)
    client.update(action=1, reward=0.0)


def test_api_client_raises_clean_import_error_when_anthropic_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing an APILLMClient without the SDK installed must raise
    ImportError with an actionable install hint, never a generic AttributeError
    or NameError leak."""
    real_import = builtins.__import__

    def blocked_import(
        name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0
    ) -> Any:
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"contingency-bench\[llm\]"):
        APILLMClient("claude-haiku-4-5-20251001")


def test_api_client_accepts_injected_client() -> None:
    """When a pre-built client is injected, no SDK import is attempted."""

    class FakeMessages:
        def create(self, **_kwargs: Any) -> Any:
            class FakeBlock:
                text = "1"

            class FakeResponse:
                content = [FakeBlock()]

            return FakeResponse()

    class FakeAnthropic:
        messages = FakeMessages()

    client = APILLMClient("claude-haiku-4-5-20251001", client=FakeAnthropic())
    action = client.get_action("どっち？")
    assert action == 0


def test_api_client_falls_back_on_api_exception() -> None:
    class ExplodingMessages:
        def create(self, **_kwargs: Any) -> Any:
            raise RuntimeError("transient API failure")

    class ExplodingAnthropic:
        messages = ExplodingMessages()

    client = APILLMClient("claude-haiku-4-5-20251001", client=ExplodingAnthropic())
    action = client.get_action("どっち？")
    assert action in (0, 1)

"""LLM configuration models.

This module defines configuration classes for LLM clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM clients.

    Attributes:
        provider: Provider type (openai_compat, anthropic, etc.).
        base_url: API base URL.
        api_key: API key for authentication.
        model: Model name or path.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        retry_delay: Base delay between retries in seconds.
        rate_limit_rpm: Maximum requests per minute (0 = unlimited).
        rate_limit_tpm: Maximum tokens per minute (0 = unlimited).
        extra: Additional provider-specific options.
    """

    provider: str = "openai_compat"
    base_url: str = "http://127.0.0.1:5000/v1"
    api_key: str = "local"
    model: str = "Mistral-7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1.0
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 0
    rate_limit_tpm: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            LLMConfig instance.
        """
        known_keys = {
            "provider",
            "base_url",
            "api_key",
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "timeout",
            "max_retries",
            "retry_delay",
            "rate_limit_rpm",
            "rate_limit_tpm",
        }

        known = {k: v for k, v in data.items() if k in known_keys}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(**known, extra=extra)

    @classmethod
    def for_local_model(
        cls,
        model: str = "Mistral-7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        base_url: str = "http://127.0.0.1:5000/v1",
    ) -> LLMConfig:
        """Create configuration for local text-generation-webui.

        Args:
            model: Model name/path.
            base_url: API endpoint URL.

        Returns:
            LLMConfig configured for local use.
        """
        return cls(
            provider="openai_compat",
            base_url=base_url,
            api_key="local",
            model=model,
        )

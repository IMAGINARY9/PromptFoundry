"""Base LLM client class.

This module provides the abstract base class for LLM client adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM backend adapters.

    Subclasses must implement the completion methods for their
    specific backend (OpenAI API, local models, etc.).
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for a prompt.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt.
            **kwargs: Additional generation parameters.

        Returns:
            The generated completion text.
        """
        pass

    async def complete_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate completions for multiple prompts.

        Default implementation calls complete() sequentially.
        Subclasses may override for parallel execution.

        Args:
            prompts: List of prompts to complete.
            system_prompt: Optional system prompt for all.
            **kwargs: Additional generation parameters.

        Returns:
            List of completions, one per prompt.
        """
        results = []
        for prompt in prompts:
            result = await self.complete(prompt, system_prompt, **kwargs)
            results.append(result)
        return results

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM backend is reachable.

        Returns:
            True if backend is healthy, False otherwise.
        """
        pass

    def get_client_info(self) -> dict[str, Any]:
        """Return information about this client.

        Returns:
            Dictionary with client metadata.
        """
        return {
            "name": self.__class__.__name__,
        }

"""OpenAI-compatible LLM client.

This module provides a client for OpenAI-compatible APIs,
including local models via text-generation-webui.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from promptfoundry.llm.base import BaseLLMClient
from promptfoundry.llm.config import LLMConfig
from promptfoundry.llm.rate_limiter import RateLimiter


class OpenAICompatClient(BaseLLMClient):
    """Client for OpenAI-compatible API endpoints.

    Works with:
    - OpenAI API
    - text-generation-webui (local)
    - vLLM
    - Any OpenAI-compatible server
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            config: LLM configuration. Uses defaults for local model if None.
        """
        self.config = config or LLMConfig.for_local_model()
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = RateLimiter(
            rpm=self.config.rate_limit_rpm,
            tpm=self.config.rate_limit_tpm,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            Configured httpx AsyncClient.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion using chat completions API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Override generation parameters.

        Returns:
            Generated text.

        Raises:
            httpx.HTTPError: On network errors.
            ValueError: On invalid response.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Apply rate limiting
        estimated_tokens = len(prompt) // 4 + self.config.max_tokens
        await self._rate_limiter.acquire_request(estimated_tokens)

        # Retry logic
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                client = await self._get_client()
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()

                data = response.json()
                return self._extract_content(data)

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    # Retryable error
                    delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

            except httpx.RequestError as e:
                last_error = e
                delay = self.config.retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise RuntimeError(f"Max retries exceeded: {last_error}")

    def _extract_content(self, response: dict[str, Any]) -> str:
        """Extract content from API response.

        Args:
            response: Raw API response.

        Returns:
            The generated text content.

        Raises:
            ValueError: If response format is unexpected.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")

            message = choices[0].get("message", {})
            content: str = str(message.get("content", ""))

            if not content:
                # Try alternative formats
                content = str(choices[0].get("text", ""))

            return content.strip()

        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format: {e}") from e

    async def complete_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate completions for multiple prompts concurrently.

        Args:
            prompts: List of prompts.
            system_prompt: Optional system prompt for all.
            **kwargs: Generation parameters.

        Returns:
            List of completions.
        """
        tasks = [self.complete(prompt, system_prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """Check if the API endpoint is reachable.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def get_client_info(self) -> dict[str, Any]:
        """Return client information."""
        return {
            "name": self.__class__.__name__,
            "provider": self.config.provider,
            "base_url": self.config.base_url,
            "model": self.config.model,
        }

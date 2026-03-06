"""Tests for LLM clients."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from promptfoundry.llm.config import LLMConfig
from promptfoundry.llm.openai_compat import OpenAICompatClient


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == "openai_compat"
        assert config.base_url == "http://127.0.0.1:5000/v1"
        assert config.api_key == "local"
        assert config.temperature == 0.7

    def test_for_local_model(self) -> None:
        """Test local model configuration factory."""
        config = LLMConfig.for_local_model(
            model="test-model",
            base_url="http://localhost:8000/v1",
        )
        assert config.model == "test-model"
        assert config.base_url == "http://localhost:8000/v1"
        assert config.api_key == "local"

    def test_to_dict(self) -> None:
        """Test configuration serialization."""
        config = LLMConfig()
        data = config.to_dict()

        assert "provider" in data
        assert "model" in data
        assert "api_key" not in data  # Should not expose API key

    def test_from_dict(self) -> None:
        """Test configuration deserialization."""
        data = {
            "provider": "test",
            "base_url": "http://test:5000/v1",
            "model": "test-model",
            "temperature": 0.5,
            "custom_option": "value",  # Extra option
        }
        config = LLMConfig.from_dict(data)

        assert config.provider == "test"
        assert config.temperature == 0.5
        assert config.extra["custom_option"] == "value"


class TestOpenAICompatClient:
    """Tests for OpenAICompatClient."""

    @pytest.fixture
    def client(self) -> OpenAICompatClient:
        """Create a client for testing."""
        config = LLMConfig(
            base_url="http://test:5000/v1",
            api_key="test-key",
            model="test-model",
        )
        return OpenAICompatClient(config)

    def test_client_creation(self, client: OpenAICompatClient) -> None:
        """Test client initialization."""
        assert client.config.base_url == "http://test:5000/v1"
        assert client.config.model == "test-model"

    @pytest.mark.asyncio
    async def test_extract_content(self, client: OpenAICompatClient) -> None:
        """Test response content extraction."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, World!",
                    }
                }
            ]
        }
        content = client._extract_content(response)
        assert content == "Hello, World!"

    @pytest.mark.asyncio
    async def test_extract_content_alternative_format(self, client: OpenAICompatClient) -> None:
        """Test alternative response format extraction."""
        response = {
            "choices": [
                {
                    "text": "Alternative format",
                }
            ]
        }
        content = client._extract_content(response)
        assert content == "Alternative format"

    @pytest.mark.asyncio
    async def test_extract_content_invalid(self, client: OpenAICompatClient) -> None:
        """Test error on invalid response."""
        response = {"invalid": "format"}
        with pytest.raises(ValueError, match="No choices"):
            client._extract_content(response)

    def test_get_client_info(self, client: OpenAICompatClient) -> None:
        """Test client info retrieval."""
        info = client.get_client_info()
        assert info["name"] == "OpenAICompatClient"
        assert info["provider"] == "openai_compat"
        assert info["base_url"] == "http://test:5000/v1"

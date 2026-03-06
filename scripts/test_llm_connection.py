#!/usr/bin/env python
"""Script to test LLM connection.

Usage:
    python scripts/test_llm_connection.py
    python scripts/test_llm_connection.py --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def test_connection(base_url: str, model: str | None = None) -> bool:
    """Test connection to LLM backend.

    Args:
        base_url: API base URL.
        model: Model name (optional).

    Returns:
        True if connection successful.
    """
    from promptfoundry.llm import OpenAICompatClient, LLMConfig

    config = LLMConfig(base_url=base_url)
    if model:
        config.model = model

    client = OpenAICompatClient(config)

    print(f"Testing connection to {base_url}...")

    # Health check
    healthy = await client.health_check()
    if not healthy:
        print("✗ Health check failed - server may be unreachable")
        await client.close()
        return False

    print("✓ Health check passed")

    # Try a simple completion
    try:
        print("Testing completion...")
        response = await client.complete(
            "Say 'Hello, PromptFoundry!' and nothing else.",
            max_tokens=20,
        )
        print(f"✓ Got response: {response}")
    except Exception as e:
        print(f"✗ Completion failed: {e}")
        await client.close()
        return False

    await client.close()
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test LLM connection")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5000/v1",
        help="LLM API base URL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (optional)",
    )

    args = parser.parse_args()

    success = asyncio.run(test_connection(args.base_url, args.model))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

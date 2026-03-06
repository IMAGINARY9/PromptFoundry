#!/usr/bin/env python
"""
PromptFoundry - Local LLM Connection Test

Validates that the local LLM backend (text-generation-webui) is accessible
and can generate completions for prompt optimization tasks.

Setup Instructions:
    1. Install text-generation-webui: https://github.com/oobabooga/text-generation-webui
    2. Download a model (recommended: Mistral-7B-Instruct)
    3. Start server with API: python server.py --api --listen
    4. Default endpoint: http://127.0.0.1:5000/v1

Usage:
    python scripts/test_llm_connection.py
    python scripts/test_llm_connection.py --base-url http://localhost:8000/v1
    python scripts/test_llm_connection.py --model mistral-7b-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def print_header() -> None:
    """Print script header."""
    print("=" * 60)
    print("PromptFoundry - Local LLM Connection Test")
    print("=" * 60)
    print()


def print_footer(success: bool) -> None:
    """Print script footer."""
    print()
    print("=" * 60)
    status = "✓ All tests passed" if success else "✗ Some tests failed"
    print(status)
    print("=" * 60)


async def test_connection(base_url: str, model: str | None = None) -> bool:
    """Test basic connection to LLM backend.

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

    print(f"Endpoint: {base_url}")
    print(f"Model: {model or 'default'}")
    print()

    # Test 1: Health check
    print("[1/3] Health check...")
    healthy = await client.health_check()
    if not healthy:
        print("  ✗ Server unreachable")
        print()
        print("Troubleshooting:")
        print("  - Is text-generation-webui running?")
        print("  - Start with: python server.py --api --listen")
        print("  - Check port: default is 5000")
        await client.close()
        return False
    print("  ✓ Server is healthy")

    # Test 2: Simple completion
    print()
    print("[2/3] Basic completion...")
    try:
        response = await client.complete(
            "Say 'Hello, PromptFoundry!' and nothing else.",
            max_tokens=32,
            temperature=0.1,
        )
        print(f"  ✓ Response: {response[:80]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print()
        print("Note: Model may need to be loaded in text-generation-webui")
        await client.close()
        return False

    # Test 3: Prompt optimization style test
    print()
    print("[3/3] Prompt optimization test...")
    try:
        optimization_prompt = """You are a prompt engineering expert.
Given this prompt that sometimes fails:
"Classify the sentiment as positive or negative: {input}"

Suggest ONE improved version that would be more reliable.
Output only the improved prompt, nothing else."""

        response = await client.complete(
            optimization_prompt,
            max_tokens=100,
            temperature=0.7,
        )
        print(f"  ✓ Got optimization suggestion ({len(response)} chars)")
        print(f"  Preview: {response[:100]}...")
    except Exception as e:
        print(f"  ⚠ Failed (non-critical): {e}")
        # Non-critical - basic completion worked

    await client.close()
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test LLM connection for PromptFoundry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (localhost:5000)
    python scripts/test_llm_connection.py
    
    # Custom endpoint
    python scripts/test_llm_connection.py --base-url http://192.168.1.10:5000/v1
    
    # Specify model
    python scripts/test_llm_connection.py --model mistral-7b-instruct
        """,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5000/v1",
        help="LLM API base URL (default: http://127.0.0.1:5000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (uses server default if not specified)",
    )

    args = parser.parse_args()

    print_header()
    success = asyncio.run(test_connection(args.base_url, args.model))
    print_footer(success)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

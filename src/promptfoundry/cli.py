"""PromptFoundry CLI entry point.

This module provides the command-line interface for PromptFoundry.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="promptfoundry",
    help="Optimization-driven prompt engineering tool",
    add_completion=False,
)
console = Console()


@app.command()
def optimize(
    task: Path = typer.Option(
        ...,
        "--task",
        "-t",
        help="Path to task YAML file",
        exists=True,
        readable=True,
    ),
    seed_prompt: str = typer.Option(
        ...,
        "--seed-prompt",
        "-s",
        help="Initial prompt to optimize (use {input} for placeholder)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
    ),
    strategy: str = typer.Option(
        "evolutionary",
        "--strategy",
        help="Optimization strategy: evolutionary, bayesian, grid",
    ),
    max_generations: int = typer.Option(
        50,
        "--max-generations",
        "-g",
        help="Maximum number of generations",
    ),
    population_size: int = typer.Option(
        20,
        "--population-size",
        "-p",
        help="Population size",
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output",
        "-o",
        help="Output directory for results",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Optimize a prompt for a given task.

    Example:
        promptfoundry optimize --task task.yaml --seed-prompt "Classify: {input}"
    """
    console.print(
        Panel(
            f"[bold blue]PromptFoundry[/bold blue] - Prompt Optimization\n"
            f"Task: {task.name}\n"
            f"Strategy: {strategy}\n"
            f"Generations: {max_generations}",
            title="Starting Optimization",
        )
    )

    # This is a stub - full implementation in future versions
    console.print("\n[yellow]Note: Full optimization not yet implemented.[/yellow]")
    console.print("Run unit tests to verify component functionality.\n")

    # Demo output
    table = Table(title="Optimization Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Task file", str(task))
    table.add_row("Seed prompt", seed_prompt[:50] + "..." if len(seed_prompt) > 50 else seed_prompt)
    table.add_row("Strategy", strategy)
    table.add_row("Max generations", str(max_generations))
    table.add_row("Population size", str(population_size))
    table.add_row("Output directory", str(output_dir))

    console.print(table)


@app.command()
def validate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file to validate",
        exists=True,
        readable=True,
    ),
) -> None:
    """Validate a configuration file.

    Example:
        promptfoundry validate --config config.yaml
    """
    import yaml

    try:
        with config.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        console.print("[green]✓[/green] Configuration file is valid YAML")

        # Check required sections
        required_sections = ["llm"]
        for section in required_sections:
            if section in data:
                console.print(f"[green]✓[/green] Found '{section}' section")
            else:
                console.print(f"[yellow]![/yellow] Missing '{section}' section (will use defaults)")

    except yaml.YAMLError as e:
        console.print(f"[red]✗[/red] Invalid YAML: {e}")
        raise typer.Exit(1)


@app.command()
def test_llm(
    base_url: str = typer.Option(
        "http://127.0.0.1:5000/v1",
        "--base-url",
        "-u",
        help="LLM API base URL",
    ),
    prompt: str = typer.Option(
        "Say 'Hello, PromptFoundry!' and nothing else.",
        "--prompt",
        "-p",
        help="Test prompt to send",
    ),
) -> None:
    """Test connection to LLM backend.

    Example:
        promptfoundry test-llm --base-url http://localhost:5000/v1
    """
    from promptfoundry.llm import LLMConfig, OpenAICompatClient

    async def _test() -> None:
        config = LLMConfig(base_url=base_url)
        client = OpenAICompatClient(config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Checking health...", total=None)
            healthy = await client.health_check()

        if not healthy:
            console.print(f"[red]✗[/red] Cannot connect to {base_url}")
            console.print("  Make sure your LLM server is running.")
            await client.close()
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Connected to {base_url}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating response...", total=None)
            response = await client.complete(prompt)

        console.print(f"\n[bold]Response:[/bold] {response}")
        await client.close()

    asyncio.run(_test())


@app.command()
def version() -> None:
    """Show version information."""
    from promptfoundry import __version__

    console.print(f"PromptFoundry v{__version__}")


@app.command()
def list_strategies() -> None:
    """List available optimization strategies."""
    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Status")

    table.add_row(
        "evolutionary",
        "Genetic algorithm with mutation and crossover",
        "[green]Available[/green]",
    )
    table.add_row(
        "bayesian",
        "Bayesian optimization with Optuna",
        "[yellow]MVP 2[/yellow]",
    )
    table.add_row(
        "grid",
        "Grid search over prompt components",
        "[yellow]MVP 2[/yellow]",
    )

    console.print(table)


@app.command()
def list_evaluators() -> None:
    """List available evaluators."""
    table = Table(title="Available Evaluators")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Status")

    table.add_row(
        "exact_match",
        "Exact string match (case-insensitive by default)",
        "[green]Available[/green]",
    )
    table.add_row(
        "fuzzy_match",
        "Fuzzy string matching with Levenshtein distance",
        "[green]Available[/green]",
    )
    table.add_row(
        "regex",
        "Regex pattern matching",
        "[green]Available[/green]",
    )
    table.add_row(
        "json_schema",
        "JSON schema validation",
        "[yellow]MVP 2[/yellow]",
    )
    table.add_row(
        "custom",
        "Custom Python scoring function",
        "[yellow]MVP 2[/yellow]",
    )

    console.print(table)


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()

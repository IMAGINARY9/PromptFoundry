"""PromptFoundry CLI entry point.

This module provides the command-line interface for PromptFoundry.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from promptfoundry.core import (
    BenchmarkSummary,
    Example,
    OptimizationResult,
    Optimizer,
    OptimizerConfig,
    Prompt,
    RunDiagnostics,
    RuntimeConfig,
    Task,
    create_custom_gate,
    format_diagnostics_report,
    get_available_profiles,
    get_profile_description,
)
from promptfoundry.evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    FieldCoverageEvaluator,
    FuzzyMatchEvaluator,
    JsonParseEvaluator,
    JsonSchemaEvaluator,
    KeywordPresenceEvaluator,
    LengthConstraintEvaluator,
    OutputShapeEvaluator,
    RegexEvaluator,
)
from promptfoundry.llm import LLMConfig, OpenAICompatClient
from promptfoundry.strategies import GeneticAlgorithmStrategy
from promptfoundry.strategies.evolutionary import EvolutionaryConfig

app = typer.Typer(
    name="promptfoundry",
    help="Optimization-driven prompt engineering tool",
    add_completion=False,
)
console = Console()


def _apply_runtime_llm_overrides(
    llm_config: LLMConfig,
    runtime_config: RuntimeConfig,
    llm_settings: dict[str, Any] | None = None,
) -> LLMConfig:
    """Apply runtime-derived LLM overrides with config-file precedence.

    Runtime profiles define request timeout behavior for slow local backends.
    If the user did not explicitly set an LLM timeout in config, inherit the
    effective runtime timeout so the actual HTTP client matches the profile.
    """
    llm_settings = llm_settings or {}
    if "timeout" not in llm_settings and "timeout_per_request" not in llm_settings:
        llm_config.timeout = runtime_config.timeout_per_request
    return llm_config


def _load_task(task_path: Path) -> tuple[Task, str, dict[str, Any]]:
    """Load a task from a YAML file.

    Returns:
        Tuple of (Task, evaluator_type, evaluator_config).
    """
    with task_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    examples = [
        Example(
            input=ex["input"],
            expected_output=ex["expected"],
            metadata=ex.get("metadata", {}),
        )
        for ex in data.get("examples", [])
    ]

    task = Task(
        name=data["name"],
        examples=examples,
        system_prompt=data.get("system_prompt"),
        metadata=data.get("metadata", {}),
    )

    evaluator_type = data.get("evaluator", "exact_match")
    evaluator_config = data.get("evaluator_config", {})

    return task, evaluator_type, evaluator_config


def _get_evaluator(evaluator_type: str, config: dict[str, Any]) -> Any:
    """Get an evaluator instance by type name."""
    evaluators: dict[str, Any] = {
        # Accuracy evaluators
        "exact_match": lambda: ExactMatchEvaluator(**config),
        "fuzzy_match": lambda: FuzzyMatchEvaluator(**config),
        # Format evaluators
        "regex": lambda: RegexEvaluator(**config),
        "contains": lambda: ContainsEvaluator(**config),
        # Cheap proxy metrics (MVP 2)
        "json_parse": lambda: JsonParseEvaluator(**config),
        "json_schema": lambda: JsonSchemaEvaluator(**config),
        "field_coverage": lambda: FieldCoverageEvaluator(**config),
        "keyword_presence": lambda: KeywordPresenceEvaluator(**config),
        "length_constraint": lambda: LengthConstraintEvaluator(**config),
        "output_shape": lambda: OutputShapeEvaluator(**config),
    }

    if evaluator_type not in evaluators:
        raise ValueError(
            f"Unknown evaluator type: {evaluator_type}. Available: {list(evaluators.keys())}"
        )

    return evaluators[evaluator_type]()


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
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-P",
        help="Runtime profile: slow-local, balanced, or throughput",
    ),
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        help="Optimization strategy override",
    ),
    max_generations: int | None = typer.Option(
        None,
        "--max-generations",
        "-g",
        help="Maximum number of generations override",
    ),
    population_size: int | None = typer.Option(
        None,
        "--population-size",
        "-p",
        help="Population size override",
    ),
    patience: int | None = typer.Option(
        None,
        "--patience",
        help="Generations without improvement before early stop",
    ),
    max_concurrency: int | None = typer.Option(
        None,
        "--max-concurrency",
        help="Maximum concurrent LLM requests override",
    ),
    runtime_budget: float | None = typer.Option(
        None,
        "--runtime-budget",
        help="Maximum runtime in seconds (0 for unlimited)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory override",
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
        promptfoundry optimize --task task.yaml --seed-prompt "Answer: {input}" --profile slow-local
    """
    # Load task
    try:
        task_obj, evaluator_type, evaluator_config = _load_task(task)
        console.print(f"[green]✓[/green] Loaded task: {task_obj.name}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load task: {e}")
        raise typer.Exit(1)

    config_data: dict[str, Any] = {}

    # Load config file if provided
    llm_config = LLMConfig()
    if config:
        try:
            with config.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            if "llm" in config_data:
                llm_config = LLMConfig.from_dict(config_data["llm"])
            console.print(f"[green]✓[/green] Loaded config: {config.name}")
        except Exception as e:
            console.print(f"[yellow]![/yellow] Failed to load config: {e}, using defaults")
            config_data = {}

    # Build RuntimeConfig with proper precedence:
    # 1. Start with profile (CLI or config file)
    # 2. Apply config file values
    # 3. Apply CLI overrides
    effective_profile = profile or config_data.get("optimization", {}).get("profile", "balanced")

    try:
        runtime_config = RuntimeConfig.from_profile(effective_profile)
        if verbose:
            console.print(f"[dim]Using profile: {effective_profile}[/dim]")
    except ValueError as e:
        console.print(f"[yellow]![/yellow] {e}, using 'balanced' profile")
        runtime_config = RuntimeConfig.from_profile("balanced")

    # Apply config file values
    if config_data:
        runtime_config = RuntimeConfig.from_dict(config_data).with_overrides(
            profile=runtime_config.profile  # Keep the profile
        )

    # Apply CLI overrides (highest priority)
    runtime_config = runtime_config.with_overrides(
        max_generations=max_generations,
        population_size=population_size,
        patience=patience,
        max_concurrency=max_concurrency,
        runtime_budget_seconds=runtime_budget,
    )

    llm_config = _apply_runtime_llm_overrides(
        llm_config,
        runtime_config,
        config_data.get("llm", {}),
    )

    strategy_settings = config_data.get("strategy", {}).get("evolutionary", {})
    output_settings = config_data.get("output", {})
    effective_strategy = strategy or config_data.get("optimization", {}).get(
        "strategy", "evolutionary"
    )
    effective_output_dir = output_dir or Path(output_settings.get("directory", "./output"))

    console.print(
        Panel(
            f"[bold blue]PromptFoundry[/bold blue] - Prompt Optimization\n"
            f"Task: {task.name}\n"
            f"Profile: {runtime_config.profile.value}\n"
            f"Strategy: {effective_strategy}\n"
            f"Generations: {runtime_config.max_generations}\n"
            f"Population: {runtime_config.population_size}\n"
            f"Max Concurrency: {runtime_config.max_concurrency}"
            + (
                f"\nRuntime Budget: {runtime_config.runtime_budget_seconds}s"
                if runtime_config.runtime_budget_seconds > 0
                else ""
            ),
            title="Starting Optimization",
        )
    )

    # Validate strategy
    if effective_strategy != "evolutionary":
        console.print(
            f"[yellow]![/yellow] Strategy '{effective_strategy}' not yet implemented, using 'evolutionary'"
        )
        effective_strategy = "evolutionary"

    # Create components
    evaluator = _get_evaluator(evaluator_type, evaluator_config)
    console.print(f"[green]✓[/green] Using evaluator: {evaluator_type}")

    strategy_config = EvolutionaryConfig(
        population_size=runtime_config.population_size,
        max_generations=runtime_config.max_generations,
        patience=runtime_config.patience,
        seed=runtime_config.seed,
        mutation_rate=strategy_settings.get("mutation_rate", 0.3),
        crossover_rate=strategy_settings.get("crossover_rate", 0.7),
        tournament_size=strategy_settings.get("tournament_size", 3),
        elitism=strategy_settings.get("elitism", 2),
        adaptive_mutation_weights=strategy_settings.get("adaptive_mutation_weights", True),
        min_operator_weight=strategy_settings.get("min_operator_weight", 0.4),
        weight_learning_rate=strategy_settings.get("weight_learning_rate", 0.8),
        use_semantic_mutations=strategy_settings.get("use_semantic_mutations", True),
        use_diversity_control=strategy_settings.get("use_diversity_control", True),
        use_adaptive_schedule=strategy_settings.get("use_adaptive_schedule", False),
        schedule_type=strategy_settings.get("schedule_type", "adaptive"),
        enable_ablation_tracking=strategy_settings.get("enable_ablation_tracking", True),
        min_diversity_ratio=strategy_settings.get("min_diversity_ratio", 0.7),
        crowding_penalty=strategy_settings.get("crowding_penalty", 0.1),
    )
    optimization_strategy = GeneticAlgorithmStrategy(strategy_config)
    console.print("[green]✓[/green] Using strategy: evolutionary")

    # Create optimizer config from runtime config
    optimizer_config = OptimizerConfig.from_runtime_config(runtime_config)

    # Ensure output directory exists
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # Create seed prompt
    seed = Prompt(text=seed_prompt)
    console.print("[green]✓[/green] Created seed prompt\n")

    # Progress tracking
    progress_data: dict[str, Any] = {"generation": 0, "best_fitness": 0.0, "last_time": time.time()}

    # Run optimization
    async def _run_optimization() -> OptimizationResult:
        llm_client = OpenAICompatClient(llm_config)

        # Check LLM health
        with console.status("[bold green]Checking LLM connection..."):
            healthy = await llm_client.health_check()

        if not healthy:
            console.print(f"[red]✗[/red] Cannot connect to LLM at {llm_config.base_url}")
            console.print("  Make sure your LLM server is running.")
            await llm_client.close()
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Connected to LLM at {llm_config.base_url}\n")

        optimizer = Optimizer(
            strategy=optimization_strategy,
            evaluator=evaluator,
            llm_client=llm_client,
            config=optimizer_config,
        )

        # Add progress callback (using type: ignore due to Protocol matching quirk)
        def _progress_callback(
            generation: int,
            best_fitness: float,
            avg_fitness: float,
            best_prompt_text: str,
        ) -> None:
            now = time.time()
            last = progress_data.get("last_time", now)
            delta = now - last
            progress_data["generation"] = generation
            progress_data["best_fitness"] = best_fitness
            progress_data["last_time"] = now
            if verbose:
                console.print(
                    f"  Gen {generation}: best={best_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}  (took {delta:.1f}s)"
                )

        optimizer.add_callback(_progress_callback)  # type: ignore[arg-type]

        console.print("[bold]Running optimization...[/bold]\n")

        # Use rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress_task = progress.add_task(
                "[cyan]Optimizing prompts...",
                total=runtime_config.max_generations,
            )

            # Wrap callback to update progress bar
            original_callbacks = optimizer._callbacks.copy()
            optimizer._callbacks.clear()

            def _wrapped_callback(
                generation: int,
                best_fitness: float,
                avg_fitness: float,
                best_prompt_text: str,
            ) -> None:
                for cb in original_callbacks:
                    cb(generation, best_fitness, avg_fitness, best_prompt_text)
                progress.update(progress_task, completed=generation + 1)

            optimizer.add_callback(_wrapped_callback)  # type: ignore[arg-type]

            try:
                result = await optimizer.optimize(
                    seed_prompt=seed,
                    task=task_obj,
                )
            finally:
                # ensure client is closed even if cancelled or error occurs
                await llm_client.close()

        return result

    try:
        result = asyncio.run(_run_optimization())
        best_prompt = result.best_prompt
        best_fitness = result.best_score
        total_generations = result.total_generations
    except KeyboardInterrupt:
        console.print("\n[red]✗[/red] Optimization cancelled by user.")
        # We may or may not have a partial result; nothing to print further.
        raise typer.Exit(1)
    except typer.Exit:
        # a controlled exit was already signalled (e.g. health check failed);
        # simply propagate to avoid extra error messages or traceback.
        raise
    except Exception as e:
        console.print(f"\n[red]✗[/red] Optimization failed: {e}")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)

    # Display results
    console.print("\n" + "=" * 60)
    console.print(
        Panel(
            f"[bold green]Optimization Complete![/bold green]\n\n"
            f"[cyan]Best Fitness:[/cyan] {best_fitness:.4f}\n"
            f"[cyan]Generations:[/cyan] {total_generations}\n\n"
            f"[cyan]Best Prompt:[/cyan]\n{best_prompt.text}",
            title="Results",
        )
    )

    # Save results
    result_data = {
        "task": task_obj.name,
        "seed_prompt": seed_prompt,
        "strategy": effective_strategy,
        "profile": runtime_config.profile.value,
        "population_size": runtime_config.population_size,
        "max_concurrency": runtime_config.max_concurrency,
        "timestamp": datetime.now().isoformat(),
        "best_prompt": best_prompt.text,
        "best_fitness": best_fitness,
        "generations": total_generations,
        "evolutionary_config": {
            "mutation_rate": strategy_config.mutation_rate,
            "crossover_rate": strategy_config.crossover_rate,
            "tournament_size": strategy_config.tournament_size,
            "elitism": strategy_config.elitism,
            "adaptive_mutation_weights": strategy_config.adaptive_mutation_weights,
            "use_semantic_mutations": strategy_config.use_semantic_mutations,
            "use_diversity_control": strategy_config.use_diversity_control,
            "use_adaptive_schedule": strategy_config.use_adaptive_schedule,
            "schedule_type": strategy_config.schedule_type,
            "enable_ablation_tracking": strategy_config.enable_ablation_tracking,
            "min_diversity_ratio": strategy_config.min_diversity_ratio,
            "crowding_penalty": strategy_config.crowding_penalty,
        },
        "detected_task_type": optimization_strategy.get_detected_task_type(),
        "detected_output_mode": optimization_strategy.get_detected_output_mode(),
        "diversity_metrics": optimization_strategy.get_diversity_metrics(),
        "schedule_state": optimization_strategy.get_schedule_state(),
        "ablation_result": optimization_strategy.get_ablation_result(),
        "ablation_summary": optimization_strategy.get_ablation_summary(),
        **result.to_dict(),
    }

    result_file = effective_output_dir / f"optimization_{datetime.now():%Y%m%d_%H%M%S}.json"

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to: {result_file}")


@app.command()
def validate(
    config_path: Path = typer.Option(
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
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        console.print("[green]✓[/green] Configuration file is valid YAML")

        # Check required sections
        required_sections = ["llm"]
        for section in required_sections:
            if section in data:
                console.print(f"[green]✓[/green] Found '{section}' section")
            else:
                console.print(f"[yellow]![/yellow] Missing '{section}' section (will use defaults)")

        # Validate LLM config if present
        if "llm" in data:
            try:
                LLMConfig.from_dict(data["llm"])
                console.print("[green]✓[/green] LLM configuration is valid")
            except Exception as e:
                console.print(f"[red]✗[/red] Invalid LLM config: {e}")
                raise typer.Exit(1)

    except yaml.YAMLError as e:
        console.print(f"[red]✗[/red] Invalid YAML: {e}")
        raise typer.Exit(1)


@app.command()
def report(
    result_file: Path = typer.Argument(
        ...,
        help="Path to optimization result JSON file",
        exists=True,
        readable=True,
    ),
) -> None:
    """View an optimization result report.

    Example:
        promptfoundry report ./output/optimization_20240101_120000.json
    """
    try:
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]✗[/red] Invalid JSON file: {e}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold blue]Optimization Report[/bold blue]\n\n"
            f"[cyan]Task:[/cyan] {data.get('task', 'N/A')}\n"
            f"[cyan]Timestamp:[/cyan] {data.get('timestamp', 'N/A')}\n"
            f"[cyan]Strategy:[/cyan] {data.get('strategy', 'N/A')}\n"
            f"[cyan]Population Size:[/cyan] {data.get('population_size', 'N/A')}\n"
            f"[cyan]Generations:[/cyan] {data.get('generations', 'N/A')}\n"
            f"[cyan]Best Fitness:[/cyan] {data.get('best_fitness', 0):.4f}",
            title=f"Report: {result_file.name}",
        )
    )

    # Show prompts
    table = Table(title="Prompts")
    table.add_column("Type", style="cyan")
    table.add_column("Template")

    table.add_row("Seed Prompt", data.get("seed_prompt", "N/A"))
    table.add_row("Best Prompt", data.get("best_prompt", "N/A"))

    console.print(table)

    # Show improvement
    seed = data.get("seed_prompt", "")
    best = data.get("best_prompt", "")
    if seed and best and seed != best:
        console.print("\n[green]✓[/green] Prompt was improved from seed")
    elif seed == best:
        console.print("\n[yellow]![/yellow] Best prompt is same as seed (no improvement)")


@app.command()
def list_results(
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output",
        "-o",
        help="Output directory to scan for results",
    ),
) -> None:
    """List all optimization results in output directory.

    Example:
        promptfoundry list-results --output ./output
    """
    if not output_dir.exists():
        console.print(f"[yellow]![/yellow] Directory does not exist: {output_dir}")
        return

    results = list(output_dir.glob("optimization_*.json"))

    if not results:
        console.print(f"[yellow]![/yellow] No optimization results found in {output_dir}")
        return

    table = Table(title=f"Optimization Results in {output_dir}")
    table.add_column("File", style="cyan")
    table.add_column("Task")
    table.add_column("Fitness", justify="right")
    table.add_column("Generations", justify="right")
    table.add_column("Timestamp")

    for result_file in sorted(results, reverse=True):
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            table.add_row(
                result_file.name,
                data.get("task", "N/A"),
                f"{data.get('best_fitness', 0):.4f}",
                str(data.get("generations", "N/A")),
                data.get("timestamp", "N/A")[:19] if data.get("timestamp") else "N/A",
            )
        except Exception:
            table.add_row(result_file.name, "[red]Error[/red]", "-", "-", "-")

    console.print(table)


@app.command()
def diagnose(
    result_file: Path = typer.Argument(
        ...,
        help="Path to optimization result JSON file",
        exists=True,
        readable=True,
    ),
) -> None:
    """Show detailed diagnostics for an optimization run.

    Analyzes the optimization results and reports:
    - Improvement metrics and status (success/no-signal/partial)
    - Termination reason
    - Per-generation timing and latency
    - Cache statistics
    - Warnings about potential issues

    Example:
        promptfoundry diagnose ./output/optimization_20240101_120000.json
    """
    try:
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]✗[/red] Invalid JSON file: {e}")
        raise typer.Exit(1)

    # Analyze the run
    diag = RunDiagnostics.analyze(
        history_data=data,
        termination_reason=data.get("termination_reason", "unknown"),
        elapsed_time=data.get("elapsed_time", 0.0),
        total_llm_calls=data.get("total_llm_calls", 0),
        total_cache_hits=data.get("total_cache_hits", 0),
    )

    # Format and print the report
    report = format_diagnostics_report(diag)
    console.print(report)

    # Summary panel with color-coded status
    status_color = {
        "success": "green",
        "no_signal": "yellow",
        "partial": "yellow",
        "failed": "red",
    }.get(diag.status.value, "white")

    console.print(f"\n[{status_color}]Status: {diag.status.value.upper()}[/{status_color}]")

    # Show warnings prominently
    if diag.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in diag.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning}")


@app.command()
def benchmark_summary(
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output",
        "-o",
        help="Output directory containing results",
    ),
) -> None:
    """Generate a summary report across multiple optimization runs.

    Aggregates statistics from all optimization results in the directory.

    Example:
        promptfoundry benchmark-summary --output ./output
    """
    if not output_dir.exists():
        console.print(f"[yellow]![/yellow] Directory does not exist: {output_dir}")
        raise typer.Exit(1)

    results = list(output_dir.glob("optimization_*.json"))
    if not results:
        console.print(f"[yellow]![/yellow] No optimization results found in {output_dir}")
        raise typer.Exit(1)

    summary = BenchmarkSummary()

    for result_file in results:
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            diag = RunDiagnostics.analyze(
                history_data=data,
                termination_reason=data.get("termination_reason", "unknown"),
                elapsed_time=data.get("elapsed_time", 0.0),
                total_llm_calls=data.get("total_llm_calls", 0),
                total_cache_hits=data.get("total_cache_hits", 0),
            )
            summary.add_run(diag)
        except Exception as e:
            console.print(f"[yellow]![/yellow] Failed to parse {result_file.name}: {e}")

    if summary.total_runs == 0:
        console.print("[yellow]![/yellow] No valid results to summarize")
        raise typer.Exit(1)

    # Print summary
    console.print(Panel("[bold blue]Benchmark Summary[/bold blue]", expand=False))

    # Overview table
    overview = Table(title="Overview")
    overview.add_column("Metric", style="cyan")
    overview.add_column("Value", justify="right")

    overview.add_row("Total Runs", str(summary.total_runs))
    overview.add_row("Successful", f"[green]{summary.successful_runs}[/green]")
    overview.add_row("No Signal", f"[yellow]{summary.no_signal_runs}[/yellow]")
    overview.add_row("Avg Improvement", f"{summary.average_improvement:.4f}")
    overview.add_row("Avg Runtime", f"{summary.average_runtime:.2f}s")

    console.print(overview)

    # Per-task breakdown
    task_stats = summary.task_stats()
    if task_stats:
        task_table = Table(title="Per-Task Statistics")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Runs", justify="right")
        task_table.add_column("Success Rate", justify="right")
        task_table.add_column("Avg Improve", justify="right")
        task_table.add_column("Max Improve", justify="right")
        task_table.add_column("Avg Time", justify="right")

        for task, stats in task_stats.items():
            task_table.add_row(
                task,
                f"{stats['num_runs']:.0f}",
                f"{stats['success_rate']*100:.0f}%",
                f"{stats['avg_improvement']:.4f}",
                f"{stats['max_improvement']:.4f}",
                f"{stats['avg_runtime_s']:.1f}s",
            )

        console.print(task_table)


@app.command()
def gate_check(
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output",
        "-o",
        help="Output directory containing results",
    ),
    min_improvement: float = typer.Option(
        0.05,
        "--min-improvement",
        help="Minimum required improvement (absolute)",
    ),
    min_success_rate: float = typer.Option(
        0.6,
        "--min-success-rate",
        help="Minimum required success rate (0.0-1.0)",
    ),
    max_no_signal: float = typer.Option(
        0.3,
        "--max-no-signal",
        help="Maximum allowed no-signal rate (0.0-1.0)",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with error code if gate fails",
    ),
) -> None:
    """Run benchmark gate validation on optimization results.

    Validates that optimization runs meet quality thresholds:
    - Minimum improvement threshold
    - Minimum success rate across runs
    - Maximum no-signal rate tolerance

    Example:
        promptfoundry gate-check --output ./output
        promptfoundry gate-check --min-improvement 0.1 --strict
    """
    if not output_dir.exists():
        console.print(f"[yellow]![/yellow] Directory does not exist: {output_dir}")
        raise typer.Exit(1)

    results = list(output_dir.glob("optimization_*.json"))
    if not results:
        console.print(f"[yellow]![/yellow] No optimization results found in {output_dir}")
        raise typer.Exit(1)

    # Create gate with custom thresholds
    gate = create_custom_gate(
        min_improvement=min_improvement,
        min_success_rate=min_success_rate,
        max_no_signal_rate=max_no_signal,
    )

    # Collect diagnostics from all results
    diagnostics: list[RunDiagnostics] = []

    for result_file in results:
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            diag = RunDiagnostics.analyze(
                history_data=data,
                termination_reason=data.get("termination_reason", "unknown"),
                elapsed_time=data.get("elapsed_time", 0.0),
                total_llm_calls=data.get("total_llm_calls", 0),
                total_cache_hits=data.get("total_cache_hits", 0),
            )
            diagnostics.append(diag)
        except Exception as e:
            console.print(f"[yellow]![/yellow] Failed to parse {result_file.name}: {e}")

    if not diagnostics:
        console.print("[yellow]![/yellow] No valid results to check")
        raise typer.Exit(1)

    # Run gate check
    result = gate.check_results(diagnostics)

    # Display thresholds
    console.print(
        Panel(
            f"[bold blue]Benchmark Gate Check[/bold blue]\n\n"
            f"Min Improvement: {min_improvement:.2%}\n"
            f"Min Success Rate: {min_success_rate:.0%}\n"
            f"Max No-Signal Rate: {max_no_signal:.0%}",
            title="Thresholds",
        )
    )

    # Display gate result
    report = gate.format_report(result)
    console.print(report)

    # Summary table
    if result.summary:
        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Runs", str(result.summary.get("total_runs", 0)))
        summary_table.add_row("Successful Runs", str(result.summary.get("successful_runs", 0)))
        summary_table.add_row("Tasks Checked", str(result.summary.get("tasks_checked", 0)))
        summary_table.add_row("Tasks Passed", str(result.summary.get("tasks_passed", 0)))

        console.print(summary_table)

    # Per-task results
    if result.task_results:
        task_table = Table(title="Per-Task Results")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Passed", justify="center")
        task_table.add_column("Improvement", justify="right")
        task_table.add_column("Details")

        for task_name, task_result in result.task_results.items():
            passed = task_result.get("passed", False)
            improvement = task_result.get("improvement", 0.0)
            details = task_result.get("status", "")

            passed_str = "[green]✓[/green]" if passed else "[red]✗[/red]"
            task_table.add_row(task_name, passed_str, f"{improvement:.4f}", str(details))

        console.print(task_table)

    # Final status
    if result.passed:
        console.print("\n[bold green]✓ GATE PASSED[/bold green]")
    else:
        console.print("\n[bold red]✗ GATE FAILED[/bold red]")
        if result.failures:
            console.print("\n[bold red]Failures:[/bold red]")
            for failure in result.failures:
                console.print(f"  [red]•[/red] {failure}")
        if strict:
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
    table.add_column("Type")

    # Accuracy evaluators
    table.add_row(
        "exact_match",
        "Exact string match (case-insensitive by default)",
        "[blue]Accuracy[/blue]",
    )
    table.add_row(
        "fuzzy_match",
        "Fuzzy string matching with Levenshtein distance",
        "[blue]Accuracy[/blue]",
    )

    # Format evaluators
    table.add_row(
        "regex",
        "Regex pattern matching",
        "[magenta]Format[/magenta]",
    )
    table.add_row(
        "contains",
        "Check if output contains expected text",
        "[magenta]Format[/magenta]",
    )

    # Cheap proxy metrics (MVP 2)
    table.add_row(
        "json_parse",
        "Validates JSON syntax (cheap pre-filter)",
        "[green]Proxy[/green]",
    )
    table.add_row(
        "json_schema",
        "JSON schema validation with partial credit",
        "[green]Proxy[/green]",
    )
    table.add_row(
        "field_coverage",
        "Check for required patterns/sections in output",
        "[green]Proxy[/green]",
    )
    table.add_row(
        "keyword_presence",
        "Check for required/forbidden keywords",
        "[green]Proxy[/green]",
    )
    table.add_row(
        "length_constraint",
        "Score based on output length constraints",
        "[green]Proxy[/green]",
    )
    table.add_row(
        "output_shape",
        "Validate structural shape (prefix, suffix, markers)",
        "[green]Proxy[/green]",
    )

    # Custom evaluators
    table.add_row(
        "custom",
        "Custom Python scoring function",
        "[yellow]Custom[/yellow]",
    )
    table.add_row(
        "composite",
        "Weighted combination of multiple evaluators",
        "[yellow]Custom[/yellow]",
    )

    console.print(table)
    console.print("\n[dim]Proxy evaluators provide partial credit and are ideal for staged pipelines.[/dim]")


@app.command()
def profiles(
    name: str | None = typer.Argument(
        None,
        help="Profile name to show details for",
    ),
) -> None:
    """List available runtime profiles or show details for a specific profile.

    Examples:
        promptfoundry profiles              # List all profiles
        promptfoundry profiles slow-local   # Show slow-local profile details
    """
    available = get_available_profiles()

    if name:
        # Show details for specific profile
        if name not in available:
            console.print(f"[red]✗[/red] Unknown profile: {name}")
            console.print(f"Available profiles: {', '.join(available)}")
            raise typer.Exit(1)

        config = RuntimeConfig.from_profile(name)
        console.print(
            Panel(
                f"[bold]{name}[/bold]\n\n"
                f"[dim]{get_profile_description(name)}[/dim]\n\n"
                f"{config.describe()}",
                title="Runtime Profile",
            )
        )
    else:
        # List all profiles
        table = Table(title="Available Runtime Profiles")
        table.add_column("Profile", style="cyan")
        table.add_column("Population", justify="right")
        table.add_column("Generations", justify="right")
        table.add_column("Concurrency", justify="right")
        table.add_column("Description")

        for profile_name in available:
            config = RuntimeConfig.from_profile(profile_name)
            desc = get_profile_description(profile_name)
            # Truncate description
            short_desc = desc[:50] + "..." if len(desc) > 50 else desc
            table.add_row(
                profile_name,
                str(config.population_size),
                str(config.max_generations),
                str(config.max_concurrency),
                short_desc,
            )

        console.print(table)
        console.print("\n[dim]Use 'promptfoundry profiles <name>' for details[/dim]")
        console.print("[dim]Use '--profile <name>' with optimize command[/dim]")


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()

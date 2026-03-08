#!/usr/bin/env python
"""Benchmark runner for PromptFoundry.

This script runs optimization benchmarks on example tasks and records
performance metrics for evaluation.

Usage:
    python scripts/run_benchmarks.py [--task TASK_FILE] [--generations N] [--population N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from promptfoundry.core import (
    Example,
    Optimizer,
    OptimizerConfig,
    Prompt,
    Task,
)
from promptfoundry.evaluators import (
    ExactMatchEvaluator,
    FuzzyMatchEvaluator,
    NumericAnswerEvaluator,
    RegexEvaluator,
)
from promptfoundry.llm import LLMConfig, OpenAICompatClient
from promptfoundry.strategies import GeneticAlgorithmStrategy
from promptfoundry.strategies.evolutionary import EvolutionaryConfig


def load_task(task_path: Path) -> tuple[Task, str, dict]:
    """Load a task from a YAML file."""
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


def get_evaluator(evaluator_type: str, config: dict):
    """Get an evaluator instance by type name."""
    evaluators = {
        "exact_match": lambda: ExactMatchEvaluator(**config),
        "fuzzy_match": lambda: FuzzyMatchEvaluator(**config),
        "numeric_answer": lambda: NumericAnswerEvaluator(**config),
        "regex": lambda: RegexEvaluator(**config),
    }

    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")

    return evaluators[evaluator_type]()


async def run_benchmark(
    task_path: Path,
    seed_prompt: str,
    max_generations: int = 10,
    population_size: int = 10,
    llm_url: str = "http://127.0.0.1:5000/v1",
) -> dict:
    """Run a single benchmark.
    
    Returns:
        Dictionary with benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"Running benchmark: {task_path.name}")
    print(f"{'='*60}")

    # Load task
    task, evaluator_type, evaluator_config = load_task(task_path)
    print(f"Task: {task.name} ({len(task.examples)} examples)")
    print(f"Evaluator: {evaluator_type}")

    # Create components
    llm_config = LLMConfig(base_url=llm_url)
    llm_client = OpenAICompatClient(llm_config)

    # Check LLM connection
    print("Checking LLM connection...")
    healthy = await llm_client.health_check()
    if not healthy:
        print(f"ERROR: Cannot connect to LLM at {llm_url}")
        await llm_client.close()
        return {"error": "LLM connection failed"}
    print("LLM connected!")

    evaluator = get_evaluator(evaluator_type, evaluator_config)
    strategy = GeneticAlgorithmStrategy(
        EvolutionaryConfig(population_size=population_size)
    )

    optimizer_config = OptimizerConfig(
        max_generations=max_generations,
        population_size=population_size,
        patience=5,
    )

    optimizer = Optimizer(
        strategy=strategy,
        evaluator=evaluator,
        llm_client=llm_client,
        config=optimizer_config,
    )

    seed = Prompt(text=seed_prompt)

    # Run optimization
    print(f"\nRunning optimization for {max_generations} generations...")
    start_time = time.time()

    result = await optimizer.optimize(
        seed_prompt=seed,
        task=task,
    )

    elapsed = time.time() - start_time

    await llm_client.close()

    # Collect results
    benchmark_result = {
        "task": task.name,
        "task_file": str(task_path),
        "seed_prompt": seed_prompt,
        "best_prompt": result.best_prompt.text,
        "best_score": result.best_score,
        "total_generations": result.total_generations,
        "total_evaluations": result.total_evaluations,
        "convergence_generation": result.convergence_generation,
        "elapsed_time": elapsed,
        "evaluator": evaluator_type,
        "population_size": population_size,
        "timestamp": datetime.now().isoformat(),
    }

    print("\nResults:")
    print(f"  Best score: {result.best_score:.4f}")
    print(f"  Generations: {result.total_generations}")
    print(f"  Evaluations: {result.total_evaluations}")
    print(f"  Converged at: gen {result.convergence_generation}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"\nBest prompt:\n  {result.best_prompt.text[:100]}...")

    return benchmark_result


async def run_all_benchmarks(
    task_dir: Path,
    seed_prompts: dict[str, str],
    output_dir: Path,
    **kwargs,
) -> None:
    """Run benchmarks on all task files in a directory."""
    task_files = list(task_dir.glob("*.yaml"))

    print(f"Found {len(task_files)} task files")

    results = []
    for task_file in task_files:
        task_name = task_file.stem
        seed = seed_prompts.get(task_name, "{input}")

        try:
            result = await run_benchmark(task_file, seed, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"ERROR running benchmark {task_file.name}: {e}")
            results.append({
                "task_file": str(task_file),
                "error": str(e),
            })

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark results saved to: {output_file}")
    print(f"{'='*60}")

    # Summary
    print("\nSummary:")
    for r in results:
        if "error" in r:
            print(f"  {r.get('task', 'Unknown')}: ERROR - {r['error']}")
        else:
            print(f"  {r['task']}: score={r['best_score']:.4f}, gens={r['total_generations']}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PromptFoundry benchmarks")
    parser.add_argument(
        "--task",
        type=Path,
        help="Run a specific task file (default: run all in examples/)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Maximum generations (default: 10)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=10,
        help="Population size (default: 10)",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://127.0.0.1:5000/v1",
        help="LLM API URL (default: http://127.0.0.1:5000/v1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks"),
        help="Output directory for results (default: benchmarks/)",
    )

    args = parser.parse_args()

    # Default seed prompts for each task
    seed_prompts = {
        "sentiment_task": "Classify the sentiment: {input}",
        "json_formatting_task": "Extract data as JSON: {input}",
        "arithmetic_task": "Solve: {input}",
    }

    if args.task:
        # Run single task
        asyncio.run(run_benchmark(
            args.task,
            seed_prompts.get(args.task.stem, "{input}"),
            max_generations=args.generations,
            population_size=args.population,
            llm_url=args.llm_url,
        ))
    else:
        # Run all tasks
        examples_dir = Path(__file__).parent.parent / "examples"
        asyncio.run(run_all_benchmarks(
            examples_dir,
            seed_prompts,
            args.output,
            max_generations=args.generations,
            population_size=args.population,
            llm_url=args.llm_url,
        ))


if __name__ == "__main__":
    main()

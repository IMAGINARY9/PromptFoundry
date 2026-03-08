# PromptFoundry Benchmarks

This directory contains benchmark results from optimization runs.

## Running Benchmarks

```bash
# Run all benchmarks (requires LLM server running)
python scripts/run_benchmarks.py

# Run with custom settings
python scripts/run_benchmarks.py --generations 20 --population 15

# Run a specific task
python scripts/run_benchmarks.py --task examples/sentiment_task.yaml

# Use a different LLM endpoint
python scripts/run_benchmarks.py --llm-url http://localhost:8080/v1
```

## Benchmark Tasks

1. **Sentiment Classification** (`sentiment_task.yaml`)
   - Classify text as positive/negative/neutral
   - Evaluator: exact_match (case-insensitive, strict single-label scoring)
   - 10 training + 3 validation examples

2. **JSON Formatting** (`json_formatting_task.yaml`)
   - Extract structured data as JSON
   - Evaluator: fuzzy_match (threshold: 0.85)
   - 10 examples

3. **Arithmetic Reasoning** (`arithmetic_task.yaml`)
   - Solve word math problems
   - Evaluator: numeric_answer with strict bare-number perfect scoring and prose partial credit
   - 12 examples

For the full bundled task catalog, real-world use cases, and evaluator contract notes, see [docs/TASKS.md](../docs/TASKS.md).

## Result Format

Benchmark results are saved as JSON with:
- task name, seed prompt, best prompt
- best_score, total_generations, total_evaluations
- convergence_generation, elapsed_time
- configuration used (population_size, evaluator)

## Baseline Results

*Run benchmarks with your LLM to establish baselines.*

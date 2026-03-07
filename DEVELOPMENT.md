# PromptFoundry Development Guide

> Development setup, conventions, and workflow for PromptFoundry.

**Status:** MVP 2 Complete | full test suite passing

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip or poetry
- Local LLM server (for integration testing)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/PromptFoundry.git
cd PromptFoundry

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install in development mode
pip install -e ".[dev]"
```

---

## Project Structure

```
src/promptfoundry/
├── core/           # Domain models and orchestration
│   ├── prompt.py       # Prompt, PromptTemplate
│   ├── task.py         # Example, Task
│   ├── population.py   # Individual, Population
│   ├── history.py      # OptimizationHistory, OptimizationResult
│   ├── optimizer.py    # Optimizer controller
│   └── protocols.py    # Protocol interfaces
├── strategies/     # Optimization strategies
│   ├── base.py         # BaseStrategy
│   └── evolutionary.py # GeneticAlgorithmStrategy
├── evaluators/     # Scoring functions
│   ├── accuracy.py     # ExactMatchEvaluator
│   ├── format.py       # FuzzyMatchEvaluator, RegexEvaluator
│   └── custom.py       # CustomFunctionEvaluator, CompositeEvaluator
├── llm/            # LLM clients
│   ├── config.py       # LLMConfig
│   ├── openai_compat.py # OpenAICompatClient
│   └── rate_limiter.py # TokenBucket, RateLimiter
└── cli.py          # Typer CLI
```

---

## Commands

### Development

```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/promptfoundry --cov-report=html

# Type checking
mypy src/promptfoundry --ignore-missing-imports

# Linting
ruff check src/promptfoundry

# Formatting
ruff format src/promptfoundry
```

### CLI

```bash
# Run CLI
python -m promptfoundry --help

# Test LLM connection
python -m promptfoundry test-llm

# Run optimization
python -m promptfoundry optimize --task examples/sentiment_task.yaml --seed-prompt "Classify: {input}"
```

---

## Code Quality

### Type Annotations

- **Required** on all public functions and methods
- Use `typing.Protocol` for interfaces
- Frozen dataclasses for domain objects

### Testing

- Unit tests in `tests/`
- Use pytest fixtures from `conftest.py`
- Mock external dependencies (LLM, HTTP)

### Documentation

- Docstrings on all public classes/methods
- Update [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for progress
- Update [README.md](README.md) for user-facing changes

---

## Git Workflow

```bash
# Commit convention
git commit -m "feat(core): add new feature"
git commit -m "fix(evaluators): correct fuzzy match"
git commit -m "docs: update architecture"
git commit -m "test: add optimizer tests"
```

---

## Extension Points

When extending PromptFoundry:

1. **New optimization algorithm** → Implement `OptimizationStrategy` protocol
2. **New LLM backend** → Implement `LLMClient` protocol  
3. **New evaluator** → Implement `Evaluator` protocol
4. **New mutation** → Implement `MutationOperator` protocol

All protocols defined in `src/promptfoundry/core/protocols.py`.

---

## MVP Status

### ✅ MVP 1: CLI Optimizer — COMPLETE

- Core domain models
- Evolutionary optimization
- OpenAI-compatible client with rate limiting
- Multiple evaluators
- CLI with optimize/validate/report commands
- 3 benchmark tasks

### 🔄 MVP 2: Extended Search & Reporting — NEXT

- Bayesian optimization (Optuna)
- Grid search
- JSON schema evaluator
- Visualization
- Python library API

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for full roadmap.

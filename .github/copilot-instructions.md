# PromptFoundry Copilot Instructions

> Single source of truth for AI-assisted development. Reference authoritative docs, don't duplicate.

## Project Overview

PromptFoundry is an optimization-driven prompt engineering tool that treats prompt engineering as a systematic optimization problem, using evolutionary algorithms and local LLMs.

**Tech Stack:** Python 3.10+, Pydantic 2.x, httpx, Typer CLI, pytest

## Authoritative Documentation

| Topic | Source | Purpose |
|-------|--------|---------|
| Requirements | [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | FR-xxx, NFR-xxx specifications |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered design, domain models, data flow |
| Implementation | [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | MVP phases, milestones |
| Documentation Standards | [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) | Code quality, hygiene rules |
| Configuration | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | YAML config schema |

## Key Architecture Decisions

### Design Patterns
- **Strategy Pattern**: `OptimizationStrategy` protocol → swap algorithms (evolutionary, Bayesian, grid search)
- **Adapter Pattern**: `LLMClient` protocol → backends (local, OpenAI, Anthropic)
- **Protocol-based interfaces**: Loose coupling via `typing.Protocol` in `core/protocols.py`

### Package Structure
```
src/promptfoundry/
├── core/         # Domain models: Prompt, Task, Population, History
├── strategies/   # Optimization algorithms (evolutionary.py)
├── evaluators/   # Scoring (accuracy.py, format.py)
├── llm/          # LLM backends (openai_compat.py)
└── cli.py        # Typer CLI
```

### Immutability
- `Prompt` is frozen dataclass with UUID-based identity
- Mutations create new instances, never modify existing

## Local LLM Configuration

See [docs/CONFIGURATION.md#6-local-llm-setup](../docs/CONFIGURATION.md) for full setup.

**Quick start:**
```python
from promptfoundry.llm import LLMConfig, OpenAICompatClient

config = LLMConfig.for_local_model()  # defaults to localhost:5000
client = OpenAICompatClient(config)
```

**Test:** `python scripts/test_llm_connection.py`

## Development Workflow

### Commands
```bash
# Activate venv
.\.venv\Scripts\Activate.ps1

# Run tests
pytest -v

# Type check (strict mode)
mypy src/promptfoundry --ignore-missing-imports

# Lint + auto-fix
ruff check src/promptfoundry --fix

# Format
ruff format src/promptfoundry
```

### Adding New Strategy
1. Create `src/promptfoundry/strategies/{name}.py`
2. Implement `OptimizationStrategy` protocol from `core/protocols.py`
3. Register in `strategies/__init__.py`
4. Add tests in `tests/test_strategies.py`

### Adding New Evaluator
1. Create evaluator class implementing `Evaluator` protocol
2. Add to `evaluators/__init__.py`
3. Usage: `evaluate(predicted, expected, metadata)`

## Code Quality Rules

1. **Type annotations required** on all public functions
2. **Docstrings required** on all public classes/methods
3. **No `Any` types** in return positions (use explicit types)
4. **Protocol over ABC** for interfaces
5. **Frozen dataclasses** for domain objects

## Current Implementation Status

### Completed (Phase 1.1)
- ✅ Domain models (Prompt, Task, Population, History)
- ✅ GeneticAlgorithmStrategy with mutation/crossover operators
- ✅ ExactMatch, FuzzyMatch, Regex, Contains evaluators  
- ✅ OpenAI-compatible LLM client with retry logic
- ✅ Typer CLI scaffold

### Next Steps (Phase 1.2+)
- [ ] Full CLI integration (optimize command)
- [ ] Prompt serialization/checkpointing
- [ ] Configuration file loading
- [ ] Batch evaluation pipeline

## Testing Fixtures

Key fixtures in `tests/conftest.py`:
- `sample_prompt` - Basic prompt for testing
- `sample_task` - Task with 5 examples
- `sample_population` - Population of 5 individuals
- `MockLLMClient` - Deterministic LLM mock for testing

## Git Workflow

```bash
# Remote
git remote -v  # origin: git@github.com:IMAGINARY9/PromptFoundry.git

# Commit convention
git commit -m "feat(core): add new feature"
git commit -m "fix(evaluators): correct fuzzy match"
git commit -m "docs: update architecture"
```

## Extension Points

When extending PromptFoundry:
1. **New optimization algorithm** → Implement `OptimizationStrategy` protocol
2. **New LLM backend** → Implement `LLMClient` protocol  
3. **New evaluator** → Implement `Evaluator` protocol
4. **New mutation** → Implement `MutationOperator` protocol

All protocols defined in `src/promptfoundry/core/protocols.py`.

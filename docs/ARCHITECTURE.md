# PromptFoundry — Architecture Document

> **Version:** 1.0.0  
> **Status:** Active  
> **Last Updated:** 2026-03-06  
> **Authoritative Source:** This document is the single source of truth for system architecture.

---

## 1. Overview

PromptFoundry follows a modular, layered architecture designed for extensibility and testability. The system is built around the Strategy pattern for optimization algorithms and the Adapter pattern for LLM backends.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│                   (CLI / Python API / Web UI)                   │
├─────────────────────────────────────────────────────────────────┤
│                      Orchestration Layer                         │
│                    (Optimizer Controller)                        │
├───────────────┬──────────────────┬──────────────────────────────┤
│   Strategies  │    Evaluators    │         LLM Adapters         │
│  (Evolutionary│   (Objective     │  (OpenAI, Local, Custom)     │
│   Bayesian,   │    Functions)    │                              │
│   Grid)       │                  │                              │
├───────────────┴──────────────────┴──────────────────────────────┤
│                         Core Domain                              │
│            (Prompt, Task, Population, History)                   │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure                              │
│         (Config, Logging, Persistence, HTTP Client)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Design Principles

### 2.1 Loose Coupling
- Components communicate through well-defined interfaces (protocols/ABCs).
- Strategies, evaluators, and LLM adapters are pluggable.
- No direct dependencies between horizontal components.

### 2.2 Single Responsibility
- Each module handles one concern.
- Optimization logic is separate from evaluation logic.
- LLM communication is isolated in adapters.

### 2.3 Dependency Inversion
- High-level modules depend on abstractions.
- Concrete implementations are injected at runtime.
- Configuration drives component selection.

### 2.4 Testability First
- All components accept dependencies via constructor injection.
- Pure functions where possible.
- Side effects isolated to adapter boundaries.

---

## 3. Package Structure

```
src/promptfoundry/
├── __init__.py              # Public API exports
├── cli.py                   # CLI entry point (typer)
├── core/                    # Domain models and interfaces
│   ├── __init__.py
│   ├── prompt.py            # Prompt, PromptTemplate, PromptVariant
│   ├── task.py              # Task, Example, TaskDataset
│   ├── population.py        # Population, Individual
│   ├── history.py           # OptimizationHistory, Generation
│   └── protocols.py         # Abstract interfaces (Strategy, Evaluator, LLMClient)
├── strategies/              # Optimization algorithms
│   ├── __init__.py
│   ├── base.py              # BaseStrategy abstract class
│   ├── evolutionary.py      # GeneticAlgorithmStrategy
│   ├── bayesian.py          # BayesianStrategy (MVP 2)
│   └── grid.py              # GridSearchStrategy (MVP 2)
├── evaluators/              # Objective functions
│   ├── __init__.py
│   ├── base.py              # BaseEvaluator abstract class
│   ├── accuracy.py          # ExactMatch, FuzzyMatch
│   ├── format.py            # JSONSchemaEvaluator, RegexEvaluator
│   └── composite.py         # WeightedComposite
├── llm/                     # LLM backend adapters
│   ├── __init__.py
│   ├── base.py              # BaseLLMClient abstract class
│   ├── openai_compat.py     # OpenAI-compatible client
│   └── config.py            # LLM configuration models
├── operators/               # Genetic operators
│   ├── __init__.py
│   ├── mutation.py          # Mutation operators
│   ├── crossover.py         # Crossover operators
│   └── selection.py         # Selection operators
├── reporting/               # Output generation
│   ├── __init__.py
│   ├── console.py           # Rich console output
│   ├── export.py            # JSON/CSV export
│   └── visualization.py     # Charts (MVP 2)
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── schema.py            # Pydantic config models
│   └── loader.py            # YAML/env loading
└── utils/                   # Shared utilities
    ├── __init__.py
    ├── logging.py           # Structured logging
    └── retry.py             # Retry/backoff utilities
```

---

## 4. Core Domain Models

### 4.1 Prompt

```python
@dataclass
class Prompt:
    """Immutable prompt representation."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
@dataclass
class PromptTemplate:
    """Prompt with variable placeholders."""
    template: str
    variables: list[str]
    
    def render(self, **kwargs) -> Prompt:
        """Render template with variables."""
        ...
```

### 4.2 Task

```python
@dataclass
class Example:
    """Single input-output example."""
    input: str
    expected_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass  
class Task:
    """Optimization task definition."""
    name: str
    examples: list[Example]
    system_prompt: str | None = None
```

### 4.3 Population

```python
@dataclass
class Individual:
    """Single prompt in population."""
    prompt: Prompt
    fitness: float | None = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    
@dataclass
class Population:
    """Collection of individuals."""
    individuals: list[Individual]
    generation: int
```

---

## 5. Key Interfaces (Protocols)

### 5.1 Strategy Protocol

```python
class OptimizationStrategy(Protocol):
    """Interface for optimization algorithms."""
    
    def initialize(self, seed_prompt: Prompt, config: StrategyConfig) -> Population:
        """Create initial population."""
        ...
    
    def evolve(
        self, 
        population: Population, 
        fitness_scores: list[float]
    ) -> Population:
        """Generate next generation."""
        ...
    
    def should_terminate(self, history: OptimizationHistory) -> bool:
        """Check termination criteria."""
        ...
```

### 5.2 Evaluator Protocol

```python
class Evaluator(Protocol):
    """Interface for objective functions."""
    
    def evaluate(
        self, 
        predicted: str, 
        expected: str, 
        metadata: dict[str, Any] | None = None
    ) -> float:
        """Score a single prediction. Returns 0.0-1.0."""
        ...

    def evaluate_batch(
        self,
        predictions: list[str],
        expected: list[str]
    ) -> list[float]:
        """Score multiple predictions."""
        ...
```

### 5.3 LLM Client Protocol

```python
class LLMClient(Protocol):
    """Interface for LLM backends."""
    
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs
    ) -> str:
        """Generate completion."""
        ...
    
    async def complete_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        **kwargs
    ) -> list[str]:
        """Generate batch completions."""
        ...
```

---

## 6. Data Flow

### 6.1 Optimization Loop

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Initialize  │────▶│   Evaluate   │────▶│    Select    │
│  Population  │     │  Fitness     │     │   Parents    │
└──────────────┘     └──────────────┘     └──────────────┘
       ▲                                         │
       │                                         ▼
       │              ┌──────────────┐     ┌──────────────┐
       │              │   Terminate? │◀────│   Generate   │
       │              │              │     │  Offspring   │
       │              └──────────────┘     └──────────────┘
       │                    │ No
       └────────────────────┘
                            │ Yes
                            ▼
                    ┌──────────────┐
                    │   Report     │
                    │   Results    │
                    └──────────────┘
```

### 6.2 Evaluation Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Prompt  │───▶│  LLM    │───▶│  Parse  │───▶│  Score  │
│         │    │ Invoke  │    │ Output  │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │                                             │
     │         For each example                    │
     └─────────────────────────────────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  Aggregate  │
                  │   Fitness   │
                  └─────────────┘
```

---

## 7. Configuration Architecture

### 7.1 Configuration Hierarchy

```yaml
# config.yaml
optimization:
  strategy: evolutionary      # Strategy selection
  max_generations: 100
  population_size: 20
  
strategy:
  evolutionary:
    mutation_rate: 0.3
    crossover_rate: 0.7
    tournament_size: 3
    
llm:
  provider: openai_compat
  base_url: http://127.0.0.1:5000/v1
  model: Mistral-7B
  temperature: 0.7
  max_tokens: 256
  
evaluation:
  type: accuracy
  metric: exact_match
  
logging:
  level: INFO
  format: structured
```

### 7.2 Environment Override

```
PROMPTFOUNDRY_LLM__BASE_URL=http://localhost:5000/v1
PROMPTFOUNDRY_LLM__API_KEY=local
PROMPTFOUNDRY_OPTIMIZATION__MAX_GENERATIONS=50
```

---

## 8. Extension Points

### 8.1 Custom Strategy

```python
from promptfoundry.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    """Custom optimization strategy."""
    
    def evolve(self, population, fitness_scores):
        # Custom evolution logic
        ...

# Register via config or entry point
```

### 8.2 Custom Evaluator

```python
from promptfoundry.evaluators.base import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    """Custom objective function."""
    
    def evaluate(self, predicted, expected, metadata=None):
        # Custom scoring logic
        return score

# Register via config or entry point
```

### 8.3 Custom LLM Backend

```python
from promptfoundry.llm.base import BaseLLMClient

class MyLLMClient(BaseLLMClient):
    """Custom LLM backend."""
    
    async def complete(self, prompt, system_prompt=None, **kwargs):
        # Custom completion logic
        return response
```

---

## 9. Error Handling Strategy

### 9.1 Error Categories

| Category | Handling | Example |
|----------|----------|---------|
| Configuration | Fail fast with clear message | Invalid YAML |
| Validation | Fail fast, list all errors | Missing examples |
| LLM Transient | Retry with backoff | Rate limit, timeout |
| LLM Fatal | Log, skip individual, continue | Invalid API key |
| Optimization | Log warning, use fallback | Degenerate population |

### 9.2 Result Types

```python
@dataclass
class Success(Generic[T]):
    value: T
    
@dataclass
class Failure:
    error: str
    details: dict[str, Any] | None = None
    
Result = Success[T] | Failure
```

---

## 10. Testing Architecture

### 10.1 Test Layers

```
┌─────────────────────────────────────────┐
│            Integration Tests            │  ← Full optimization runs
├─────────────────────────────────────────┤
│           Component Tests               │  ← Strategy, Evaluator tests
├─────────────────────────────────────────┤
│              Unit Tests                 │  ← Pure function tests
└─────────────────────────────────────────┘
```

### 10.2 Test Fixtures

- Mock LLM client for deterministic tests
- Sample tasks with known optimal prompts
- Seed data for reproducible optimization runs

---

## 11. Deployment Considerations

### 11.1 CLI Distribution

```bash
pip install promptfoundry
promptfoundry optimize --task task.yaml --config config.yaml
```

### 11.2 Library Usage

```python
from promptfoundry import Optimizer, Task, EvolutionaryStrategy

optimizer = Optimizer(
    strategy=EvolutionaryStrategy(),
    llm_client=OpenAICompatClient(base_url="http://localhost:5000/v1")
)

result = optimizer.optimize(
    task=Task.from_file("task.yaml"),
    seed_prompt="Classify the sentiment: {input}"
)
```

---

## 12. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-06 | Initial | Document created |

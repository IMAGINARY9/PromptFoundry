# PromptFoundry — Documentation Standards & Hygiene Rules

> **Version:** 1.0.0  
> **Status:** Active  
> **Last Updated:** 2026-03-06  
> **Authoritative Source:** This document governs all project documentation practices.

---

## 1. Single Source of Truth Principle

**Every piece of information must have exactly one authoritative location.**

Duplication leads to inconsistency, confusion, and maintenance burden. When information must appear in multiple places, always reference the authoritative source rather than copying content.

### 1.1 Authoritative Sources Table

| Information Type | Authoritative Source | Never Duplicate In |
|------------------|----------------------|-------------------|
| System architecture | [docs/ARCHITECTURE.md](ARCHITECTURE.md) | README, inline comments |
| Functional requirements | [docs/REQUIREMENTS.md](REQUIREMENTS.md) | GitHub issues, code comments |
| Development roadmap | [docs/IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | README, meeting notes |
| API documentation | Docstrings in source code | External markdown |
| Configuration options | [docs/CONFIGURATION.md](CONFIGURATION.md) | README, inline comments |
| Changelog | [CHANGELOG.md](../CHANGELOG.md) | GitHub releases (auto-sync) |
| License | [LICENSE](../LICENSE) | README (link only) |
| Dependencies | [pyproject.toml](../pyproject.toml) | README, docs |
| Development setup | [CONTRIBUTING.md](../CONTRIBUTING.md) | README (link only) |
| Code standards | This document + [.copilot-instructions.md](../.copilot-instructions.md) | Scattered comments |
| LLM configuration | [config/llm.example.yaml](../config/llm.example.yaml) | Hardcoded values |

### 1.2 Reference Pattern

When you need to mention information owned elsewhere:

```markdown
<!-- CORRECT: Reference the source -->
For architecture details, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

<!-- INCORRECT: Duplicating information -->
The system uses a layered architecture with strategies, evaluators, and LLM adapters...
```

---

## 2. Documentation Hygiene Rules

### 2.1 Keep Documentation Current

| Rule | Enforcement |
|------|-------------|
| Update docs in the same PR as code changes | PR checklist item |
| Mark outdated sections with `> ⚠️ OUTDATED` | Code review |
| Remove obsolete content immediately | Don't comment out |
| Version all major documents | Header metadata |

### 2.2 Document Structure

Every documentation file must include:

```markdown
# Document Title

> **Version:** X.Y.Z
> **Status:** Active | Draft | Deprecated
> **Last Updated:** YYYY-MM-DD
> **Authoritative Source:** [What this document owns]

---

## 1. Overview
[Brief description]

## 2-N. Content Sections
[Organized content]

---

## N+1. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
```

### 2.3 Linking Rules

- **Internal links**: Use relative paths (`./docs/ARCHITECTURE.md`)
- **External links**: Use full URLs with descriptive text
- **Code references**: Use exact file paths and line numbers
- **Never use**: Bare URLs, ambiguous references like "see above"

### 2.4 Anti-Duplication Checklist

Before adding new content, verify:

- [ ] Does this information already exist elsewhere?
- [ ] Is this the right authoritative location for this content?
- [ ] Am I duplicating or referencing?
- [ ] Have I linked to the source if referencing?

---

## 3. Code Quality Requirements

### 3.1 Proactive Improvement Mandate

**Discovered problems must be fixed, not ignored.**

When working on any task, if you discover:
- Code that violates patterns/standards
- Architectural limitations requiring updates
- Technical debt affecting stability
- Missing tests or documentation

**You must either:**
1. Fix it in the current PR (if scope allows)
2. Create a detailed GitHub issue with `tech-debt` label
3. Add a `# TODO(issue-number): description` comment

### 3.2 Code Standards

#### Type Safety
```python
# REQUIRED: Full type annotations on all public APIs
def optimize(
    self,
    task: Task,
    seed_prompt: str | Prompt,
    config: OptimizationConfig | None = None,
) -> OptimizationResult:
    """Optimize a prompt for the given task."""
    ...

# REQUIRED: Use strict typing
from typing import TypeVar, Protocol, Generic

T = TypeVar("T", covariant=True)

class Result(Generic[T]):
    ...
```

#### Patterns & Abstractions

| Pattern | When to Use | Example |
|---------|-------------|---------|
| Strategy | Interchangeable algorithms | Optimization strategies |
| Adapter | External system integration | LLM clients |
| Factory | Complex object construction | Strategy/Evaluator creation |
| Protocol | Interface definitions | `Evaluator`, `LLMClient` protocols |
| Result types | Error handling | `Success[T] | Failure` |

#### Code Abstraction Levels

```python
# HIGH ABSTRACTION: Public API - stable, well-documented
class Optimizer:
    def optimize(self, task: Task, seed_prompt: str) -> OptimizationResult:
        ...

# MEDIUM ABSTRACTION: Internal components - clear interfaces
class GeneticAlgorithmStrategy(OptimizationStrategy):
    def evolve(self, population: Population, fitness: list[float]) -> Population:
        ...

# LOW ABSTRACTION: Implementation details - may change
def _tournament_select(population: list[Individual], k: int) -> Individual:
    ...
```

### 3.3 Architecture Principles

#### Loose Coupling
```python
# CORRECT: Depend on protocol, inject dependency
class Optimizer:
    def __init__(self, llm_client: LLMClient, strategy: OptimizationStrategy):
        self._llm = llm_client
        self._strategy = strategy

# INCORRECT: Hard dependency on concrete class
class Optimizer:
    def __init__(self):
        self._llm = OpenAIClient()  # Tight coupling!
```

#### Testability
```python
# CORRECT: Pure function, easy to test
def calculate_fitness(predictions: list[str], expected: list[str]) -> float:
    return sum(p == e for p, e in zip(predictions, expected)) / len(expected)

# CORRECT: Dependency injection for side effects
class Evaluator:
    def __init__(self, llm: LLMClient):  # Inject mock in tests
        self._llm = llm
```

#### Single Responsibility
```python
# CORRECT: Separate concerns
class PromptMutator:
    """Only handles prompt mutation."""
    def mutate(self, prompt: Prompt) -> Prompt: ...

class FitnessEvaluator:
    """Only handles fitness evaluation."""
    def evaluate(self, prompt: Prompt, task: Task) -> float: ...

# INCORRECT: Mixed responsibilities
class PromptHandler:
    def mutate_and_evaluate_and_log(self, prompt: Prompt) -> float: ...
```

---

## 4. Development Requirements

### 4.1 Mandatory Improvements

When potential problems are discovered during development:

| Situation | Required Action |
|-----------|-----------------|
| Code violates established patterns | Fix immediately or create issue |
| Missing type annotations | Add them |
| Untested code path discovered | Add test |
| Hardcoded values found | Extract to configuration |
| Duplicate logic found | Refactor to shared utility |
| Unclear code | Add documentation or refactor for clarity |

### 4.2 Optimal Data Structures

Always choose the appropriate data structure:

| Use Case | Recommended | Avoid |
|----------|-------------|-------|
| Ordered, mutable sequence | `list` | Multiple appends to tuple |
| Key-value lookup | `dict` | Linear search in list |
| Unique elements | `set` | List with manual deduplication |
| Immutable data | `dataclass(frozen=True)` | Dict for structured data |
| Type-safe configs | `pydantic.BaseModel` | Plain dict |
| Optional values | `T | None` with explicit handling | Sentinel values |

### 4.3 Error Handling Standards

```python
# CORRECT: Explicit result types
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass
class Success(Generic[T]):
    value: T

@dataclass
class Failure:
    error: str
    code: str | None = None

Result = Success[T] | Failure

# CORRECT: Specific exceptions with context
class LLMConnectionError(Exception):
    """Raised when LLM backend is unreachable."""
    def __init__(self, url: str, cause: Exception):
        super().__init__(f"Failed to connect to {url}")
        self.url = url
        self.cause = cause
```

---

## 5. Quality Gates

### 5.1 Pre-Commit Checks

All commits must pass:
- `ruff check .` — No lint errors
- `ruff format --check .` — Code formatted
- `mypy src/` — No type errors
- `pytest tests/ -q` — All tests pass

### 5.2 PR Requirements

- [ ] All quality gates pass
- [ ] Documentation updated (if applicable)
- [ ] Tests added/updated (if applicable)
- [ ] No new `# type: ignore` without justification
- [ ] Follows single source of truth principle

---

## 6. File Organization

```
PromptFoundry/
├── .copilot-instructions.md   # AI assistant context
├── .gitignore
├── CHANGELOG.md               # Version history
├── CONTRIBUTING.md            # Development setup
├── LICENSE
├── README.md                  # Project overview (links to docs)
├── pyproject.toml             # Dependencies, tooling
├── config/                    # Example configurations
├── docs/                      # Detailed documentation
│   ├── ARCHITECTURE.md        # System design
│   ├── CONFIGURATION.md       # Config reference
│   ├── DOCUMENTATION.md       # This file
│   ├── IMPLEMENTATION_PLAN.md # Roadmap
│   └── REQUIREMENTS.md        # Specifications
├── src/promptfoundry/         # Source code
├── tests/                     # Test suite
├── benchmarks/                # Performance benchmarks
├── examples/                  # Usage examples
└── scripts/                   # Development scripts
```

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-06 | Initial | Document created |

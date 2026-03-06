# PromptFoundry — Implementation Plan

> **Version:** 1.1.0  
> **Status:** MVP 1 Complete  
> **Last Updated:** 2026-03-07  
> **Authoritative Source:** This document is the single source of truth for development roadmap.

---

## 1. Overview

This document outlines the phased implementation plan for PromptFoundry. Development proceeds through multiple MVP versions, each delivering demonstrable value while building toward the complete system.

---

## 2. MVP Versions

### ✅ MVP 1: CLI Optimizer (Foundation) — COMPLETED
**Goal:** Working command-line tool with evolutionary optimization

**Timeline:** Week 1-3 (Completed)

**Deliverables:**
- ✅ Core domain models (Prompt, Task, Population, History)
- ✅ Evolutionary strategy with mutation/crossover/selection
- ✅ OpenAI-compatible LLM client with rate limiting
- ✅ Multiple evaluators (exact, fuzzy, regex, custom, composite)
- ✅ CLI interface with optimize, validate, report commands
- ✅ JSON output with results saving
- ✅ 3 benchmark tasks (sentiment, JSON formatting, arithmetic)

### MVP 2: Extended Search & Reporting
**Goal:** Multiple strategies, richer evaluation, better reports

**Timeline:** Week 4-6

**Deliverables:**
- Bayesian optimization strategy (Optuna)
- Grid search strategy
- JSON schema evaluator
- Performance visualization (matplotlib/plotly)
- Ablation analysis utilities
- Python library API (importable as package)
- Enhanced checkpoint/resume

### MVP 3: Web UI & Task Library
**Goal:** Visual interface, pre-built tasks

**Timeline:** Week 7-9

**Deliverables:**
- Gradio/Streamlit web interface
- Real-time optimization visualization
- Pre-built task templates
- Save/load configurations
- Budget-aware optimization
- Documentation website

### MVP 4: Advanced Features
**Goal:** Production-ready with advanced capabilities

**Timeline:** Week 10-12

**Deliverables:**
- Human-in-loop feedback
- Multi-stage prompt chains
- Domain-specific operators
- Plugin architecture
- Comprehensive benchmarks

---

## 3. MVP 1 Detailed Plan — COMPLETED

### Phase 1.1: Project Setup ✅
- [x] Create project structure
- [x] Set up pyproject.toml with dependencies
- [x] Configure development tools (ruff, mypy, pytest)
- [x] Initialize git repository
- [x] Create CI/CD pipeline skeleton
- [x] Write documentation structure

### Phase 1.2: Core Domain ✅
- [x] Implement `Prompt` and `PromptTemplate` models
- [x] Implement `Example` and `Task` models
- [x] Implement `Individual` and `Population` models
- [x] Implement `OptimizationHistory` model
- [x] Define protocol interfaces (Strategy, Evaluator, LLMClient)
- [x] Write unit tests for all models (26 tests)

### Phase 1.3: LLM Client ✅
- [x] Implement `OpenAICompatClient` with httpx
- [x] Add retry logic with exponential backoff
- [x] Add rate limiting (TokenBucket algorithm)
- [x] Configure for local text-generation-webui
- [x] Write integration tests with mock server (16 tests)

### Phase 1.4: Evaluators ✅
- [x] Implement `ExactMatchEvaluator`
- [x] Implement `RegexEvaluator`
- [x] Implement `FuzzyMatchEvaluator`
- [x] Implement `CustomFunctionEvaluator`
- [x] Implement `CompositeEvaluator`
- [x] Add batch evaluation support
- [x] Write unit tests (25 tests)

### Phase 1.5: Evolutionary Strategy ✅
- [x] Implement mutation operators
  - Rephrase instruction
  - Add/remove constraint
  - Swap example order
  - Modify formatting hints
- [x] Implement crossover operators
  - Single-point crossover
  - Component mixing
- [x] Implement selection operators
  - Tournament selection
  - Elitism
- [x] Implement `GeneticAlgorithmStrategy`
- [x] Write strategy unit tests (8 tests)

### Phase 1.6: Orchestration ✅
- [x] Implement `Optimizer` controller
- [x] Implement optimization loop
- [x] Add checkpointing/resume
- [x] Add progress callbacks
- [x] Write orchestration tests (9 tests)

### Phase 1.7: CLI ✅
- [x] Implement `optimize` command with full integration
- [x] Implement `validate` command (config validation)
- [x] Implement `report` command (view history)
- [x] Implement `list-results` command
- [x] Add rich progress display
- [x] Create `__main__.py` for module execution

### Phase 1.8: Benchmarks ✅
- [x] Create sentiment classification task
- [x] Create JSON formatting task
- [x] Create arithmetic reasoning task
- [x] Create benchmark runner script
- [x] Document benchmark usage

**Total Tests:** 84 passing across 5 test files

---

## 4. MVP 2 Detailed Plan

### Phase 2.1: Bayesian Optimization (Day 1-3)
- [ ] Add optuna dependency
- [ ] Implement `BayesianOptStrategy` with Optuna backend
- [ ] Create prompt parameterization for Bayesian search
- [ ] Add strategy registration
- [ ] Write unit tests

### Phase 2.2: Grid Search (Day 4-5)
- [ ] Implement `GridSearchStrategy`
- [ ] Support template variable combinations
- [ ] Add exhaustive and random sampling modes
- [ ] Write unit tests

### Phase 2.3: JSON Schema Evaluator (Day 6-7)
- [ ] Add jsonschema dependency
- [ ] Implement `JSONSchemaEvaluator`
- [ ] Support partial schema validation
- [ ] Write unit tests

### Phase 2.4: Visualization (Day 8-10)
- [ ] Add plotly/matplotlib dependency
- [ ] Implement fitness curve plotting
- [ ] Implement population diversity visualization
- [ ] Add export to HTML/PNG
- [ ] Add `visualize` CLI command

### Phase 2.5: Python Library API (Day 11-12)
- [ ] Create high-level `optimize()` function
- [ ] Create `load_task()` convenience function
- [ ] Document public API
- [ ] Add Jupyter notebook examples

### Phase 2.6: Enhanced Checkpointing (Day 13-14)
- [ ] Implement full population state save/restore
- [ ] Add checkpoint browsing
- [ ] Add checkpoint comparison
- [ ] Write checkpoint tests

---

## 4. Technical Decisions

### 4.1 Language & Runtime
- **Python 3.10+**: Modern syntax, pattern matching, improved typing
- **Async I/O**: httpx for async HTTP, asyncio for concurrency
- **Type Safety**: Full type annotations, mypy strict mode

### 4.2 Key Libraries

| Library | Purpose | Rationale |
|---------|---------|-----------|
| `pydantic` | Data validation | Industry standard, great DX |
| `httpx` | HTTP client | Async support, modern API |
| `typer` | CLI | Simple, type-hint based |
| `rich` | Console output | Beautiful progress/tables |
| `pytest` | Testing | Fixtures, async support |
| `ruff` | Linting/formatting | Fast, replaces flake8+black |

### 4.3 Development Tools
- **ruff**: Linting and formatting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pre-commit**: Git hooks

---

## 5. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM API instability | Comprehensive retry logic, mock client for tests |
| Slow optimization convergence | Configurable early stopping, checkpointing |
| Complex mutation semantics | Start simple (text substitution), iterate |
| Scope creep | Strict MVP boundaries, feature backlog |

---

## 6. Quality Gates

### Per-MVP Gates

| Gate | Criteria |
|------|----------|
| Tests | All tests pass, >80% coverage |
| Types | mypy passes with no errors |
| Lint | ruff passes with no warnings |
| Docs | All public APIs documented |
| Demo | End-to-end demo works |

### Release Checklist
- [ ] All quality gates pass
- [ ] CHANGELOG updated
- [ ] Version bumped
- [ ] Demo script verified
- [ ] Documentation reviewed

---

## 7. Development Workflow

### Branch Strategy
```
main ─────────────────────────────────────────▶
  │
  ├── develop ─────────────────────────────────▶
  │     │
  │     ├── feature/core-models
  │     ├── feature/llm-client
  │     └── feature/evolutionary-strategy
  │
  └── release/mvp-1
```

### Commit Convention
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scope: core, llm, strategy, evaluator, cli, docs
```

---

## 8. Milestones

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| M1: Setup Complete | Day 2 | Project skeleton, CI, docs structure |
| M2: Core Domain | Day 5 | Models, protocols, unit tests |
| M3: LLM Integration | Day 7 | Working LLM client |
| M4: Evaluation | Day 9 | Working evaluators |
| M5: Evolution | Day 14 | Working GA strategy |
| M6: MVP 1 Complete | Day 21 | CLI, benchmarks, demo |

---

## 9. Dependencies Between Tasks

```mermaid
graph TD
    A[Project Setup] --> B[Core Domain]
    B --> C[LLM Client]
    B --> D[Evaluators]
    C --> E[Orchestration]
    D --> E
    B --> F[Evolutionary Strategy]
    F --> E
    E --> G[CLI]
    E --> H[Benchmarks]
    G --> I[MVP 1 Release]
    H --> I
```

---

## 10. Documentation Requirements

Each phase must include:
- API documentation (docstrings)
- Usage examples
- Test documentation
- CHANGELOG entry

---

## 11. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-06 | Initial | Document created |

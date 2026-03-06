# PromptFoundry — Requirements Specification

> **Version:** 1.0.0  
> **Status:** Active  
> **Last Updated:** 2026-03-06  
> **Authoritative Source:** This document is the single source of truth for project requirements.

---

## 1. Overview

PromptFoundry is an optimization-driven prompt engineering tool that treats prompt engineering as a systematic optimization problem rather than manual trial-and-error. The system applies evolutionary strategies, Bayesian optimization, and gradient-free methods to discover prompts that maximize task performance.

---

## 2. Functional Requirements

### 2.1 Core Optimization Engine

| ID | Requirement | Priority | MVP |
|----|-------------|----------|-----|
| FR-001 | Support evolutionary optimization (genetic algorithm) | Must | 1 |
| FR-002 | Support Bayesian optimization | Should | 2 |
| FR-003 | Support grid search over prompt components | Should | 2 |
| FR-004 | Accept seed prompt with optional template syntax | Must | 1 |
| FR-005 | Generate prompt variants via mutation operators | Must | 1 |
| FR-006 | Generate prompt variants via crossover operators | Must | 1 |
| FR-007 | Implement tournament selection for population management | Must | 1 |
| FR-008 | Track optimization history with full lineage | Should | 2 |
| FR-009 | Support parallel prompt evaluation | Should | 2 |
| FR-010 | Implement early stopping based on convergence criteria | Should | 2 |

### 2.2 Task Definition

| ID | Requirement | Priority | MVP |
|----|-------------|----------|-----|
| FR-020 | Accept input-output example pairs (10-50 minimum) | Must | 1 |
| FR-021 | Support JSON, CSV, and YAML example formats | Must | 1 |
| FR-022 | Validate example format on load | Must | 1 |
| FR-023 | Support train/validation/test splits | Should | 2 |
| FR-024 | Provide pre-built task templates (sentiment, QA, formatting) | Could | 3 |

### 2.3 Objective Functions

| ID | Requirement | Priority | MVP |
|----|-------------|----------|-----|
| FR-030 | Support exact match accuracy | Must | 1 |
| FR-031 | Support fuzzy match accuracy (Levenshtein, semantic) | Should | 2 |
| FR-032 | Support JSON schema compliance scoring | Should | 2 |
| FR-033 | Support regex-based output validation | Must | 1 |
| FR-034 | Support custom Python scoring functions | Must | 1 |
| FR-035 | Support composite objectives (weighted combination) | Should | 2 |
| FR-036 | Support length/token budget constraints | Should | 2 |

### 2.4 LLM Integration

| ID | Requirement | Priority | MVP |
|----|-------------|----------|-----|
| FR-040 | Support OpenAI-compatible API endpoints | Must | 1 |
| FR-041 | Support local models via text-generation-webui | Must | 1 |
| FR-042 | Support configurable model parameters (temp, max_tokens) | Must | 1 |
| FR-043 | Implement rate limiting and retry logic | Should | 2 |
| FR-044 | Track token usage and costs | Should | 2 |
| FR-045 | Support multiple LLM backends simultaneously | Could | 3 |

### 2.5 Interfaces

| ID | Requirement | Priority | MVP |
|----|-------------|----------|-----|
| FR-050 | Provide CLI for optimization runs | Must | 1 |
| FR-051 | Provide Python library API | Must | 2 |
| FR-052 | Provide web UI for visualization | Could | 3 |
| FR-053 | Export results to JSON, CSV | Must | 1 |
| FR-054 | Generate optimization reports with visualizations | Should | 2 |

### 2.6 Reporting & Analysis

| ID | Requirement | Priority | MVP |
|----|-------------|----------|-----|
| FR-060 | Output best-performing prompt | Must | 1 |
| FR-061 | Output performance trajectory (score vs iteration) | Must | 1 |
| FR-062 | Perform ablation analysis (component importance) | Should | 2 |
| FR-063 | Compare multiple optimization strategies | Could | 3 |
| FR-064 | Export prompt history with genealogy | Should | 2 |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Single prompt evaluation latency | < 5s (network-dependent) |
| NFR-002 | Support population sizes up to 100 | Required |
| NFR-003 | Support optimization runs up to 500 generations | Required |
| NFR-004 | Memory usage for standard runs | < 2GB |

### 3.2 Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-010 | Graceful handling of LLM API failures | Required |
| NFR-011 | Checkpoint and resume optimization runs | Required |
| NFR-012 | Input validation with clear error messages | Required |

### 3.3 Maintainability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-020 | Code coverage > 80% | Required |
| NFR-021 | Type annotations on all public APIs | Required |
| NFR-022 | Documentation for all public modules | Required |
| NFR-023 | Modular architecture for strategy plugins | Required |

### 3.4 Portability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-030 | Python 3.10+ support | Required |
| NFR-031 | Cross-platform (Windows, Linux, macOS) | Required |
| NFR-032 | Minimal external dependencies | Required |

---

## 4. Constraints

| ID | Constraint |
|----|------------|
| C-001 | Must work with any OpenAI-compatible API (no vendor lock-in) |
| C-002 | Must not require GPU for optimization logic (LLM inference is external) |
| C-003 | Must support offline operation with local models |
| C-004 | Configuration via YAML files (no hardcoded settings) |

---

## 5. Assumptions

1. Users have access to at least one LLM (local or API-based).
2. Users can provide task-specific input-output examples.
3. Network connectivity is available for API-based LLMs.
4. Python 3.10+ is installed on the target system.

---

## 6. Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| `pydantic` | Data validation and settings | ^2.0 |
| `httpx` | Async HTTP client for API calls | ^0.24 |
| `rich` | CLI output formatting | ^13.0 |
| `typer` | CLI framework | ^0.9 |
| `pyyaml` | Configuration parsing | ^6.0 |
| `numpy` | Numerical operations | ^1.24 |
| `optuna` | Bayesian optimization (MVP 2) | ^3.0 |
| `deap` | Evolutionary algorithms | ^1.4 |

---

## 7. Glossary

| Term | Definition |
|------|------------|
| **Prompt** | Text instruction given to an LLM to elicit a response |
| **Seed Prompt** | Initial prompt used to start optimization |
| **Objective Function** | Scoring function that evaluates prompt performance |
| **Mutation** | Operation that modifies a prompt to create variants |
| **Crossover** | Operation that combines parts of two prompts |
| **Population** | Set of prompt variants in an evolutionary generation |
| **Generation** | One iteration of the optimization loop |

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-06 | Initial | Document created |

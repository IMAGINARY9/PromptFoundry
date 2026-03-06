# PromptFoundry

> Optimization-driven prompt engineering tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: MVP 1](https://img.shields.io/badge/Status-MVP%201%20Complete-green.svg)]()

---

## Overview

PromptFoundry treats **prompt engineering as a systematic optimization problem**. The current implementation focuses on an evolutionary search loop with pluggable evaluators, while the roadmap expands into proxy-metric pipelines and additional search methods only after the baseline method is reliable.

### Key Features

- 🧬 **Evolutionary optimization**: Genetic algorithms with mutation/crossover
- 🔌 **LLM-agnostic**: Works with any OpenAI-compatible API (including local models)
- 📊 **Multiple evaluators**: Exact match, fuzzy match, regex, custom functions
- 🚀 **Rate limiting**: Built-in token bucket for API compliance  
- 📈 **Progress tracking**: Rich CLI with progress bars, per-generation timing, and cancelable runs
- ⚡ **Caching & concurrency**: Avoids duplicate LLM requests and evaluates examples in parallel
- 🛠️ **Extensible**: Protocol-based interfaces for custom components

### Current Scope

- Current search method: evolutionary optimization only
- Current strengths: format-constrained tasks, extraction, classification, and tasks with cheap proxy metrics
- Current limitation: performance is still dominated by the LLM backend; the framework now exposes runtime controls through YAML and CLI so slow local models can be tuned explicitly
- Planned next step: improve evaluation quality and runtime efficiency before adding more search algorithms

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/PromptFoundry.git
cd PromptFoundry

# Install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Test LLM connection (requires running LLM server)
python -m promptfoundry test-llm --base-url http://localhost:5000/v1

# Run optimization
python -m promptfoundry optimize \
  --task examples/sentiment_task.yaml \
  --seed-prompt "Classify the sentiment: {input}" \
  --max-generations 20 \
  --population-size 4 \
  --max-concurrency 1

# Validate a configuration file
python -m promptfoundry validate --config config/config.example.yaml

# List available evaluators
python -m promptfoundry list-evaluators

# View optimization results
python -m promptfoundry report output/optimization_20260307_120000.json
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `optimize` | Run prompt optimization on a task (Ctrl-C interrupts, partial results may be lost) |
| `validate` | Validate configuration file |
| `report` | View optimization result details |
| `list-results` | List all saved optimization results |
| `test-llm` | Test LLM connection |
| `list-strategies` | Show available strategies |
| `list-evaluators` | Show available evaluators |
| `version` | Show version information |

### Tuning Slow Local Models

For slow local backends, tune the optimizer through `config/config.example.yaml` or CLI overrides instead of changing source defaults. Recommended starting point for text-generation-webui:

```yaml
optimization:
  max_generations: 10
  population_size: 2-4
  max_concurrency: 1
  patience: 3-5
```

Prefer cheaper proxy metrics first, then reserve exact-match or expensive judge-style checks for a shortlist of promising candidates.

---

## Task File Format

```yaml
# examples/my_task.yaml
name: my_classification_task

system_prompt: |
  You are a helpful assistant.

evaluator: exact_match
evaluator_config:
  case_sensitive: false

examples:
  - input: "Sample input text"
    expected: "expected output"
  - input: "Another input"
    expected: "another output"
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and components |
| [REQUIREMENTS.md](docs/REQUIREMENTS.md) | Functional and non-functional requirements |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Development roadmap |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration reference |
| [DOCUMENTATION.md](docs/DOCUMENTATION.md) | Documentation standards |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/promptfoundry

# Type checking
mypy src/promptfoundry --ignore-missing-imports
```

---

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development setup and guidelines.

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check .
```

---

## Local LLM Setup

PromptFoundry works with local LLMs via text-generation-webui. Configure:

```yaml
# config.yaml
llm:
  base_url: http://127.0.0.1:5000/v1
  api_key: local
  model: Mistral-7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

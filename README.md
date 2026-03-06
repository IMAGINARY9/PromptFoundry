# PromptFoundry

> Optimization-driven prompt engineering tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

PromptFoundry treats **prompt engineering as a systematic optimization problem**. Instead of manually tweaking prompts, define an objective function and let the system search the prompt space using evolutionary strategies, Bayesian optimization, or gradient-free methods.

### Key Features

- 🧬 **Multiple optimization strategies**: Evolutionary algorithms, Bayesian optimization, grid search
- 🔌 **LLM-agnostic**: Works with any OpenAI-compatible API (including local models)
- 📊 **Rich evaluation**: Exact match, fuzzy match, JSON schema validation, custom scorers
- 📈 **Detailed reporting**: Performance trajectories, ablation analysis, prompt genealogy
- 🛠️ **Extensible**: Plugin architecture for custom strategies, evaluators, and operators

---

## Quick Start

### Installation

```bash
pip install promptfoundry
```

For development:
```bash
git clone git@github.com:IMAGINARY9/PromptFoundry.git
cd PromptFoundry
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Optimize a prompt for sentiment classification
promptfoundry optimize \
  --task examples/sentiment_task.yaml \
  --seed-prompt "Classify the sentiment of this text: {input}"
```

### Python API

```python
from promptfoundry import Optimizer, Task

optimizer = Optimizer.from_config("config.yaml")
result = optimizer.optimize(
    task=Task.from_file("task.yaml"),
    seed_prompt="Classify: {input}"
)

print(f"Best prompt: {result.best_prompt}")
print(f"Accuracy: {result.best_score:.2%}")
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

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

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

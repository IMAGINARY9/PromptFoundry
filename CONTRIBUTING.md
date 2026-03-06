# Contributing to PromptFoundry

Thank you for your interest in contributing to PromptFoundry! This document provides guidelines and instructions for contributing.

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) text-generation-webui for local LLM testing

### Quick Start

```bash
# Clone the repository
git clone git@github.com:IMAGINARY9/PromptFoundry.git
cd PromptFoundry

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/macOS

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run type checking
mypy src/

# Run linting
ruff check .
```

---

## Code Standards

**For complete code standards, see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).**

Key points:
- Full type annotations on all public APIs
- Follow established patterns (Strategy, Adapter, Protocol)
- Write tests for all new functionality
- Keep documentation current

---

## Making Changes

### Branch Workflow

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with meaningful commits:
   ```bash
   git commit -m "feat(core): add prompt template support"
   ```

3. Push and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Convention

```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scope: core, llm, strategy, evaluator, cli, docs
```

Examples:
- `feat(strategy): add bayesian optimization strategy`
- `fix(llm): handle timeout errors gracefully`
- `docs(readme): update installation instructions`

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/promptfoundry --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run tests matching pattern
pytest -k "test_mutation"
```

---

## Documentation

- Keep [docs/](docs/) current when making changes
- Follow single source of truth principle
- Update [CHANGELOG.md](CHANGELOG.md) for user-facing changes

---

## Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally (`pytest`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Linting passes (`ruff check .`)
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG updated (if user-facing change)
- [ ] Commits follow convention

---

## Getting Help

- Open an issue for bugs or feature requests
- Check existing documentation first
- Provide minimal reproducible examples for bugs

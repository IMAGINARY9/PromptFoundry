# PromptFoundry — Configuration Reference

> **Version:** 1.0.0  
> **Status:** Active  
> **Last Updated:** 2026-03-06  
> **Authoritative Source:** This document is the single source of truth for configuration options.

---

## 1. Overview

PromptFoundry uses a layered configuration system:
1. **Default values** (built into code)
2. **Configuration file** (YAML)
3. **Environment variables** (override file settings)
4. **CLI arguments** (highest priority)

---

## 2. Configuration File

The default configuration file is `config.yaml` in the working directory. Specify a different path with `--config` CLI flag.

### 2.1 Full Configuration Schema

```yaml
# PromptFoundry Configuration
# See docs/CONFIGURATION.md for details

# =============================================================================
# Optimization Settings
# =============================================================================
optimization:
  # Strategy to use: evolutionary | bayesian | grid
  strategy: evolutionary
  
  # Maximum number of generations/iterations
  max_generations: 100
  
  # Population size for evolutionary/bayesian strategies
  population_size: 20
  
  # Early stopping: generations without improvement before stopping
  patience: 10
  
  # Random seed for reproducibility (null for random)
  seed: null

# =============================================================================
# Strategy-Specific Settings
# =============================================================================
strategy:
  evolutionary:
    # Probability of mutation per individual
    mutation_rate: 0.3
    
    # Probability of crossover between individuals
    crossover_rate: 0.7
    
    # Number of individuals in tournament selection
    tournament_size: 3
    
    # Number of top individuals to preserve unchanged
    elitism: 2
    
    # Mutation operators to use
    mutation_operators:
      - rephrase
      - add_constraint
      - remove_constraint
      - swap_examples
    
    # Crossover operators to use
    crossover_operators:
      - single_point
      - component_mix
  
  bayesian:
    # Acquisition function: ei | lcb | pi
    acquisition: ei
    
    # Number of initial random samples
    n_initial: 10
    
    # Exploration-exploitation trade-off
    exploration_weight: 0.1
  
  grid:
    # Components to vary
    components:
      - instruction_style
      - example_count
      - format_specification

# =============================================================================
# LLM Settings
# =============================================================================
llm:
  # Provider type: openai_compat | anthropic | custom
  provider: openai_compat
  
  # API base URL (for openai_compat)
  base_url: http://127.0.0.1:5000/v1
  
  # API key (use environment variable for security)
  api_key: ${OPENAI_API_KEY:local}
  
  # Model name/path
  model: Mistral-7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf
  
  # Generation parameters
  temperature: 0.7
  max_tokens: 256
  top_p: 1.0
  
  # Request settings
  timeout: 30
  max_retries: 3
  retry_delay: 1.0

# =============================================================================
# Evaluation Settings
# =============================================================================
evaluation:
  # Primary evaluator type: exact_match | fuzzy_match | regex | json_schema | custom
  type: exact_match
  
  # Case sensitivity for text matching
  case_sensitive: false
  
  # Strip whitespace before comparison
  strip_whitespace: true
  
  # For fuzzy_match: similarity threshold (0.0-1.0)
  similarity_threshold: 0.8
  
  # For regex: pattern to match
  regex_pattern: null
  
  # For json_schema: path to schema file
  json_schema_path: null
  
  # For custom: path to Python module with evaluate function
  custom_evaluator: null
  
  # Composite evaluator weights (if using multiple)
  weights: null

# =============================================================================
# Logging Settings
# =============================================================================
logging:
  # Log level: DEBUG | INFO | WARNING | ERROR
  level: INFO
  
  # Output format: text | json | structured
  format: text
  
  # Log file path (null for console only)
  file: null
  
  # Include timestamps in console output
  timestamps: true

# =============================================================================
# Output Settings
# =============================================================================
output:
  # Directory for output files
  directory: ./output
  
  # Save optimization history
  save_history: true
  
  # Export format: json | csv | both
  format: json
  
  # Include all variants (not just best)
  include_all_variants: false
  
  # Generate performance plots (requires matplotlib)
  generate_plots: false
```

---

## 3. Environment Variables

All configuration options can be overridden via environment variables using the pattern:

```
PROMPTFOUNDRY_<SECTION>__<KEY>=value
```

### 3.1 Common Environment Variables

| Variable | Config Path | Description |
|----------|-------------|-------------|
| `PROMPTFOUNDRY_LLM__BASE_URL` | `llm.base_url` | LLM API endpoint |
| `PROMPTFOUNDRY_LLM__API_KEY` | `llm.api_key` | API key |
| `PROMPTFOUNDRY_LLM__MODEL` | `llm.model` | Model name |
| `PROMPTFOUNDRY_OPTIMIZATION__STRATEGY` | `optimization.strategy` | Strategy type |
| `PROMPTFOUNDRY_OPTIMIZATION__MAX_GENERATIONS` | `optimization.max_generations` | Max iterations |
| `PROMPTFOUNDRY_LOGGING__LEVEL` | `logging.level` | Log verbosity |

### 3.2 Special Variables

| Variable | Description |
|----------|-------------|
| `PROMPTFOUNDRY_CONFIG` | Path to config file |
| `OPENAI_API_KEY` | Fallback for `llm.api_key` |
| `OPENAI_API_BASE` | Fallback for `llm.base_url` |

---

## 4. Task File Format

Tasks are defined in YAML files:

```yaml
# task.yaml
name: sentiment_classification

# System prompt (optional)
system_prompt: |
  You are a sentiment analysis assistant.

# Input-output examples (minimum 10)
examples:
  - input: "I love this product!"
    output: "positive"
  
  - input: "This is terrible, I want a refund."
    output: "negative"
  
  - input: "It works as expected, nothing special."
    output: "neutral"
  
  # ... more examples

# Validation examples (optional, used for early stopping)
validation_examples:
  - input: "Best purchase I've ever made!"
    output: "positive"
```

---

## 5. CLI Reference

```bash
# Basic optimization
promptfoundry optimize --task task.yaml --seed-prompt "Classify: {input}"

# With custom config
promptfoundry optimize --task task.yaml --config my_config.yaml

# Override settings via CLI
promptfoundry optimize --task task.yaml \
  --strategy bayesian \
  --max-generations 50 \
  --llm-base-url http://localhost:5000/v1

# Validate configuration
promptfoundry validate --config config.yaml

# Generate report from history
promptfoundry report --history output/history.json --format html

# List available strategies and evaluators
promptfoundry list-strategies
promptfoundry list-evaluators
```

---

## 6. Local LLM Configuration

For use with text-generation-webui (see `USAGE.md` in project root):

```yaml
llm:
  provider: openai_compat
  base_url: http://127.0.0.1:5000/v1
  api_key: local  # Any non-empty string
  model: Mistral-7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf
  temperature: 0.7
  max_tokens: 256
```

Environment variables:
```bash
export OPENAI_API_BASE=http://127.0.0.1:5000/v1
export OPENAI_API_KEY=local
```

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-06 | Initial | Document created |

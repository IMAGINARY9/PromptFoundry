# PromptFoundry — Configuration Reference

> **Version:** 1.1.0  
> **Status:** Active  
> **Last Updated:** 2026-03-07  
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

  # Maximum parallel LLM calls
  max_concurrency: 1

  # Runtime cap in seconds (0 disables budget stop)
  runtime_budget: 0

  # Adaptive plateau stopping for stagnant runs
  adaptive_early_stopping: true
  plateau_window: 3
  min_progress_delta: 0.01
  budget_buffer_ratio: 0.85
  
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

    # Reweight mutation operators from observed fitness gains
    adaptive_mutation_weights: true
    min_operator_weight: 0.4
    weight_learning_rate: 0.8

    # MVP 3 evolutionary-quality controls
    use_semantic_mutations: true
    use_diversity_control: true
    use_adaptive_schedule: true
    schedule_type: operator  # adaptive | linear | constant | operator
    enable_ablation_tracking: true
    min_diversity_ratio: 0.7
    crowding_penalty: 0.1

    # The semantic mutation/crossover operator library is built in.
    # Tune behavior through weights and rates rather than operator name lists.
  
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
  # Note: if using a runtime profile (e.g., slow-local) and no explicit
  # LLM timeout is provided, the optimizer will inherit
  # `runtime_config.timeout_per_request` automatically.
  # This ensures the LLM client honors slow-local timeouts.


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

  # For exact_match: normalize concise final answers out of verbose completions
  normalize_output: true
  # Set to false for strict label-only or number-only tasks
  
  # For fuzzy_match: similarity threshold (0.0-1.0)
  similarity_threshold: 0.8
  
  # For regex: pattern to match
  regex_pattern: null
  # Fixed regex patterns may include an {expected} placeholder, for example:
  #   "\\b{expected}\\b"
  # Set full_match=true when the task requires a bare answer with no explanation.
  
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

  # Optional checkpoint directory for resumable optimization runs
  checkpoint_dir: null

  # Save a checkpoint every N completed generations
  checkpoint_frequency: 5
  
  # Save optimization history
  save_history: true
  
  # Export format: json | csv | both
  format: json
  
  # Include all variants (not just best)
  include_all_variants: false
  
  # Generate performance plots (requires matplotlib)
  generate_plots: false
  
  # Debugging: the optimizer now records every prompt/completion pair
  # evaluated during a run in the history metadata under
  # `history.generations[].metadata.interactions`.  This log can grow large,
  # so only enable it when diagnosing problems.

  # MVP 3 diagnostics included in each result file:
  # - detected_task_type
  # - detected_output_mode
  # - diversity_metrics
  # - schedule_state
  # - ablation_result
  # - ablation_summary
  # - lineage_report

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

## 6. Local LLM Setup

### 6.1 text-generation-webui Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/oobabooga/text-generation-webui
   cd text-generation-webui
   ```

2. **Run the one-click installer:**
   - Windows: `start_windows.bat`
   - Linux: `./start_linux.sh`
   - macOS: `./start_macos.sh`

3. **Download a model (recommended: Mistral-7B-Instruct):**
   ```
   # In the webui interface, go to Model → Download
   # Or use the CLI:
   python download-model.py TheBloke/Mistral-7B-Instruct-v0.2-GGUF
   ```

4. **Start with API enabled:**
   ```bash
   python server.py --api --listen
   ```

### 6.2 PromptFoundry Configuration

YAML configuration:
```yaml
llm:
  provider: openai_compat
  base_url: http://127.0.0.1:5000/v1
  api_key: local  # Any non-empty string
  model: Mistral-7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf
  temperature: 0.7
  max_tokens: 256
```

Environment variables (alternative):
```bash
export OPENAI_API_BASE=http://127.0.0.1:5000/v1
export OPENAI_API_KEY=local
```

### 6.3 Test Connection

```bash
# Verify LLM is accessible and responding
python scripts/test_llm_connection.py

# Custom endpoint
python scripts/test_llm_connection.py --base-url http://192.168.1.10:5000/v1

# Specific model
python scripts/test_llm_connection.py --model mistral-7b-instruct
```

The test script performs:
1. **Health check** - Server reachability
2. **Basic completion** - Simple prompt response
3. **Optimization test** - Domain-specific prompt optimization task

### 6.4 Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure server is running: `python server.py --api --listen` |
| No models available | Load a model in the webui interface |
| Slow responses | Use quantized models (Q4_K_M) for faster inference |
| Out of memory | Try smaller model or lower context length |

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-06 | Initial | Document created |

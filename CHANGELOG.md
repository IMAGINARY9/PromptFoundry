# Changelog

All notable changes to PromptFoundry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core documentation (REQUIREMENTS, ARCHITECTURE, IMPLEMENTATION_PLAN)
- Development tooling configuration (ruff, mypy, pytest)
- pyproject.toml with dependencies
- MVP 3 semantic mutation, diversity, scheduling, and ablation modules
- MVP 3 validation configs for baseline and full evolutionary-quality runs
- LabelAnswerEvaluator for strict classification tasks that keep exact-match full credit while awarding conservative partial credit to verbose local-model outputs
- JsonValueCoverageEvaluator and a value-recovery stage for long-context extraction pipelines before strict JSON formatting succeeds
- Hierarchical intent routing and long-context extraction benchmark tasks

### Changed
- Evolutionary runtime now applies crowding penalties during selection
- CLI configuration now exposes MVP 3 strategy controls and saves diversity/schedule/ablation diagnostics in result files
- Optimization results now persist lineage reports for best prompts, normalized final answers in `completion`, and original model traces in `raw_completion` when cleanup is applied
- Pipeline evaluator failures now feed stage-aware mutation bias for later generations
- OpenAI-compatible client requests now forward provider-specific `llm.extra` payload fields instead of silently dropping them

### Fixed
- Wired adaptive mutation schedules into the live evolutionary loop instead of leaving them dormant
- JSON parsing no longer treats placeholder instructional objects as valid extraction outputs

### Removed
- N/A

## [0.1.0] - TBD

### Added
- MVP 1: CLI optimizer with evolutionary strategy
- Core domain models (Prompt, Task, Population)
- OpenAI-compatible LLM client
- Basic evaluators (exact match, regex)
- CLI interface
- Benchmark tasks

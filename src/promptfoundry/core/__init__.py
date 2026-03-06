"""Core domain models and interfaces for PromptFoundry."""

from promptfoundry.core.config import (
    RuntimeConfig,
    RuntimeProfile,
    get_available_profiles,
    get_profile_description,
)
from promptfoundry.core.diagnostics import (
    BenchmarkSummary,
    GenerationMetrics,
    RunDiagnostics,
    RunStatus,
    TerminationReason,
    format_benchmark_summary,
    format_diagnostics_report,
)
from promptfoundry.core.history import OptimizationHistory, OptimizationResult
from promptfoundry.core.optimizer import Optimizer, OptimizerConfig
from promptfoundry.core.population import Individual, Population
from promptfoundry.core.prompt import Prompt, PromptTemplate
from promptfoundry.core.protocols import (
    Evaluator,
    LLMClient,
    OptimizationStrategy,
)
from promptfoundry.core.task import Example, Task

__all__ = [
    "BenchmarkSummary",
    "Evaluator",
    "Example",
    "GenerationMetrics",
    "Individual",
    "LLMClient",
    "OptimizationHistory",
    "OptimizationResult",
    "OptimizationStrategy",
    "Optimizer",
    "OptimizerConfig",
    "Population",
    "Prompt",
    "PromptTemplate",
    "RunDiagnostics",
    "RunStatus",
    "RuntimeConfig",
    "RuntimeProfile",
    "Task",
    "TerminationReason",
    "format_benchmark_summary",
    "format_diagnostics_report",
    "get_available_profiles",
    "get_profile_description",
]

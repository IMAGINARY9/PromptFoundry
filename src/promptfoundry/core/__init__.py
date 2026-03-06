"""Core domain models and interfaces for PromptFoundry."""

from promptfoundry.core.benchmark_gate import (
    BenchmarkGate,
    BenchmarkTask,
    BenchmarkTaskType,
    BenchmarkThreshold,
    GateResult,
    create_custom_gate,
    get_default_suite,
)
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
    "BenchmarkGate",
    "BenchmarkSummary",
    "BenchmarkTask",
    "BenchmarkTaskType",
    "BenchmarkThreshold",
    "Evaluator",
    "Example",
    "GateResult",
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
    "create_custom_gate",
    "format_benchmark_summary",
    "format_diagnostics_report",
    "get_available_profiles",
    "get_default_suite",
    "get_profile_description",
]

"""Core domain models and interfaces for PromptFoundry."""

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
    "Evaluator",
    "Example",
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
    "Task",
]

"""PromptFoundry - Optimization-driven prompt engineering tool."""

from promptfoundry.core.history import OptimizationHistory, OptimizationResult
from promptfoundry.core.population import Individual, Population
from promptfoundry.core.prompt import Prompt, PromptTemplate
from promptfoundry.core.task import Example, Task

__version__ = "0.1.0"
__all__ = [
    "Example",
    "Individual",
    "OptimizationHistory",
    "OptimizationResult",
    "Population",
    "Prompt",
    "PromptTemplate",
    "Task",
]

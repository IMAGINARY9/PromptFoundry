"""Optimization strategies for PromptFoundry.

This package provides optimization strategies and supporting components:
- GeneticAlgorithmStrategy: Main evolutionary optimization strategy
- SemanticMutationLibrary: Task-aware mutation operators
- DiversityController: Population diversity management
- MutationSchedule variants: Adaptive mutation rate schedules
- AblationTracker: Operator effectiveness analysis
"""

from promptfoundry.strategies.ablation import (
    AblationResult,
    AblationStudy,
    AblationTracker,
    OperatorMetrics,
)
from promptfoundry.strategies.base import BaseStrategy
from promptfoundry.strategies.diversity import (
    DiversityController,
    DiversityMetrics,
    LineageNode,
)
from promptfoundry.strategies.evolutionary import GeneticAlgorithmStrategy
from promptfoundry.strategies.schedules import (
    AdaptiveSchedule,
    ConstantSchedule,
    LinearDecaySchedule,
    MutationSchedule,
    MutationScheduleState,
    OperatorAdaptiveSchedule,
    SchedulePhase,
    create_schedule,
)
from promptfoundry.strategies.semantic_mutations import (
    OutputMode,
    SemanticMutation,
    SemanticMutationLibrary,
    TaskDetector,
    TaskType,
    get_mutation_library,
)

__all__ = [
    # Base
    "BaseStrategy",
    # Evolutionary
    "GeneticAlgorithmStrategy",
    # Semantic Mutations
    "SemanticMutation",
    "SemanticMutationLibrary",
    "TaskDetector",
    "TaskType",
    "OutputMode",
    "get_mutation_library",
    # Diversity
    "DiversityController",
    "DiversityMetrics",
    "LineageNode",
    # Schedules
    "MutationSchedule",
    "MutationScheduleState",
    "SchedulePhase",
    "ConstantSchedule",
    "LinearDecaySchedule",
    "AdaptiveSchedule",
    "OperatorAdaptiveSchedule",
    "create_schedule",
    # Ablation
    "AblationTracker",
    "AblationStudy",
    "AblationResult",
    "OperatorMetrics",
]

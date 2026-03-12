"""Evolutionary (Genetic Algorithm) optimization strategy.

This module implements a genetic algorithm-based optimization strategy
for prompt engineering, including mutation, crossover, and selection.

MVP 3 Enhancements:
- Semantic mutation integration for task-aware transformations
- Diversity controls for preventing premature convergence
- Adaptive mutation schedules responding to optimization dynamics
- Ablation tracking for operator effectiveness analysis
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from promptfoundry.core.population import Individual, Population
from promptfoundry.core.prompt import Prompt
from promptfoundry.strategies.base import BaseStrategy, StrategyConfig


@dataclass
class EvolutionaryConfig(StrategyConfig):
    """Configuration for evolutionary strategy.

    Attributes:
        mutation_rate: Probability of mutation per individual (0.0-1.0).
        crossover_rate: Probability of crossover (0.0-1.0).
        tournament_size: Number of individuals in tournament selection.
        elitism: Number of top individuals to preserve unchanged.
        adaptive_mutation_weights: Whether to adapt operator weights.
        min_operator_weight: Minimum weight for any operator.
        weight_learning_rate: Rate of weight adaptation.
        use_semantic_mutations: Enable semantic mutation library (MVP 3).
        use_diversity_control: Enable diversity tracking (MVP 3).
        use_adaptive_schedule: Enable adaptive mutation schedule (MVP 3).
        enable_ablation_tracking: Track operator effectiveness (MVP 3).
        min_diversity_ratio: Minimum unique prompt ratio before diversity injection.
        crowding_penalty: Penalty factor for similar individuals (0-1).
    """

    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elitism: int = 2
    adaptive_mutation_weights: bool = True
    min_operator_weight: float = 0.4
    weight_learning_rate: float = 0.8
    # MVP 3 options
    use_semantic_mutations: bool = True
    use_diversity_control: bool = True
    use_adaptive_schedule: bool = False  # Off by default for backward compatibility
    schedule_type: str = "adaptive"
    enable_ablation_tracking: bool = True
    min_diversity_ratio: float = 0.7
    crowding_penalty: float = 0.1


@dataclass(frozen=True)
class MutationOperator:
    """Declarative mutation operator used by the genetic algorithm."""

    name: str
    weight: float
    transform: Callable[[str], str]


class GeneticAlgorithmStrategy(BaseStrategy):
    """Genetic algorithm-based optimization strategy.

    Uses tournament selection, single-point crossover, and text mutation
    to evolve a population of prompts toward better performance.

    MVP 3 Enhancements:
    - Semantic mutation integration for task-aware transformations
    - Diversity controls for preventing premature convergence
    - Ablation tracking for operator effectiveness analysis
    """

    def __init__(self, config: EvolutionaryConfig | None = None) -> None:
        """Initialize the genetic algorithm strategy.

        Args:
            config: Evolutionary configuration. Uses defaults if None.
        """
        self.evo_config = config or EvolutionaryConfig()
        super().__init__(self.evo_config)
        self._operator_base_weights: dict[str, float] = {}
        self._operator_current_weights: dict[str, float] = {}
        self._operator_stats: dict[str, dict[str, float]] = {}
        self._last_generation_summary: dict[str, dict[str, float]] = {}

        # MVP 3: Task type detection and semantic mutations
        self._detected_task_type: str | None = None
        self._detected_output_mode: str | None = None
        self._semantic_library: Any = None  # Lazy imported

        # MVP 3: Diversity controller
        self._diversity_controller: Any = None  # Lazy imported
        self._last_diversity_metrics: dict[str, Any] = {}

        # MVP 3: Ablation tracker
        self._ablation_tracker: Any = None  # Lazy imported

        # MVP 3: Adaptive schedule
        self._mutation_schedule: Any = None  # Lazy imported
        self._schedule_state: Any = None  # Lazy imported
        self._effective_mutation_rate = self.evo_config.mutation_rate
        self._effective_crossover_rate = self.evo_config.crossover_rate
        self._stage_feedback_by_prompt: dict[str, dict[str, Any]] = {}

    def _ensure_mvp3_components(self) -> None:
        """Lazy-load MVP 3 components to avoid circular imports."""
        if self.evo_config.use_semantic_mutations and self._semantic_library is None:
            from promptfoundry.strategies.semantic_mutations import (
                get_mutation_library,
            )

            self._semantic_library = get_mutation_library()

        if self.evo_config.use_diversity_control and self._diversity_controller is None:
            from promptfoundry.strategies.diversity import DiversityController

            self._diversity_controller = DiversityController(
                min_unique_ratio=self.evo_config.min_diversity_ratio,
            )

        if self.evo_config.enable_ablation_tracking and self._ablation_tracker is None:
            from promptfoundry.strategies.ablation import AblationTracker

            self._ablation_tracker = AblationTracker()

        if self.evo_config.use_adaptive_schedule and self._mutation_schedule is None:
            from promptfoundry.strategies.schedules import MutationScheduleState, create_schedule

            schedule_type = (
                "operator" if self.evo_config.adaptive_mutation_weights else self.evo_config.schedule_type
            )
            self._mutation_schedule = create_schedule(
                schedule_type=schedule_type,
                base_mutation_rate=self.evo_config.mutation_rate,
                crossover_rate=self.evo_config.crossover_rate,
                operator_learning_rate=self.evo_config.weight_learning_rate,
                min_operator_weight=self.evo_config.min_operator_weight,
            )
            self._schedule_state = MutationScheduleState(
                max_generations=self.evo_config.max_generations,
                current_mutation_rate=self.evo_config.mutation_rate,
                current_crossover_rate=self.evo_config.crossover_rate,
            )

    def initialize(
        self,
        seed_prompt: Prompt,
        population_size: int,
    ) -> Population:
        """Create initial population from seed prompt.

        The initial population includes the seed prompt and variants
        created through mutation.

        Args:
            seed_prompt: The starting prompt.
            population_size: Desired population size.

        Returns:
            Initial population.
        """
        individuals: list[Individual] = []
        self._ensure_operator_stats_initialized()
        self._ensure_mvp3_components()

        # MVP 3: Detect task type from seed prompt
        if self.evo_config.use_semantic_mutations:
            from promptfoundry.strategies.semantic_mutations import TaskDetector

            task_type = TaskDetector.detect_task_type(seed_prompt.text, seed_prompt.metadata)
            output_mode = TaskDetector.detect_output_mode(
                seed_prompt.text,
                task_type,
                seed_prompt.metadata,
            )
            self._detected_task_type = task_type.value
            self._detected_output_mode = output_mode.value
        else:
            task_type = None
            output_mode = None

        # MVP 3: Reset diversity controller
        if self._diversity_controller:
            self._diversity_controller.reset()
        self._stage_feedback_by_prompt = {}

        if self._schedule_state is not None:
            self._schedule_state.generation = 0
            self._schedule_state.max_generations = self.evo_config.max_generations
            self._schedule_state.current_mutation_rate = self.evo_config.mutation_rate
            self._schedule_state.current_crossover_rate = self.evo_config.crossover_rate
            self._schedule_state.stall_count = 0
            self._schedule_state.best_fitness = 0.0
            self._schedule_state.avg_fitness = 0.0
            self._schedule_state.diversity_score = 1.0
            self._schedule_state.operator_weights = {}
            self._effective_mutation_rate = self.evo_config.mutation_rate
            self._effective_crossover_rate = self.evo_config.crossover_rate

        # Include the original seed
        seed_individual = Individual(
            prompt=self._copy_prompt(seed_prompt, metadata_updates={"is_seed": True}),
            generation=0,
            parent_ids=[],
        )
        individuals.append(seed_individual)
        seen_texts = {seed_prompt.text}

        # MVP 3: Register seed with diversity controller
        if self._diversity_controller:
            self._diversity_controller.register_prompt(
                prompt_id=seed_individual.id,
                text=seed_prompt.text,
                generation=0,
            )

        # Seed the initial population with a few task-aware variants so low-budget
        # runs still explore viable strict-output prompts in generation 0.
        for guided_prompt in self._generate_guided_seed_variants(
            seed_prompt,
            seen_texts,
            population_size - 1,
            task_type,
            output_mode,
        ):
            seen_texts.add(guided_prompt.text)
            individual = Individual(
                prompt=guided_prompt,
                generation=0,
                parent_ids=[seed_prompt.id],
            )
            individuals.append(individual)
            if self._diversity_controller:
                self._diversity_controller.register_prompt(
                    prompt_id=individual.id,
                    text=guided_prompt.text,
                    generation=0,
                    parent_ids=[seed_prompt.id],
                    mutation_operator=guided_prompt.metadata.get("mutation_operator"),
                )

        # Generate remaining variants through mutation
        while len(individuals) < population_size:
            mutated = self._create_unique_prompt(seed_prompt, seen_texts)
            seen_texts.add(mutated.text)
            individual = Individual(
                prompt=mutated,
                generation=0,
                parent_ids=[seed_prompt.id],
            )
            individuals.append(individual)

            # MVP 3: Register with diversity controller
            if self._diversity_controller:
                self._diversity_controller.register_prompt(
                    prompt_id=individual.id,
                    text=mutated.text,
                    generation=0,
                    parent_ids=[seed_prompt.id],
                    mutation_operator=mutated.metadata.get("mutation_operator"),
                )

        return Population(individuals=individuals, generation=0)

    def _generate_guided_seed_variants(
        self,
        seed_prompt: Prompt,
        seen_texts: set[str],
        limit: int,
        task_type: Any,
        output_mode: Any,
    ) -> list[Prompt]:
        """Build deterministic high-signal seed variants for low-budget runs."""
        if limit <= 0:
            return []

        task_value = getattr(task_type, "value", None)
        variants: list[Prompt] = []
        recipes: list[tuple[str, list[Any]]] = []

        if task_value == "classification":
            recipes.extend(
                [
                    (
                        "guided_label_exact",
                        [
                            self._mutate_add_answer_only_directive,
                            lambda text: self._append_first_missing_clause(
                                text,
                                [
                                    " Return exactly one label.",
                                    " Answer with one label only and no punctuation.",
                                ],
                            ),
                        ],
                    ),
                    (
                        "guided_label_suppressed",
                        [
                            lambda text: self._append_first_missing_clause(
                                text,
                                [" Output only the classification label. No explanation."],
                            ),
                            self._mutate_add_constraint,
                        ],
                    ),
                ]
            )
        elif task_value == "numeric":
            recipes.extend(
                [
                    (
                        "guided_numeric_digits",
                        [
                            self._mutate_add_answer_only_directive,
                            lambda text: self._append_first_missing_clause(
                                text,
                                [
                                    " Return only the number.",
                                    " Output digits only with no words or units.",
                                ],
                            ),
                        ],
                    ),
                    (
                        "guided_numeric_verified",
                        [
                            lambda text: self._append_first_missing_clause(
                                text,
                                [
                                    " Provide the final numeric result only. No words, units, or explanation.",
                                ],
                            ),
                            self._mutate_add_verification_directive,
                        ],
                    ),
                    (
                        "guided_numeric_structured",
                        [
                            self._mutate_promote_structured_layout,
                            lambda text: self._append_first_missing_clause(
                                text,
                                [" If the answer is numeric, output digits only with no words or units."],
                            ),
                        ],
                    ),
                ]
            )
        elif task_value == "extraction" or getattr(output_mode, "value", None) == "structured":
            recipes.extend(
                [
                    (
                        "guided_structured_json",
                        [
                            self._mutate_promote_structured_layout,
                            lambda text: self._append_first_missing_clause(
                                text,
                                [
                                    " Return a single JSON object.",
                                    " Output valid JSON only with no explanation.",
                                ],
                            ),
                        ],
                    ),
                    (
                        "guided_structured_verified",
                        [
                            lambda text: self._append_first_missing_clause(
                                text,
                                [
                                    " Preserve the required keys and return only the final structured output.",
                                ],
                            ),
                            self._mutate_add_verification_directive,
                        ],
                    ),
                ]
            )

            if seed_prompt.metadata.get("task_metadata", {}).get("requires_missing_field_handling"):
                recipes.append(
                    (
                        "guided_structured_nulls",
                        [
                            lambda text: self._append_first_missing_clause(
                                text,
                                [" Use null for any missing or unknown fields."],
                            ),
                            self._mutate_add_answer_only_directive,
                        ],
                    )
                )

        for recipe_name, transforms in recipes:
            if len(variants) >= limit:
                break
            text = seed_prompt.text
            for transform in transforms:
                text = transform(text)
            normalized = self._normalize_prompt_text(
                self._preserve_required_placeholders(seed_prompt.text, text)
            )
            if normalized == seed_prompt.text or normalized in seen_texts:
                continue
            variants.append(
                self._copy_prompt(
                    seed_prompt,
                    text=normalized,
                    metadata_updates={
                        "mutation_operator": recipe_name,
                        "mutation_base_weight": 2.0,
                        "parent_baseline_fitness": None,
                    },
                )
            )

        return variants

    def evolve(
        self,
        population: Population,
        fitness_scores: list[float],
    ) -> Population:
        """Generate the next generation through selection, crossover, and mutation.

        Args:
            population: Current population.
            fitness_scores: Fitness scores for each individual.

        Returns:
            New population for the next generation.
        """
        # Update fitness scores
        evaluated = []
        for ind, score in zip(population.individuals, fitness_scores, strict=True):
            evaluated.append(ind.with_fitness(score))

        # Sort by fitness (descending)
        evaluated.sort(key=lambda x: x.fitness or 0.0, reverse=True)

        new_individuals: list[Individual] = []
        next_gen = population.generation + 1

        # Elitism: preserve top individuals, but always leave room for
        # offspring when population size allows evolution.
        seen_texts: set[str] = set()
        effective_elitism = self.evo_config.elitism
        if len(population) > 1:
            effective_elitism = min(effective_elitism, len(population) - 1)

        for elite in evaluated[:effective_elitism]:
            seen_texts.add(elite.prompt.text)
            elite_individual = Individual(
                prompt=self._copy_prompt(elite.prompt),
                fitness=None,  # Will be re-evaluated
                generation=next_gen,
                parent_ids=[elite.id],
            )
            new_individuals.append(elite_individual)
            self._register_lineage_node(elite_individual, next_gen)

        # Generate offspring
        while len(new_individuals) < len(population):
            # Tournament selection
            parent1 = self._tournament_select(evaluated)
            parent2 = self._tournament_select(evaluated)

            # Crossover
            if random.random() < self._effective_crossover_rate:
                child1_prompt, child2_prompt = self._crossover(parent1.prompt, parent2.prompt)
            else:
                child1_prompt = self._copy_prompt(parent1.prompt)
                child2_prompt = self._copy_prompt(parent2.prompt)

            # Mutation
            if random.random() < self._effective_mutation_rate:
                child1_prompt = self._mutate_prompt(
                    child1_prompt,
                    parent_baseline_fitness=max(parent1.fitness or 0.0, parent2.fitness or 0.0),
                )
            if random.random() < self._effective_mutation_rate:
                child2_prompt = self._mutate_prompt(
                    child2_prompt,
                    parent_baseline_fitness=max(parent1.fitness or 0.0, parent2.fitness or 0.0),
                )

            child1_prompt = self._ensure_unique_child(child1_prompt, seen_texts)
            child2_prompt = self._ensure_unique_child(child2_prompt, seen_texts)

            # Add children
            seen_texts.add(child1_prompt.text)
            child1_individual = Individual(
                prompt=child1_prompt,
                generation=next_gen,
                parent_ids=[parent1.id, parent2.id],
            )
            new_individuals.append(child1_individual)
            self._register_lineage_node(child1_individual, next_gen)
            if len(new_individuals) < len(population):
                seen_texts.add(child2_prompt.text)
                child2_individual = Individual(
                    prompt=child2_prompt,
                    generation=next_gen,
                    parent_ids=[parent1.id, parent2.id],
                )
                new_individuals.append(child2_individual)
                self._register_lineage_node(child2_individual, next_gen)

        return Population(individuals=new_individuals, generation=next_gen)

    def record_generation_feedback(
        self,
        population: Population,
        fitness_scores: list[float],
    ) -> None:
        """Update mutation telemetry from an evaluated generation."""
        self._ensure_operator_stats_initialized()
        self._ensure_mvp3_components()

        seed_baseline = 0.0
        for individual, score in zip(population.individuals, fitness_scores, strict=True):
            if individual.prompt.metadata.get("is_seed"):
                seed_baseline = score
                # MVP 3: Set ablation baseline
                if self._ablation_tracker:
                    self._ablation_tracker.set_baseline(score)
                break

        generation_attempts: dict[str, int] = {}
        generation_wins: dict[str, int] = {}
        generation_delta: dict[str, float] = {}

        for individual, score in zip(population.individuals, fitness_scores, strict=True):
            operator_name = individual.prompt.metadata.get("mutation_operator")
            if not operator_name:
                continue

            baseline = individual.prompt.metadata.get("parent_baseline_fitness")
            if baseline is None and population.generation == 0:
                baseline = seed_baseline
            baseline = float(baseline or 0.0)
            delta = score - baseline

            stats = self._operator_stats.setdefault(
                operator_name,
                {
                    "attempts": 0.0,
                    "wins": 0.0,
                    "total_delta": 0.0,
                    "best_delta": 0.0,
                },
            )
            stats["attempts"] += 1.0
            stats["total_delta"] += delta
            stats["best_delta"] = max(stats["best_delta"], delta)
            if delta > 0.0:
                stats["wins"] += 1.0

            generation_attempts[operator_name] = generation_attempts.get(operator_name, 0) + 1
            generation_delta[operator_name] = generation_delta.get(operator_name, 0.0) + delta
            if delta > 0.0:
                generation_wins[operator_name] = generation_wins.get(operator_name, 0) + 1

            # MVP 3: Record to ablation tracker
            if self._ablation_tracker:
                self._ablation_tracker.record_mutation(
                    operator_name=operator_name,
                    parent_fitness=baseline,
                    child_fitness=score,
                )

            # MVP 3: Update diversity controller with fitness
            if self._diversity_controller:
                node = self._diversity_controller.get_lineage(individual.id)
                if node:
                    node.fitness = score

        if self.evo_config.adaptive_mutation_weights:
            self._recompute_operator_weights()

        # MVP 3: Record generation-level ablation stats
        if self._ablation_tracker:
            best_fitness = max(fitness_scores) if fitness_scores else 0.0
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
            self._ablation_tracker.record_generation(
                generation=population.generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                operator_counts=generation_attempts,
            )

        # MVP 3: Track diversity metrics
        if self._diversity_controller:
            self._last_diversity_metrics = self._diversity_controller.measure_diversity(
                population
            ).to_dict()

        if self._mutation_schedule and self._schedule_state is not None:
            if self._last_diversity_metrics:
                self._schedule_state.diversity_score = float(
                    self._last_diversity_metrics.get("unique_ratio", 1.0)
                )
            self._schedule_state = self._mutation_schedule.update(self._schedule_state, fitness_scores)
            self._effective_mutation_rate = self._mutation_schedule.get_mutation_rate(
                self._schedule_state
            )
            self._effective_crossover_rate = self._mutation_schedule.get_crossover_rate(
                self._schedule_state
            )
            if self._schedule_state.operator_weights:
                for name, weight in self._schedule_state.operator_weights.items():
                    if name in self._operator_current_weights:
                        self._operator_current_weights[name] = weight

        self._last_generation_summary = {}
        for operator_name, attempts in generation_attempts.items():
            wins = generation_wins.get(operator_name, 0)
            avg_delta = generation_delta[operator_name] / attempts
            self._last_generation_summary[operator_name] = {
                "attempts": float(attempts),
                "wins": float(wins),
                "avg_delta": avg_delta,
                "current_weight": self._operator_current_weights.get(
                    operator_name,
                    self._operator_base_weights.get(operator_name, 1.0),
                ),
            }

        if self._schedule_state is not None:
            self._last_generation_summary["__schedule__"] = {
                "mutation_rate": self._effective_mutation_rate,
                "crossover_rate": self._effective_crossover_rate,
                "stall_count": float(self._schedule_state.stall_count),
                "diversity_score": float(self._schedule_state.diversity_score),
            }

    def record_stage_feedback(
        self,
        population: Population,
        stage_feedback: dict[str, dict[str, Any]],
    ) -> None:
        """Record dominant stage failures so later mutations can react to them."""
        del population
        self._stage_feedback_by_prompt = dict(stage_feedback)

    def get_operator_stats(self) -> dict[str, dict[str, float]]:
        """Return aggregate operator performance metrics."""
        self._ensure_operator_stats_initialized()
        stats: dict[str, dict[str, float]] = {}
        for operator in self._get_mutation_operators():
            raw = self._operator_stats.get(operator.name, {})
            attempts = raw.get("attempts", 0.0)
            wins = raw.get("wins", 0.0)
            avg_delta = raw.get("total_delta", 0.0) / attempts if attempts else 0.0
            stats[operator.name] = {
                "base_weight": self._operator_base_weights[operator.name],
                "current_weight": self._operator_current_weights[operator.name],
                "attempts": attempts,
                "wins": wins,
                "win_rate": wins / attempts if attempts else 0.0,
                "avg_delta": avg_delta,
                "best_delta": raw.get("best_delta", 0.0),
            }
        return stats

    def get_last_generation_summary(self) -> dict[str, dict[str, float]]:
        """Return telemetry captured for the most recent evaluated generation."""
        return self._last_generation_summary.copy()

    # =========================================================================
    # MVP 3: Diversity and Lineage Methods
    # =========================================================================

    def get_diversity_metrics(self) -> dict[str, Any]:
        """Return current population diversity metrics.

        Returns:
            Dictionary with diversity metrics or empty if not tracking.
        """
        return self._last_diversity_metrics.copy()

    def get_schedule_state(self) -> dict[str, Any]:
        """Return current adaptive schedule state."""
        if self._schedule_state is None:
            return {}
        state: dict[str, Any] = self._schedule_state.to_dict()
        state["effective_mutation_rate"] = self._effective_mutation_rate
        state["effective_crossover_rate"] = self._effective_crossover_rate
        return state

    def get_detected_task_type(self) -> str | None:
        """Return the detected task type from the seed prompt.

        Returns:
            Task type string or None if not detected.
        """
        return self._detected_task_type

    def get_detected_output_mode(self) -> str | None:
        """Return the detected output mode from the seed prompt.

        Returns:
            Output mode string or None if not detected.
        """
        return self._detected_output_mode

    def get_lineage_report(self, individual: Individual) -> dict[str, Any]:
        """Generate lineage report for an individual.

        Args:
            individual: The individual to report on.

        Returns:
            Lineage report dictionary.
        """
        if not self._diversity_controller:
            return {"error": "Diversity control not enabled"}

        result: dict[str, Any] = self._diversity_controller.generate_lineage_report(
            individual
        )
        return result

    def get_ablation_result(self) -> dict[str, Any] | None:
        """Get ablation analysis result.

        Returns:
            Ablation result dictionary or None if not tracking.
        """
        if not self._ablation_tracker:
            return None

        result = self._ablation_tracker.generate_result()
        ablation_dict: dict[str, Any] = result.to_dict()
        return ablation_dict

    def get_ablation_summary(self) -> str:
        """Get human-readable ablation summary.

        Returns:
            Formatted summary string.
        """
        if not self._ablation_tracker:
            return "Ablation tracking not enabled"

        summary: str = self._ablation_tracker.get_summary()
        return summary

    def apply_crowding_penalty(
        self,
        population: Population,
        fitness_scores: list[float],
    ) -> list[float]:
        """Apply crowding penalty to fitness scores for diversity.

        Args:
            population: Current population.
            fitness_scores: Original fitness scores.

        Returns:
            Adjusted fitness scores.
        """
        if not self._diversity_controller:
            return fitness_scores

        adjusted: list[float] = self._diversity_controller.apply_crowding_penalty(
            population,
            fitness_scores,
            penalty_factor=self.evo_config.crowding_penalty,
        )
        return adjusted

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Serialize adaptive operator state for checkpointing."""
        self._ensure_operator_stats_initialized()
        return {
            "operator_base_weights": self._operator_base_weights.copy(),
            "operator_current_weights": self._operator_current_weights.copy(),
            "operator_stats": {
                name: stats.copy() for name, stats in self._operator_stats.items()
            },
            "last_generation_summary": {
                name: stats.copy() for name, stats in self._last_generation_summary.items()
            },
            "effective_mutation_rate": self._effective_mutation_rate,
            "effective_crossover_rate": self._effective_crossover_rate,
            "schedule_state": self._schedule_state.to_dict() if self._schedule_state else None,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore adaptive operator state from a checkpoint payload."""
        self._ensure_operator_stats_initialized()
        self._operator_base_weights.update(state.get("operator_base_weights", {}))
        self._operator_current_weights.update(state.get("operator_current_weights", {}))

        for name, stats in state.get("operator_stats", {}).items():
            self._operator_stats[name] = {
                "attempts": float(stats.get("attempts", 0.0)),
                "wins": float(stats.get("wins", 0.0)),
                "total_delta": float(stats.get("total_delta", 0.0)),
                "best_delta": float(stats.get("best_delta", 0.0)),
            }

        self._last_generation_summary = {
            name: {key: float(value) for key, value in stats.items()}
            for name, stats in state.get("last_generation_summary", {}).items()
        }
        self._effective_mutation_rate = float(
            state.get("effective_mutation_rate", self.evo_config.mutation_rate)
        )
        self._effective_crossover_rate = float(
            state.get("effective_crossover_rate", self.evo_config.crossover_rate)
        )
        schedule_state = state.get("schedule_state")
        if schedule_state and self.evo_config.use_adaptive_schedule:
            self._ensure_mvp3_components()
            if self._schedule_state is not None:
                self._schedule_state.generation = int(schedule_state.get("generation", 0))
                self._schedule_state.max_generations = int(
                    schedule_state.get("max_generations", self.evo_config.max_generations)
                )
                self._schedule_state.current_mutation_rate = float(
                    schedule_state.get("mutation_rate", self.evo_config.mutation_rate)
                )
                self._schedule_state.current_crossover_rate = float(
                    schedule_state.get("crossover_rate", self.evo_config.crossover_rate)
                )
                self._schedule_state.stall_count = int(schedule_state.get("stall_count", 0))
                self._schedule_state.best_fitness = float(schedule_state.get("best_fitness", 0.0))
                self._schedule_state.avg_fitness = float(schedule_state.get("avg_fitness", 0.0))
                self._schedule_state.diversity_score = float(
                    schedule_state.get("diversity_score", 1.0)
                )
                self._schedule_state.operator_weights = {
                    key: float(value)
                    for key, value in schedule_state.get("operator_weights", {}).items()
                }

    def _tournament_select(self, evaluated: list[Individual]) -> Individual:
        """Select an individual using tournament selection.

        Args:
            evaluated: List of evaluated individuals.

        Returns:
            Selected individual.
        """
        tournament = random.sample(evaluated, min(self.evo_config.tournament_size, len(evaluated)))
        return max(tournament, key=lambda x: x.fitness or 0.0)

    def _crossover(self, parent1: Prompt, parent2: Prompt) -> tuple[Prompt, Prompt]:
        """Perform single-point crossover on prompt texts.

        Args:
            parent1: First parent prompt.
            parent2: Second parent prompt.

        Returns:
            Tuple of two offspring prompts.
        """
        text1 = parent1.text
        text2 = parent2.text

        # Split by sentences for more meaningful crossover
        import re

        sentences1 = re.split(r"(?<=[.!?])\s+", text1)
        sentences2 = re.split(r"(?<=[.!?])\s+", text2)

        if len(sentences1) > 1 and len(sentences2) > 1:
            # Single-point crossover at sentence level
            point1 = random.randint(1, len(sentences1) - 1)
            point2 = random.randint(1, len(sentences2) - 1)

            child1_text = " ".join(sentences1[:point1] + sentences2[point2:])
            child2_text = " ".join(sentences2[:point2] + sentences1[point1:])
        else:
            # Word-level crossover for short prompts
            words1 = text1.split()
            words2 = text2.split()

            if len(words1) > 2 and len(words2) > 2:
                point = random.randint(1, min(len(words1), len(words2)) - 1)
                child1_text = " ".join(words1[:point] + words2[point:])
                child2_text = " ".join(words2[:point] + words1[point:])
            else:
                # Too short, just swap
                child1_text = text2
                child2_text = text1

        return (
            self._copy_prompt(
                parent1,
                text=self._preserve_required_placeholders(parent1.text, child1_text),
            ),
            self._copy_prompt(
                parent2,
                text=self._preserve_required_placeholders(parent2.text, child2_text),
            ),
        )

    def _mutate_prompt(
        self,
        prompt: Prompt,
        parent_baseline_fitness: float | None = None,
    ) -> Prompt:
        """Apply a random mutation to a prompt.

        Available mutations:
        - Rephrase the instruction in semantically similar ways
        - Add output-format constraints
        - Promote the prompt into a structured input/output layout
        - Add task-specific label or numeric answer constraints
        - Remove filler language

        Args:
            prompt: The prompt to mutate.

        Returns:
            Mutated prompt.
        """
        operators = self._get_mutation_operators()
        remaining = operators.copy()

        while remaining:
            mutation = self._pick_mutation_operator(remaining, prompt)
            remaining.remove(mutation)
            new_text = self._normalize_prompt_text(
                self._preserve_required_placeholders(
                    prompt.text,
                    mutation.transform(prompt.text),
                )
            )
            if new_text != prompt.text:
                return self._copy_prompt(
                    prompt,
                    text=new_text,
                    metadata_updates={
                        "mutation_operator": mutation.name,
                        "mutation_base_weight": mutation.weight,
                        "parent_baseline_fitness": parent_baseline_fitness,
                    },
                )

        return self._copy_prompt(
            prompt,
            text=self._preserve_required_placeholders(
                prompt.text,
                self._force_non_noop_mutation(prompt.text),
            ),
            metadata_updates={
                "mutation_operator": "forced_constraint",
                "mutation_base_weight": 0.5,
                "parent_baseline_fitness": parent_baseline_fitness,
            },
        )

    def _get_mutation_operators(self) -> list[MutationOperator]:
        """Return the current mutation operator library.

        The operator set is intentionally semantic rather than syntactic-noise based.
        """
        operators = [
            MutationOperator("rephrase_instruction", 1.0, self._mutate_rephrase),
            MutationOperator("add_output_constraint", 1.5, self._mutate_add_constraint),
            MutationOperator(
                "add_answer_only_directive",
                1.8,
                self._mutate_add_answer_only_directive,
            ),
            MutationOperator(
                "add_numeric_constraint",
                1.6,
                self._mutate_add_numeric_constraint,
            ),
            MutationOperator(
                "add_label_constraint",
                1.6,
                self._mutate_add_label_constraint,
            ),
            MutationOperator(
                "add_verification_directive",
                1.2,
                self._mutate_add_verification_directive,
            ),
            MutationOperator(
                "promote_structured_layout",
                1.7,
                self._mutate_promote_structured_layout,
            ),
            MutationOperator("remove_filler", 0.8, self._mutate_remove_word),
        ]
        return operators

    def _pick_mutation_operator(
        self,
        operators: list[MutationOperator],
        prompt: Prompt | None = None,
    ) -> MutationOperator:
        """Pick a mutation operator with weighted random choice."""
        self._ensure_operator_stats_initialized()
        weights = [
            self._get_operator_selection_weight(operator, prompt)
            for operator in operators
        ]
        return random.choices(operators, weights=weights, k=1)[0]

    def _get_operator_selection_weight(
        self,
        operator: MutationOperator,
        prompt: Prompt | None = None,
    ) -> float:
        """Get the current stage-aware selection weight for an operator."""
        base_weight = self._operator_current_weights.get(operator.name, operator.weight)
        if prompt is None:
            return base_weight

        stage_info = self._stage_feedback_by_prompt.get(prompt.id, {})
        dominant_stage = str(stage_info.get("dominant_stage", "")).lower()
        if not dominant_stage:
            return base_weight

        boosts: dict[str, float] = {}
        if "json" in dominant_stage:
            boosts = {
                "promote_structured_layout": 2.2,
                "add_output_constraint": 1.8,
                "add_answer_only_directive": 1.5,
            }
        elif "schema" in dominant_stage or "field" in dominant_stage:
            boosts = {
                "promote_structured_layout": 2.0,
                "add_output_constraint": 1.7,
                "add_verification_directive": 1.4,
            }
        elif "quality" in dominant_stage or "value" in dominant_stage:
            boosts = {
                "add_verification_directive": 1.8,
                "remove_filler": 1.4,
                "rephrase_instruction": 1.2,
            }

        return base_weight * boosts.get(operator.name, 1.0)

    def _ensure_operator_stats_initialized(self) -> None:
        """Initialize telemetry and adaptive weights for known operators."""
        for operator in self._get_mutation_operators():
            self._operator_base_weights.setdefault(operator.name, operator.weight)
            self._operator_current_weights.setdefault(operator.name, operator.weight)
            self._operator_stats.setdefault(
                operator.name,
                {
                    "attempts": 0.0,
                    "wins": 0.0,
                    "total_delta": 0.0,
                    "best_delta": 0.0,
                },
            )

    def _register_lineage_node(self, individual: Individual, generation: int) -> None:
        """Register an individual with the diversity controller when lineage is enabled."""
        if not self._diversity_controller:
            return

        self._diversity_controller.register_prompt(
            prompt_id=individual.id,
            text=individual.prompt.text,
            generation=generation,
            parent_ids=individual.parent_ids,
            mutation_operator=individual.prompt.metadata.get("mutation_operator"),
        )

    def _recompute_operator_weights(self) -> None:
        """Adjust operator weights using observed win rate and score deltas."""
        for operator in self._get_mutation_operators():
            stats = self._operator_stats.get(operator.name, {})
            attempts = stats.get("attempts", 0.0)
            wins = stats.get("wins", 0.0)
            avg_delta = stats.get("total_delta", 0.0) / attempts if attempts else 0.0
            win_rate = wins / attempts if attempts else 0.0
            multiplier = 1.0 + (win_rate * self.evo_config.weight_learning_rate)
            if avg_delta > 0.0:
                multiplier += avg_delta * self.evo_config.weight_learning_rate
            elif avg_delta < 0.0:
                multiplier /= 1.0 + abs(avg_delta) * self.evo_config.weight_learning_rate

            self._operator_current_weights[operator.name] = max(
                self.evo_config.min_operator_weight,
                self._operator_base_weights[operator.name] * multiplier,
            )

    def _copy_prompt(
        self,
        prompt: Prompt,
        text: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> Prompt:
        """Clone a prompt while clearing stale mutation telemetry."""
        metadata = {
            key: value
            for key, value in prompt.metadata.items()
            if key not in {"mutation_operator", "mutation_base_weight", "parent_baseline_fitness"}
        }
        if text is not None and text != prompt.text:
            metadata["parent_id"] = prompt.id
        if metadata_updates:
            metadata.update(metadata_updates)
        return Prompt(text=text or prompt.text, metadata=metadata)

    def _normalize_prompt_text(self, text: str) -> str:
        """Normalize whitespace while preserving deliberate line structure."""
        lines = [" ".join(line.split()) for line in text.splitlines()]
        compact_lines = [line for line in lines if line]
        return "\n".join(compact_lines).strip()

    def _preserve_required_placeholders(self, source_text: str, candidate_text: str) -> str:
        """Ensure structural placeholders from the source prompt survive transformations."""
        required_placeholders = ["{input}"]
        updated = candidate_text

        for placeholder in required_placeholders:
            if placeholder in source_text and placeholder not in updated:
                if "Input:" in updated or "Question:" in updated or "Task Input:" in updated:
                    updated = updated.rstrip() + f"\n{placeholder}"
                else:
                    updated = updated.rstrip() + f"\nInput: {placeholder}"

        return updated

    def _append_first_missing_clause(self, text: str, clauses: list[str]) -> str:
        """Append the first directive not already present in the prompt."""
        normalized = text.lower()
        for clause in clauses:
            clause_text = clause.strip().lower()
            if clause_text not in normalized:
                return text.rstrip() + clause
        return text

    def _contains_any(self, text: str, terms: list[str]) -> bool:
        """Return true when the text contains any of the provided terms."""
        lowered = text.lower()
        return any(term in lowered for term in terms)

    def _extract_instruction_stem(self, text: str) -> str:
        """Extract the task instruction from a prompt while preserving meaning."""
        stem = text.replace("{input}", " ")
        stem = re.sub(r"\s+", " ", stem).strip()
        stem = re.sub(r"[:\-]\s*([.!?])", r"\1", stem)
        stem = re.sub(r"\s+([.!?])", r"\1", stem)
        stem = re.sub(r"[:\-\s]+$", "", stem)
        return stem or "Complete the task below"

    def _create_unique_prompt(self, prompt: Prompt, seen_texts: set[str]) -> Prompt:
        """Create a prompt variant with text not already present in the population."""
        candidate = prompt
        for _ in range(8):
            candidate = self._mutate_prompt(prompt)
            if candidate.text not in seen_texts:
                return candidate

        return prompt.with_text(self._force_non_noop_mutation(prompt.text, seen_texts))

    def _ensure_unique_child(self, prompt: Prompt, seen_texts: set[str]) -> Prompt:
        """Ensure offspring prompt text is unique within the next generation."""
        if prompt.text not in seen_texts:
            return prompt

        for _ in range(8):
            candidate = self._mutate_prompt(prompt)
            if candidate.text not in seen_texts:
                return candidate

        return prompt.with_text(self._force_non_noop_mutation(prompt.text, seen_texts))

    def _force_non_noop_mutation(
        self,
        text: str,
        seen_texts: set[str] | None = None,
    ) -> str:
        """Append the first missing constraint to guarantee a text change."""
        constraints = [
            " Respond with only the final answer.",
            " Do not include any explanation.",
            " Use the exact required output format.",
            " If the answer is numeric, return only the number.",
            " If the task is classification, return only the label.",
            " Verify the result once before answering.",
        ]

        used_texts = seen_texts or set()
        for constraint in constraints:
            candidate = text.rstrip() + constraint
            if candidate != text and candidate not in used_texts:
                return candidate

        suffix = 2
        while True:
            candidate = f"{text.rstrip()} Variant {suffix}."
            if candidate not in used_texts:
                return candidate
            suffix += 1

    def _mutate_rephrase(self, text: str) -> str:
        """Rephrase by substituting common instruction words."""
        substitutions = {
            "Answer the question": ["Solve the task", "Answer the prompt", "Determine the answer"],
            "Classify": ["Categorize", "Label", "Determine", "Identify"],
            "Determine": ["Figure out", "Identify", "Establish", "Ascertain"],
            "Output": ["Return", "Respond with", "Provide", "Give"],
            "Answer": ["Respond to", "Solve", "Return the result for"],
            "the following": ["this", "the given", "the provided"],
            "Please": ["", "Kindly", ""],
            "must": ["should", "need to", "have to"],
        }

        for original, replacements in substitutions.items():
            if original.lower() in text.lower():
                replacement = random.choice(replacements)
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                text = pattern.sub(replacement, text, count=1)
                break

        return text

    def _mutate_add_constraint(self, text: str) -> str:
        """Add a constraint or clarification."""
        constraints = [
            " Use the exact required format.",
            " Keep the response to a single line.",
            " Do not add commentary.",
            " Preserve the meaning of the input.",
            " Be precise and accurate.",
        ]

        return self._append_first_missing_clause(text, constraints)

    def _mutate_add_answer_only_directive(self, text: str) -> str:
        """Add a directive that suppresses verbose explanations."""
        directives = [
            " Respond with only the final answer.",
            " Return only the final answer with no explanation.",
            " Output only the answer and nothing else.",
            " Give only the final answer. No explanation or extra text.",
        ]
        return self._append_first_missing_clause(text, directives)

    def _mutate_add_numeric_constraint(self, text: str) -> str:
        """Add a numeric-only response constraint for arithmetic-like prompts."""
        numeric_triggers = [
            "how many",
            "minus",
            "plus",
            "sum",
            "difference",
            "count",
            "number",
            "calculate",
            "math",
            "letters",
            "what is",
        ]
        if not self._contains_any(text, numeric_triggers):
            return text

        directives = [
            " Return only the number.",
            " If the answer is numeric, output digits only with no words or units.",
            " Provide the final numeric result only. No words, units, or explanation.",
        ]
        return self._append_first_missing_clause(text, directives)

    def _mutate_add_label_constraint(self, text: str) -> str:
        """Add a label-only response constraint for classification prompts."""
        label_triggers = [
            "classify",
            "classification",
            "sentiment",
            "label",
            "category",
            "positive",
            "negative",
            "neutral",
        ]
        if not self._contains_any(text, label_triggers):
            return text

        directives = [
            " Return exactly one label.",
            " Answer with one label only and no punctuation.",
            " Output only the classification label. No explanation.",
        ]
        return self._append_first_missing_clause(text, directives)

    def _mutate_add_verification_directive(self, text: str) -> str:
        """Encourage answer verification without asking for visible reasoning."""
        directives = [
            " Verify the result silently before answering.",
            " Double-check the final answer before responding.",
            " Check the answer once, then return only the result.",
        ]
        return self._append_first_missing_clause(text, directives)

    def _mutate_promote_structured_layout(self, text: str) -> str:
        """Convert a flat prompt into a clearer instruction/input/output layout."""
        if "{input}" not in text:
            return text

        stem = self._extract_instruction_stem(text)
        lead = stem if stem.endswith((".", "!", "?")) else f"{stem}."
        candidates = [
            f"{lead}\nInput: {{input}}\nReturn only the final answer.",
            f"{lead}\nQuestion: {{input}}\nAnswer with only the final answer.",
            f"{lead}\nTask Input: {{input}}\nOutput: only the final answer.",
        ]

        for candidate in candidates:
            if self._normalize_prompt_text(candidate) != self._normalize_prompt_text(text):
                return candidate

        return text

    def _mutate_remove_word(self, text: str) -> str:
        """Remove a random non-essential word."""
        removable = ["please", "kindly", "just", "simply", "actually", "basically"]

        for word in removable:
            if word in text.lower():
                pattern = re.compile(r"\b" + re.escape(word) + r"\b\s*", re.IGNORECASE)
                text = pattern.sub("", text, count=1)
                break

        return text.strip()

"""Tests for MVP 3 mutation schedules module."""

from __future__ import annotations

import pytest

from promptfoundry.strategies.schedules import (
    AdaptiveSchedule,
    ConstantSchedule,
    LinearDecaySchedule,
    MutationScheduleState,
    OperatorAdaptiveSchedule,
    SchedulePhase,
    create_schedule,
)


class TestMutationScheduleState:
    """Test schedule state dataclass."""

    def test_state_defaults(self) -> None:
        """Test default state values."""
        state = MutationScheduleState()
        assert state.generation == 0
        assert state.current_mutation_rate == 0.3
        assert state.phase == SchedulePhase.EXPLORATION

    def test_progress_ratio(self) -> None:
        """Test progress ratio calculation."""
        state = MutationScheduleState(generation=5, max_generations=10)
        assert state.progress_ratio() == 0.5

        state = MutationScheduleState(generation=10, max_generations=10)
        assert state.progress_ratio() == 1.0

    def test_progress_ratio_zero_max(self) -> None:
        """Test progress ratio with zero max generations."""
        state = MutationScheduleState(generation=5, max_generations=0)
        assert state.progress_ratio() == 0.0

    def test_state_to_dict(self) -> None:
        """Test state serialization."""
        state = MutationScheduleState(
            generation=3,
            best_fitness=0.85,
            stall_count=2,
        )
        result = state.to_dict()
        assert result["generation"] == 3
        assert result["best_fitness"] == 0.85
        assert result["stall_count"] == 2


class TestConstantSchedule:
    """Test constant mutation schedule."""

    def test_constant_rates(self) -> None:
        """Test rates remain constant."""
        schedule = ConstantSchedule(mutation_rate=0.4, crossover_rate=0.6)
        state = MutationScheduleState()

        assert schedule.get_mutation_rate(state) == 0.4
        assert schedule.get_crossover_rate(state) == 0.6

        # After update, rates should stay the same
        state = schedule.update(state, [0.5, 0.6, 0.7])
        assert schedule.get_mutation_rate(state) == 0.4
        assert schedule.get_crossover_rate(state) == 0.6

    def test_update_tracks_fitness(self) -> None:
        """Test update tracks best fitness."""
        schedule = ConstantSchedule()
        state = MutationScheduleState()

        state = schedule.update(state, [0.5, 0.7, 0.6])
        assert state.best_fitness == 0.7
        assert state.avg_fitness == pytest.approx(0.6, rel=0.01)

    def test_update_tracks_stall(self) -> None:
        """Test stall count tracking."""
        schedule = ConstantSchedule()
        state = MutationScheduleState()

        # First update - no stall
        state = schedule.update(state, [0.5, 0.7])
        assert state.stall_count == 0

        # No improvement - stall
        state = schedule.update(state, [0.5, 0.6])
        assert state.stall_count == 1

        # Still no improvement
        state = schedule.update(state, [0.5, 0.5])
        assert state.stall_count == 2

        # Improvement - reset stall
        state = schedule.update(state, [0.8, 0.9])
        assert state.stall_count == 0


class TestLinearDecaySchedule:
    """Test linear decay mutation schedule."""

    def test_decay_over_generations(self) -> None:
        """Test mutation rate decays linearly."""
        schedule = LinearDecaySchedule(
            initial_mutation_rate=0.5,
            final_mutation_rate=0.1,
        )
        state = MutationScheduleState(generation=0, max_generations=10)

        # At start
        assert schedule.get_mutation_rate(state) == 0.5

        # At midpoint
        state = MutationScheduleState(generation=5, max_generations=10)
        assert schedule.get_mutation_rate(state) == pytest.approx(0.3, rel=0.01)

        # At end
        state = MutationScheduleState(generation=10, max_generations=10)
        assert schedule.get_mutation_rate(state) == pytest.approx(0.1, rel=0.01)

    def test_phase_progression(self) -> None:
        """Test phase changes over generations."""
        schedule = LinearDecaySchedule()
        state = MutationScheduleState(generation=0, max_generations=9)

        # Early - exploration
        state = schedule.update(state, [0.5])
        assert state.phase == SchedulePhase.EXPLORATION

        # Middle - balanced
        state.generation = 3
        state = schedule.update(state, [0.6])
        assert state.phase == SchedulePhase.BALANCED

        # Late - exploitation
        state.generation = 7
        state = schedule.update(state, [0.7])
        assert state.phase == SchedulePhase.EXPLOITATION


class TestAdaptiveSchedule:
    """Test adaptive mutation schedule."""

    def test_adaptive_initialization(self) -> None:
        """Test adaptive schedule initialization."""
        schedule = AdaptiveSchedule(
            base_mutation_rate=0.3,
            min_mutation_rate=0.1,
            max_mutation_rate=0.7,
        )
        state = MutationScheduleState()

        # Should use base rate initially
        assert state.current_mutation_rate == 0.3

    def test_increase_on_stall(self) -> None:
        """Test mutation rate increases when stalled."""
        schedule = AdaptiveSchedule(
            base_mutation_rate=0.3,
            stall_threshold=3,
            adaptation_rate=0.2,
        )
        state = MutationScheduleState(
            generation=5,
            stall_count=4,  # Above threshold
            current_mutation_rate=0.3,
            best_fitness=0.6,  # Set to a value higher than scores below
        )

        state = schedule.update(state, [0.5, 0.5, 0.5])  # No improvement over best

        # Should be in converged phase and increase mutation
        assert state.phase == SchedulePhase.CONVERGED
        assert state.current_mutation_rate > 0.3

    def test_decrease_on_exploitation(self) -> None:
        """Test mutation rate decreases in exploitation phase."""
        schedule = AdaptiveSchedule(
            base_mutation_rate=0.4,
            adaptation_rate=0.2,
        )
        state = MutationScheduleState(
            generation=15,
            max_generations=20,
            current_mutation_rate=0.4,
            stall_count=0,
        )

        # Update in late phase
        state = schedule.update(state, [0.9])

        # Should be in exploitation and lowering mutation
        assert state.phase == SchedulePhase.EXPLOITATION
        assert state.current_mutation_rate < 0.4

    def test_respects_min_max_bounds(self) -> None:
        """Test mutation rate stays within bounds."""
        schedule = AdaptiveSchedule(
            min_mutation_rate=0.1,
            max_mutation_rate=0.7,
        )

        # Very low diversity should push to max
        state = MutationScheduleState(
            diversity_score=0.2,
            stall_count=10,
            current_mutation_rate=0.6,
        )
        state = schedule.update(state, [0.3])
        assert state.current_mutation_rate <= 0.7

        # In exploitation, should respect minimum
        state = MutationScheduleState(
            generation=19,
            max_generations=20,
            stall_count=0,
            current_mutation_rate=0.15,
        )
        state = schedule.update(state, [0.95])
        assert state.current_mutation_rate >= 0.1


class TestOperatorAdaptiveSchedule:
    """Test operator-adaptive schedule."""

    def test_operator_weight_tracking(self) -> None:
        """Test operator weight updates based on results."""
        schedule = OperatorAdaptiveSchedule()

        # Record positive results for operator A
        schedule.record_operator_result("operator_a", 0.1, True)
        schedule.record_operator_result("operator_a", 0.15, True)
        schedule.record_operator_result("operator_a", 0.05, True)

        # Record negative results for operator B
        schedule.record_operator_result("operator_b", -0.1, False)
        schedule.record_operator_result("operator_b", -0.05, False)
        schedule.record_operator_result("operator_b", -0.08, False)

        weights = schedule.get_operator_weights()

        # Operator A should have higher weight
        assert weights["operator_a"] > 1.0
        # Operator B should have lower weight
        assert weights["operator_b"] < 1.0

    def test_insufficient_data_default_weight(self) -> None:
        """Test default weight when insufficient data."""
        schedule = OperatorAdaptiveSchedule()

        # Only 2 attempts - not enough
        schedule.record_operator_result("operator_a", 0.1, True)
        schedule.record_operator_result("operator_a", 0.1, True)

        weights = schedule.get_operator_weights()
        assert weights.get("operator_a", 1.0) == 1.0

    def test_weights_in_state(self) -> None:
        """Test weights are included in state update."""
        schedule = OperatorAdaptiveSchedule()
        schedule.record_operator_result("op_1", 0.1, True)
        schedule.record_operator_result("op_1", 0.1, True)
        schedule.record_operator_result("op_1", 0.1, True)

        state = MutationScheduleState()
        state = schedule.update(state, [0.5])

        assert "op_1" in state.operator_weights


class TestCreateSchedule:
    """Test schedule factory function."""

    def test_create_constant(self) -> None:
        """Test creating constant schedule."""
        schedule = create_schedule("constant", mutation_rate=0.25)
        assert isinstance(schedule, ConstantSchedule)
        assert schedule.mutation_rate == 0.25

    def test_create_linear(self) -> None:
        """Test creating linear decay schedule."""
        schedule = create_schedule(
            "linear",
            initial_mutation_rate=0.5,
            final_mutation_rate=0.1,
        )
        assert isinstance(schedule, LinearDecaySchedule)

    def test_create_adaptive(self) -> None:
        """Test creating adaptive schedule."""
        schedule = create_schedule("adaptive", stall_threshold=5)
        assert isinstance(schedule, AdaptiveSchedule)

    def test_create_operator(self) -> None:
        """Test creating operator-adaptive schedule."""
        schedule = create_schedule("operator", operator_learning_rate=0.3)
        assert isinstance(schedule, OperatorAdaptiveSchedule)

    def test_invalid_type(self) -> None:
        """Test error on invalid schedule type."""
        with pytest.raises(ValueError):
            create_schedule("invalid_type")

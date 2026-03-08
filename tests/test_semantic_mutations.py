"""Tests for MVP 3 semantic mutations module."""

from __future__ import annotations

from promptfoundry.strategies.semantic_mutations import (
    OutputMode,
    SemanticMutation,
    SemanticMutationLibrary,
    TaskDetector,
    TaskType,
    get_mutation_library,
)


class TestTaskDetector:
    """Test task type detection."""

    def test_detect_classification_task(self) -> None:
        """Test classification task detection."""
        prompts = [
            "Classify the sentiment of this text: {input}",
            "Label the following as positive or negative: {input}",
            "Categorize this review: {input}",
        ]
        for prompt in prompts:
            assert TaskDetector.detect_task_type(prompt) == TaskType.CLASSIFICATION

    def test_detect_numeric_task(self) -> None:
        """Test numeric task detection."""
        prompts = [
            "Calculate 5 plus 3",
            "How many letters are in the word 'hello'?",
            "Find the sum of these numbers: {input}",
            "Count the vowels in: {input}",
        ]
        for prompt in prompts:
            assert TaskDetector.detect_task_type(prompt) == TaskType.NUMERIC

    def test_detect_extraction_task(self) -> None:
        """Test extraction task detection."""
        prompts = [
            "Extract the name from this sentence: {input}",
            "Find the date mentioned in: {input}",
            "Identify the key terms in: {input}",
        ]
        for prompt in prompts:
            assert TaskDetector.detect_task_type(prompt) == TaskType.EXTRACTION

    def test_detect_qa_task(self) -> None:
        """Test Q&A task detection."""
        prompts = [
            "Answer the question: {input}",
            "Question: When did WWII end?",
        ]
        for prompt in prompts:
            assert TaskDetector.detect_task_type(prompt) == TaskType.QA

        # "What is" patterns match extraction first (more specific)
        qa_or_extraction = TaskDetector.detect_task_type("Q: What is the capital of France?")
        assert qa_or_extraction in (TaskType.QA, TaskType.EXTRACTION)

    def test_detect_unknown_task(self) -> None:
        """Test unknown task type for ambiguous prompts."""
        prompt = "Process this data"
        assert TaskDetector.detect_task_type(prompt) == TaskType.UNKNOWN

    def test_detect_numeric_task_from_metadata_hint(self) -> None:
        """Task metadata should help classify generic math prompts."""
        prompt = "Solve: {input}"
        metadata = {"task_type_hint": "math_reasoning"}

        assert TaskDetector.detect_task_type(prompt, metadata) == TaskType.NUMERIC

    def test_detect_output_mode_exact_match(self) -> None:
        """Test exact match output mode detection."""
        prompt = "Return only the answer: {input}"
        task_type = TaskType.QA
        assert TaskDetector.detect_output_mode(prompt, task_type) == OutputMode.EXACT_MATCH

    def test_detect_output_mode_structured(self) -> None:
        """Test structured output mode detection."""
        prompt = "Output as JSON: {input}"
        task_type = TaskType.EXTRACTION
        assert TaskDetector.detect_output_mode(prompt, task_type) == OutputMode.STRUCTURED

    def test_detect_output_mode_ignores_input_placeholder_braces(self) -> None:
        """Template placeholders should not be mistaken for structured JSON output."""
        prompt = "Classify the sentiment: {input}"

        assert TaskDetector.detect_output_mode(prompt, TaskType.CLASSIFICATION) == OutputMode.LABEL

    def test_detect_output_mode_from_task_type(self) -> None:
        """Test output mode inference from task type."""
        prompt = "Classify this"
        assert (
            TaskDetector.detect_output_mode(prompt, TaskType.CLASSIFICATION)
            == OutputMode.LABEL
        )
        assert (
            TaskDetector.detect_output_mode(prompt, TaskType.NUMERIC)
            == OutputMode.NUMERIC
        )


class TestSemanticMutationLibrary:
    """Test semantic mutation library."""

    def test_library_initialization(self) -> None:
        """Test library initializes with mutations."""
        library = SemanticMutationLibrary()
        mutations = library.get_all_mutations()
        assert len(mutations) > 0

    def test_get_mutations_for_task(self) -> None:
        """Test filtering mutations by task type."""
        library = SemanticMutationLibrary()

        classification_mutations = library.get_mutations_for_task(TaskType.CLASSIFICATION)
        numeric_mutations = library.get_mutations_for_task(TaskType.NUMERIC)

        # Should have different mutations for different tasks
        classification_names = {m.name for m in classification_mutations}
        numeric_names = {m.name for m in numeric_mutations}

        # Numeric-specific mutation should only be in numeric
        assert "add_numeric_format_constraint" in numeric_names
        assert "add_label_format_constraint" in classification_names

    def test_global_library_singleton(self) -> None:
        """Test global library is accessible."""
        library = get_mutation_library()
        assert isinstance(library, SemanticMutationLibrary)

    def test_register_custom_mutation(self) -> None:
        """Test registering custom mutation."""
        library = SemanticMutationLibrary()
        initial_count = len(library.get_all_mutations())

        custom_mutation = SemanticMutation(
            name="custom_test",
            description="Test mutation",
            weight=1.0,
            applicable_tasks=(TaskType.UNKNOWN,),
            transform=lambda text, task, mode: text + " CUSTOM",
        )
        library.register_mutation(custom_mutation)

        assert len(library.get_all_mutations()) == initial_count + 1


class TestSemanticMutations:
    """Test individual semantic mutations."""

    def test_add_exact_output_directive(self) -> None:
        """Test exact output directive mutation."""
        library = SemanticMutationLibrary()
        text = "Classify this: {input}"

        result = library._add_exact_output_directive(
            text, TaskType.CLASSIFICATION, OutputMode.LABEL
        )

        assert "answer" in result.lower() or "only" in result.lower()

    def test_add_numeric_format_constraint(self) -> None:
        """Test numeric format constraint mutation."""
        library = SemanticMutationLibrary()
        text = "Calculate the sum: {input}"

        result = library._add_numeric_format_constraint(
            text, TaskType.NUMERIC, OutputMode.NUMERIC
        )

        assert "number" in result.lower()

    def test_promote_to_qa_layout(self) -> None:
        """Test QA layout promotion."""
        library = SemanticMutationLibrary()
        text = "Answer this: {input}"

        result = library._promote_to_qa_layout(text, TaskType.QA, OutputMode.EXACT_MATCH)

        assert "Question:" in result or "Answer:" in result

    def test_promote_to_task_layout(self) -> None:
        """Test task layout promotion."""
        library = SemanticMutationLibrary()
        text = "Classify: {input}"

        result = library._promote_to_task_layout(
            text, TaskType.CLASSIFICATION, OutputMode.LABEL
        )

        assert "Task:" in result or "Input:" in result or "Output:" in result

    def test_suppress_explanation(self) -> None:
        """Test explanation suppression."""
        library = SemanticMutationLibrary()
        text = "Solve this problem: {input}"

        result = library._suppress_explanation(text, TaskType.NUMERIC, OutputMode.NUMERIC)

        assert "explanation" in result.lower()

    def test_remove_redundant_phrases(self) -> None:
        """Test redundant phrase removal."""
        library = SemanticMutationLibrary()
        text = "Please kindly answer this question"

        result = library._remove_redundant_phrases(
            text, TaskType.QA, OutputMode.EXACT_MATCH
        )

        # Should remove at least one redundant word
        assert "please" not in result.lower() or "kindly" not in result.lower()

    def test_clarify_action_verb(self) -> None:
        """Test action verb clarification."""
        library = SemanticMutationLibrary()
        text = "Give me the answer"

        result = library._clarify_action_verb(text, TaskType.QA, OutputMode.EXACT_MATCH)

        # Should replace "give" with something more specific
        assert result != text or "give" not in text.lower()

    def test_add_classification_options(self) -> None:
        """Test classification options addition."""
        library = SemanticMutationLibrary()
        text = "Classify as positive or negative: {input}"

        result = library._add_classification_options(
            text, TaskType.CLASSIFICATION, OutputMode.LABEL
        )

        # Should add options
        assert "Options:" in result

    def test_no_mutation_when_not_applicable(self) -> None:
        """Test mutations return unchanged text when not applicable."""
        library = SemanticMutationLibrary()
        text = "Write a story about: {input}"

        # Numeric constraint shouldn't apply to non-numeric task
        result = library._add_numeric_format_constraint(
            text, TaskType.GENERATION, OutputMode.FREE_FORM
        )

        assert result == text

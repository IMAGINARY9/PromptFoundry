"""Regression tests for bundled task scoring contracts."""

from __future__ import annotations

from pathlib import Path

from promptfoundry.cli import _get_evaluator, _load_task


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_bundled_task(task_name: str):
    task_path = REPO_ROOT / "examples" / task_name
    task, evaluator_type, evaluator_config = _load_task(task_path)
    evaluator = _get_evaluator(evaluator_type, evaluator_config)
    return task, evaluator


class TestBundledTaskContracts:
    """Ensure shipped example tasks score outputs according to their contract."""

    def test_sentiment_task_rejects_explanatory_label_outputs(self) -> None:
        """Sentiment classification is a strict one-label task."""
        _task, evaluator = _load_bundled_task("sentiment_task.yaml")

        score = evaluator.evaluate(
            "Positive sentiment. The customer is satisfied with the purchase.",
            "positive",
        )

        assert score == 0.0

    def test_json_task_does_not_treat_wrapped_json_as_perfect(self) -> None:
        """JSON formatting should reward exact structure, not explanatory wrappers."""
        _task, evaluator = _load_bundled_task("json_formatting_task.yaml")

        score = evaluator.evaluate(
            'Here is the JSON: {"name": "John Smith", "age": 30, "occupation": "software engineer", "city": "New York"}',
            '{"name": "John Smith", "age": 30, "occupation": "software engineer", "city": "New York"}',
        )

        assert score < 1.0

    def test_extraction_task_does_not_treat_labeled_output_as_perfect(self) -> None:
        """Structured extraction should penalize verbose field labels."""
        _task, evaluator = _load_bundled_task("extraction_task.yaml")

        score = evaluator.evaluate(
            "Product: SuperWidget Pro | Price: $49.99 | Stock: 150",
            "SuperWidget Pro | $49.99 | 150",
        )

        assert score < 1.0

    def test_arithmetic_task_requires_numeric_only_output(self) -> None:
        """Arithmetic reasoning should reject explanatory numeric answers."""
        _task, evaluator = _load_bundled_task("arithmetic_task.yaml")

        assert evaluator.evaluate("8", "8") == 1.0
        assert evaluator.evaluate("The answer is 8.", "8") == 0.0

    def test_word_problem_task_requires_bare_numeric_answer(self) -> None:
        """Word-problem benchmark is intentionally strict to expose formatting gains."""
        _task, evaluator = _load_bundled_task("word_problems_task.yaml")

        assert evaluator.evaluate("21", "21") == 1.0
        assert evaluator.evaluate("The answer is 21.", "21") == 0.0
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
        assert evaluator.evaluate("8.", "8") == 1.0
        assert 0.0 < evaluator.evaluate("The answer is 8.", "8") < 1.0

    def test_word_problem_task_requires_bare_numeric_answer(self) -> None:
        """Word-problem benchmark is intentionally strict to expose formatting gains."""
        _task, evaluator = _load_bundled_task("word_problems_task.yaml")

        assert evaluator.evaluate("21", "21") == 1.0
        assert evaluator.evaluate("21.", "21") == 1.0
        assert 0.0 < evaluator.evaluate("The answer is 21.", "21") < 1.0

    def test_schema_task_rewards_complete_json_with_null_placeholders(self) -> None:
        """Schema extraction should reward valid JSON and explicit nulls for missing fields."""
        _task, evaluator = _load_bundled_task("schema_extraction_task.yaml")

        expected = '{"name": "Alice Chen", "email": "alice@northwind.com", "phone": null, "company": "Northwind", "title": "Senior Recruiter"}'
        missing_null = '{"name": "Alice Chen", "email": "alice@northwind.com", "company": "Northwind", "title": "Senior Recruiter"}'

        assert evaluator.evaluate(expected, expected) == 1.0
        assert 0.0 < evaluator.evaluate(missing_null, expected) < 1.0

    def test_hierarchical_intent_task_requires_single_route_label(self) -> None:
        """Hierarchical routing should still enforce a strict single-label contract."""
        _task, evaluator = _load_bundled_task("hierarchical_intent_task.yaml")

        assert evaluator.evaluate("billing/refund", "billing/refund") == 1.0
        assert evaluator.evaluate("billing/refund because the user wants money back", "billing/refund") == 0.0

    def test_long_context_task_rewards_complete_structured_extraction(self) -> None:
        """Long-context extraction should preserve partial credit for incomplete JSON."""
        _task, evaluator = _load_bundled_task("long_context_extraction_task.yaml")

        expected = '{"incident_id": "INC-4821", "severity": "sev-1", "owner": "Maya Singh", "customer": "Northwind Retail", "due_date": "2026-03-15"}'
        incomplete = '{"incident_id": "INC-4821", "severity": "sev-1", "owner": "Maya Singh"}'

        assert evaluator.evaluate(expected, expected) == 1.0
        assert 0.0 < evaluator.evaluate(incomplete, expected) < 1.0

    def test_multilingual_routing_task_still_requires_single_canonical_label(self) -> None:
        """Multilingual routing should stay strict on the final English route label."""
        _task, evaluator = _load_bundled_task("multilingual_routing_task.yaml")

        assert evaluator.evaluate("support/access", "support/access") == 1.0
        assert evaluator.evaluate("support/access porque el usuario no puede iniciar sesion", "support/access") == 0.0

    def test_multilingual_extraction_task_preserves_partial_credit(self) -> None:
        """Multilingual extraction should preserve gradient before strict JSON is perfect."""
        _task, evaluator = _load_bundled_task("multilingual_incident_extraction_task.yaml")

        expected = '{"incident_id": "INC-4821", "severity": "sev-1", "owner": "Maya Singh", "customer": "Northwind Retail", "due_date": "2026-03-15"}'
        incomplete = '{"incident_id": "INC-4821", "severity": "sev-1", "customer": "Northwind Retail"}'

        assert evaluator.evaluate(expected, expected) == 1.0
        assert 0.0 < evaluator.evaluate(incomplete, expected) < 1.0

    def test_ambiguous_routing_task_supports_explicit_abstain_label(self) -> None:
        """Ambiguous routing should require the explicit abstain/escalation label."""
        _task, evaluator = _load_bundled_task("ambiguous_intent_routing_task.yaml")

        assert evaluator.evaluate("escalate/ambiguous", "escalate/ambiguous") == 1.0
        assert evaluator.evaluate(
            "escalate/ambiguous because billing and invoice are both plausible",
            "escalate/ambiguous",
        ) == 0.0

    def test_tool_action_schema_task_requires_complete_action_object(self) -> None:
        """Tool-choice tasks should preserve partial credit until the schema is complete."""
        _task, evaluator = _load_bundled_task("tool_action_schema_task.yaml")

        expected = '{"tool": "incident_api", "action": "page_oncall", "target": "incident", "priority": "urgent"}'
        incomplete = '{"tool": "incident_api", "action": "page_oncall"}'

        assert evaluator.evaluate(expected, expected) == 1.0
        assert 0.0 < evaluator.evaluate(incomplete, expected) < 1.0
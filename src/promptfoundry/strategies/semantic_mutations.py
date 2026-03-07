"""Semantic mutation operator library for evolutionary prompt optimization.

This module provides intelligent prompt transformations that preserve meaning
while improving structure, clarity, and task alignment. Unlike blind word-order
mutations, these operators understand prompt semantics.

MVP 3 Feature: Replaces blind mutations with semantic-aware transformations.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Detected task type for adaptive mutation selection."""

    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    GENERATION = "generation"
    QA = "qa"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"


class OutputMode(Enum):
    """Expected output format for the task."""

    EXACT_MATCH = "exact_match"  # Single word/label expected
    NUMERIC = "numeric"  # Number expected
    STRUCTURED = "structured"  # JSON/formatted output
    FREE_FORM = "free_form"  # Open-ended response
    LABEL = "label"  # Classification label


@dataclass(frozen=True)
class SemanticMutation:
    """A semantic-aware mutation operator.

    Attributes:
        name: Unique identifier for the mutation.
        description: Human-readable description.
        weight: Base selection weight (higher = more likely).
        applicable_tasks: Task types this mutation applies to.
        transform: The transformation function.
    """

    name: str
    description: str
    weight: float
    applicable_tasks: tuple[TaskType, ...]
    transform: Callable[[str, TaskType, OutputMode], str]


class TaskDetector:
    """Detects task type and output mode from prompt text."""

    # Patterns for task type detection
    CLASSIFICATION_PATTERNS = [
        r"\bclassif\w*\b",
        r"\bsentiment\b",
        r"\blabel\b",
        r"\bcategor\w*\b",
        r"\bpositive\s+or\s+negative\b",
        r"\b(yes|no)\s+answer\b",
        r"\btrue\s+or\s+false\b",
    ]

    EXTRACTION_PATTERNS = [
        r"\bextract\b",
        r"\bfind\s+(?:the|all)\b",
        r"\bidentify\b",
        r"\blist\s+(?:the|all)\b",
        r"\bwhat\s+(?:is|are)\s+the\b",
        r"\bpull\s+out\b",
    ]

    REASONING_PATTERNS = [
        r"\bexplain\b",
        r"\bwhy\b",
        r"\breason\w*\b",
        r"\bstep\s+by\s+step\b",
        r"\bthink\b",
        r"\banalyze\b",
        r"\bconsider\b",
    ]

    NUMERIC_PATTERNS = [
        r"\bhow\s+many\b",
        r"\bcalculate\b",
        r"\b(add|subtract|multiply|divide)\b",
        r"\bsum\b",
        r"\bcount\b",
        r"\bnumber\s+of\b",
        r"\btotal\b",
        r"\barithmetic\b",
        r"\bmath\b",
    ]

    QA_PATTERNS = [
        r"\banswer\s+(?:the|this)?\s*question\b",
        r"^\s*q:\s*",
        r"\bquestion\s*:",
        r"\bwhat\s+is\b",
        r"\bwho\s+is\b",
        r"\bwhere\s+is\b",
        r"\bwhen\s+did\b",
    ]

    @classmethod
    def detect_task_type(cls, text: str) -> TaskType:
        """Detect the task type from prompt text.

        Args:
            text: The prompt text to analyze.

        Returns:
            Detected task type.
        """
        text_lower = text.lower()

        # Check patterns in order of specificity
        patterns_map = [
            (cls.NUMERIC_PATTERNS, TaskType.NUMERIC),
            (cls.CLASSIFICATION_PATTERNS, TaskType.CLASSIFICATION),
            (cls.EXTRACTION_PATTERNS, TaskType.EXTRACTION),
            (cls.REASONING_PATTERNS, TaskType.REASONING),
            (cls.QA_PATTERNS, TaskType.QA),
        ]

        for patterns, task_type in patterns_map:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return task_type

        return TaskType.UNKNOWN

    @classmethod
    def detect_output_mode(cls, text: str, task_type: TaskType) -> OutputMode:
        """Detect expected output mode from prompt and task type.

        Args:
            text: The prompt text.
            task_type: Already detected task type.

        Returns:
            Expected output mode.
        """
        text_lower = text.lower()

        # Explicit output mode indicators
        if any(
            phrase in text_lower
            for phrase in [
                "return only",
                "output only",
                "respond with only",
                "answer with only",
                "just the",
                "single word",
                "one word",
            ]
        ):
            return OutputMode.EXACT_MATCH

        if any(
            phrase in text_lower
            for phrase in ["json", "format:", "structured", "{", "fields:"]
        ):
            return OutputMode.STRUCTURED

        # Task-type based defaults
        task_mode_map = {
            TaskType.CLASSIFICATION: OutputMode.LABEL,
            TaskType.NUMERIC: OutputMode.NUMERIC,
            TaskType.EXTRACTION: OutputMode.EXACT_MATCH,
            TaskType.REASONING: OutputMode.FREE_FORM,
            TaskType.GENERATION: OutputMode.FREE_FORM,
            TaskType.QA: OutputMode.EXACT_MATCH,
            TaskType.UNKNOWN: OutputMode.FREE_FORM,
        }

        return task_mode_map.get(task_type, OutputMode.FREE_FORM)


class SemanticMutationLibrary:
    """Library of semantic-aware mutation operators.

    Provides intelligent transformations organized by mutation category:
    - Instruction clarity mutations
    - Output constraint mutations
    - Structure mutations
    - Task-specific mutations
    """

    def __init__(self) -> None:
        """Initialize the mutation library."""
        self._mutations: list[SemanticMutation] = []
        self._register_core_mutations()

    def _register_core_mutations(self) -> None:
        """Register all core semantic mutations."""
        # Instruction clarity mutations
        self._mutations.extend(
            [
                SemanticMutation(
                    name="clarify_action_verb",
                    description="Replace vague verbs with specific action words",
                    weight=1.2,
                    applicable_tasks=(
                        TaskType.CLASSIFICATION,
                        TaskType.EXTRACTION,
                        TaskType.QA,
                        TaskType.UNKNOWN,
                    ),
                    transform=self._clarify_action_verb,
                ),
                SemanticMutation(
                    name="add_explicit_instruction",
                    description="Add explicit task instruction at the start",
                    weight=1.5,
                    applicable_tasks=(TaskType.UNKNOWN, TaskType.GENERATION),
                    transform=self._add_explicit_instruction,
                ),
                SemanticMutation(
                    name="remove_redundant_phrases",
                    description="Remove filler and redundant language",
                    weight=0.9,
                    applicable_tasks=(
                        TaskType.CLASSIFICATION,
                        TaskType.EXTRACTION,
                        TaskType.QA,
                        TaskType.NUMERIC,
                        TaskType.UNKNOWN,
                    ),
                    transform=self._remove_redundant_phrases,
                ),
            ]
        )

        # Output constraint mutations
        self._mutations.extend(
            [
                SemanticMutation(
                    name="add_exact_output_directive",
                    description="Add directive for exact-match outputs",
                    weight=1.8,
                    applicable_tasks=(
                        TaskType.CLASSIFICATION,
                        TaskType.EXTRACTION,
                        TaskType.NUMERIC,
                        TaskType.QA,
                    ),
                    transform=self._add_exact_output_directive,
                ),
                SemanticMutation(
                    name="add_numeric_format_constraint",
                    description="Add constraint for numeric-only output",
                    weight=1.7,
                    applicable_tasks=(TaskType.NUMERIC,),
                    transform=self._add_numeric_format_constraint,
                ),
                SemanticMutation(
                    name="add_label_format_constraint",
                    description="Add constraint for label-only output",
                    weight=1.7,
                    applicable_tasks=(TaskType.CLASSIFICATION,),
                    transform=self._add_label_format_constraint,
                ),
                SemanticMutation(
                    name="suppress_explanation",
                    description="Add directive to suppress explanations",
                    weight=1.6,
                    applicable_tasks=(
                        TaskType.CLASSIFICATION,
                        TaskType.NUMERIC,
                        TaskType.QA,
                        TaskType.EXTRACTION,
                    ),
                    transform=self._suppress_explanation,
                ),
            ]
        )

        # Structure mutations
        self._mutations.extend(
            [
                SemanticMutation(
                    name="promote_to_qa_layout",
                    description="Convert to Question/Answer layout",
                    weight=1.6,
                    applicable_tasks=(TaskType.QA, TaskType.UNKNOWN),
                    transform=self._promote_to_qa_layout,
                ),
                SemanticMutation(
                    name="promote_to_task_layout",
                    description="Convert to Task/Input/Output layout",
                    weight=1.5,
                    applicable_tasks=(
                        TaskType.CLASSIFICATION,
                        TaskType.EXTRACTION,
                        TaskType.NUMERIC,
                    ),
                    transform=self._promote_to_task_layout,
                ),
                SemanticMutation(
                    name="add_input_delimiter",
                    description="Add clear delimiter before input",
                    weight=1.3,
                    applicable_tasks=(
                        TaskType.CLASSIFICATION,
                        TaskType.EXTRACTION,
                        TaskType.QA,
                        TaskType.NUMERIC,
                        TaskType.UNKNOWN,
                    ),
                    transform=self._add_input_delimiter,
                ),
            ]
        )

        # Task-specific mutations
        self._mutations.extend(
            [
                SemanticMutation(
                    name="add_classification_options",
                    description="Explicitly list classification options",
                    weight=1.4,
                    applicable_tasks=(TaskType.CLASSIFICATION,),
                    transform=self._add_classification_options,
                ),
                SemanticMutation(
                    name="add_verification_step",
                    description="Add silent verification directive",
                    weight=1.2,
                    applicable_tasks=(TaskType.NUMERIC, TaskType.REASONING),
                    transform=self._add_verification_step,
                ),
                SemanticMutation(
                    name="add_step_by_step_then_answer",
                    description="Add think-then-answer directive",
                    weight=1.3,
                    applicable_tasks=(TaskType.REASONING, TaskType.NUMERIC),
                    transform=self._add_step_by_step_then_answer,
                ),
            ]
        )

    def get_mutations_for_task(
        self,
        task_type: TaskType,
    ) -> list[SemanticMutation]:
        """Get mutations applicable to a task type.

        Args:
            task_type: The detected task type.

        Returns:
            List of applicable mutations.
        """
        return [
            m
            for m in self._mutations
            if task_type in m.applicable_tasks or TaskType.UNKNOWN in m.applicable_tasks
        ]

    def get_all_mutations(self) -> list[SemanticMutation]:
        """Get all registered mutations."""
        return self._mutations.copy()

    def register_mutation(self, mutation: SemanticMutation) -> None:
        """Register a custom mutation operator.

        Args:
            mutation: The mutation to register.
        """
        self._mutations.append(mutation)

    # =========================================================================
    # Instruction Clarity Mutations
    # =========================================================================

    def _clarify_action_verb(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Replace vague verbs with task-appropriate action words."""
        substitutions = {
            "do": ["perform", "execute", "complete"],
            "give": ["provide", "return", "output"],
            "say": ["state", "respond with", "output"],
            "tell": ["indicate", "specify", "state"],
            "figure out": ["determine", "calculate", "identify"],
            "find out": ["determine", "discover", "identify"],
            "look at": ["analyze", "examine", "inspect"],
        }

        result = text
        for vague, specific in substitutions.items():
            pattern = re.compile(r"\b" + re.escape(vague) + r"\b", re.IGNORECASE)
            if pattern.search(result):
                replacement = random.choice(specific)
                result = pattern.sub(replacement, result, count=1)
                break

        return result

    def _add_explicit_instruction(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add explicit instruction prefix when task is unclear."""
        if task_type != TaskType.UNKNOWN:
            return text

        # Check if instruction already exists
        instruction_indicators = [
            "task:",
            "instruction:",
            "please",
            "you must",
            "your task",
        ]
        if any(ind in text.lower() for ind in instruction_indicators):
            return text

        prefixes = [
            "Complete the following task: ",
            "Perform the following: ",
            "Your task is to: ",
        ]
        prefix = random.choice(prefixes)
        return prefix + text

    def _remove_redundant_phrases(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Remove filler and redundant language."""
        redundant_patterns = [
            (r"\bplease\s+", ""),
            (r"\bkindly\s+", ""),
            (r"\bjust\s+", ""),
            (r"\bsimply\s+", ""),
            (r"\bbasically\s+", ""),
            (r"\bactually\s+", ""),
            (r"\bi\s+want\s+you\s+to\s+", ""),
            (r"\bi\s+need\s+you\s+to\s+", ""),
            (r"\bcan\s+you\s+", ""),
            (r"\bwould\s+you\s+", ""),
        ]

        result = text
        for pattern, replacement in redundant_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE, count=1)
            if result != text:
                break

        return result.strip()

    # =========================================================================
    # Output Constraint Mutations
    # =========================================================================

    def _add_exact_output_directive(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add directive for exact-match outputs."""
        if output_mode not in (OutputMode.EXACT_MATCH, OutputMode.LABEL, OutputMode.NUMERIC):
            return text

        directives = [
            " Respond with only the final answer.",
            " Return only the answer, nothing else.",
            " Output just the answer.",
        ]
        return self._append_missing_directive(text, directives)

    def _add_numeric_format_constraint(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add constraint for numeric-only output."""
        if task_type != TaskType.NUMERIC:
            return text

        directives = [
            " Return only the number.",
            " Output the numeric result only.",
            " Respond with just the number, no units.",
        ]
        return self._append_missing_directive(text, directives)

    def _add_label_format_constraint(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add constraint for label-only output."""
        if task_type != TaskType.CLASSIFICATION:
            return text

        directives = [
            " Return exactly one label.",
            " Output only the classification label.",
            " Respond with a single label only.",
        ]
        return self._append_missing_directive(text, directives)

    def _suppress_explanation(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add directive to suppress verbose explanations."""
        directives = [
            " Do not include any explanation.",
            " No explanation needed.",
            " Skip any explanation or reasoning.",
        ]
        return self._append_missing_directive(text, directives)

    # =========================================================================
    # Structure Mutations
    # =========================================================================

    def _promote_to_qa_layout(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Convert to Question/Answer layout."""
        if "{input}" not in text:
            return text

        # Don't restructure if already structured
        if any(marker in text for marker in ["Question:", "Q:", "Input:", "Task:"]):
            return text

        instruction = self._extract_instruction(text)
        return f"{instruction}\n\nQuestion: {{input}}\n\nAnswer:"

    def _promote_to_task_layout(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Convert to Task/Input/Output layout."""
        if "{input}" not in text:
            return text

        if any(marker in text for marker in ["Task:", "Input:", "Output:"]):
            return text

        instruction = self._extract_instruction(text)
        return f"Task: {instruction}\n\nInput: {{input}}\n\nOutput:"

    def _add_input_delimiter(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add clear delimiter before input placeholder."""
        if "{input}" not in text:
            return text

        # Already has delimiter
        if re.search(r"(Input|Question|Text|Query)\s*:\s*\{input\}", text):
            return text

        delimiters = ["Input: ", "Text: ", "Query: "]
        delimiter = random.choice(delimiters)
        return text.replace("{input}", f"\n\n{delimiter}{{input}}")

    # =========================================================================
    # Task-Specific Mutations
    # =========================================================================

    def _add_classification_options(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Explicitly list classification options if detected in prompt."""
        if task_type != TaskType.CLASSIFICATION:
            return text

        # Detect common classification patterns
        sentiment_pattern = r"\b(positive|negative|neutral)\b"
        yes_no_pattern = r"\b(yes|no)\b"
        true_false_pattern = r"\b(true|false)\b"

        text_lower = text.lower()

        if re.search(sentiment_pattern, text_lower):
            if "options:" not in text_lower:
                return text.rstrip() + " Options: positive, negative, neutral."
        elif re.search(yes_no_pattern, text_lower):
            if "options:" not in text_lower:
                return text.rstrip() + " Options: yes, no."
        elif re.search(true_false_pattern, text_lower):
            if "options:" not in text_lower:
                return text.rstrip() + " Options: true, false."

        return text

    def _add_verification_step(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add silent verification directive."""
        directives = [
            " Verify your answer silently before responding.",
            " Double-check the result, then output only the final answer.",
            " Check your work, then respond with just the answer.",
        ]
        return self._append_missing_directive(text, directives)

    def _add_step_by_step_then_answer(
        self,
        text: str,
        task_type: TaskType,
        output_mode: OutputMode,
    ) -> str:
        """Add think-then-answer directive for reasoning tasks."""
        if "step by step" in text.lower():
            return text

        directives = [
            " Think step by step, then provide only the final answer.",
            " Reason through the problem, but output only the result.",
        ]
        return self._append_missing_directive(text, directives)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _extract_instruction(self, text: str) -> str:
        """Extract the core instruction from a prompt."""
        # Remove placeholder and clean up
        instruction = text.replace("{input}", "").strip()
        instruction = re.sub(r"\s+", " ", instruction)
        instruction = re.sub(r"[:\-]\s*$", "", instruction)

        if not instruction:
            instruction = "Complete the following task"

        return instruction

    def _append_missing_directive(self, text: str, directives: list[str]) -> str:
        """Append the first directive not already present."""
        text_lower = text.lower()
        for directive in directives:
            directive_lower = directive.strip().lower()
            # Check for semantic presence, not just exact match
            key_words = [w for w in directive_lower.split() if len(w) > 3]
            if not all(w in text_lower for w in key_words[:3]):
                return text.rstrip() + directive
        return text


# Global library instance
_mutation_library = SemanticMutationLibrary()


def get_mutation_library() -> SemanticMutationLibrary:
    """Get the global semantic mutation library."""
    return _mutation_library

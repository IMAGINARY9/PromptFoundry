"""Helpers for normalizing concise task outputs from verbose completions."""

from __future__ import annotations

import re

_PREFIX_PATTERN = re.compile(
    r"(?:answer|final answer|result|classification|label|sentiment)\s*(?:is|:)?\s*",
    re.IGNORECASE,
)
_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_ANSWER_INTRO_PATTERN = re.compile(
    r"(?:^|\b)(?:answer|final answer|result|classification|label|sentiment)(?:\b|\s*:)",
    re.IGNORECASE,
)


def normalize_for_exact_match(
    predicted: str,
    expected: str,
    case_sensitive: bool = False,
) -> str:
    """Normalize a verbose completion down to its most likely final answer."""
    pred = predicted.strip()
    exp = expected.strip()

    if not pred or not exp:
        return pred

    first_line = pred.splitlines()[0].strip()
    first_line = _PREFIX_PATTERN.sub("", first_line).strip()
    exp_token = _strip_wrapping_punctuation(exp)

    if _matches_exact(_strip_wrapping_punctuation(first_line), exp_token, case_sensitive):
        return exp_token

    if _is_numeric_answer(exp_token):
        numeric = _extract_numeric_candidate(first_line) or _extract_numeric_candidate(pred)
        if numeric is not None:
            return numeric

    if _is_short_expected(exp_token):
        exact_token_match = _match_expected_token(
            first_line,
            exp_token,
            case_sensitive=case_sensitive,
            require_answer_intro=True,
        )
        if exact_token_match is not None:
            return exact_token_match

        exact_token_match = _match_expected_token(
            pred,
            exp_token,
            case_sensitive=case_sensitive,
            require_answer_intro=True,
        )
        if exact_token_match is not None:
            return exact_token_match

    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()
    return _strip_wrapping_punctuation(first_sentence)


def _is_numeric_answer(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text))


def _is_short_expected(text: str) -> bool:
    return len(text.split()) <= 3 and len(text) <= 32


def _extract_numeric_candidate(text: str) -> str | None:
    match = _NUMERIC_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0)


def _match_expected_token(
    text: str,
    expected: str,
    case_sensitive: bool,
    require_answer_intro: bool,
) -> str | None:
    if require_answer_intro and not _ANSWER_INTRO_PATTERN.search(text):
        return None

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(
        rf"(^|[^a-z0-9]){re.escape(expected)}([^a-z0-9]|$)",
        flags,
    )
    if pattern.search(text):
        return expected
    return None


def _matches_exact(left: str, right: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return left == right
    return left.lower() == right.lower()


def _strip_wrapping_punctuation(text: str) -> str:
    stripped = text.strip()
    stripped = stripped.strip("`'\" ")
    stripped = stripped.rstrip(".!?,;:")
    return stripped.strip()

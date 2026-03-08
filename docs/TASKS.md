# PromptFoundry Task Catalog

> **Version:** 1.0.0
> **Status:** Active
> **Last Updated:** 2026-03-08
> **Authoritative Source:** Bundled task inventory, evaluator contracts, and MVP 3 task expansion guidance.

---

## 1. Overview

This document defines the bundled example tasks shipped with PromptFoundry, the evaluation contract each task enforces, and the real-world workload each task is intended to represent.

The tasks are not interchangeable. Some are intentionally strict to measure prompt improvements in output discipline, while others are intentionally tolerant to measure semantic extraction quality under noisy formatting.

---

## 2. Bundled Tasks

| Task | File | Evaluator Contract | Real-World Purpose |
|------|------|--------------------|--------------------|
| Sentiment Classification | [examples/sentiment_task.yaml](../examples/sentiment_task.yaml) | `exact_match` with `normalize_output: false` | Triage of reviews, support feedback labeling, moderation queues, basic routing classifiers |
| JSON Formatting | [examples/json_formatting_task.yaml](../examples/json_formatting_task.yaml) | `fuzzy_match` against exact JSON payload | LLM-to-system handoff, lightweight ETL, API payload drafting, document-to-JSON extraction |
| Structured Extraction | [examples/extraction_task.yaml](../examples/extraction_task.yaml) | `fuzzy_match` against compact pipe-separated output | Contact extraction, product feed parsing, event detail capture, log-to-record normalization |
| Arithmetic Reasoning | [examples/arithmetic_task.yaml](../examples/arithmetic_task.yaml) | `numeric_answer` with strict numeric-only perfect scoring and prose partial credit | Calculator-style assistants, invoice math verification, quantity checks, routing tasks that require a bare numeric result |
| Word Math Problems | [examples/word_problems_task.yaml](../examples/word_problems_task.yaml) | `numeric_answer` with strict numeric-only perfect scoring and prose partial credit | Higher-variance quantitative reasoning where prompt optimization must discover answer-only formatting |

---

## 3. Evaluation Contracts

### 3.1 Strict-output tasks

The following tasks are strict by design and should reject explanatory completions as perfect answers even when the semantic answer is correct:

- Sentiment Classification
- Arithmetic Reasoning
- Word Math Problems

These tasks are used to measure whether optimization discovers instructions like “return exactly one label” or “output only the final answer.” If verbose outputs were accepted as correct, the benchmark would overstate quality and reduce the signal available to MVP 3 operator tracking.

### 3.2 Tolerant-format tasks

The following tasks intentionally allow partial progress and non-perfect formatting:

- JSON Formatting
- Structured Extraction

These tasks model situations where an output can improve gradually across generations. Fuzzy scoring is useful here because it preserves a measurable optimization gradient instead of collapsing quality into a strict pass/fail boundary.

### 3.3 Audit corrections applied

The March 2026 audit corrected three issues in the shipped tasks:

- Sentiment Classification no longer accepts verbose sentiment explanations as perfect label matches.
- Word Math Problems no longer accept explanatory numeric answers as perfect scores.
- Arithmetic Reasoning enforces a bare numeric answer for a perfect score while preserving partial credit when the correct number appears in prose.

---

## 4. Task Coverage by Real-World Condition

### 4.1 Classification and routing

Primary task: Sentiment Classification.

Use this when the production system expects a closed label set and downstream logic branches on a single token or class name.

### 4.2 Structured extraction from semi-structured text

Primary tasks: JSON Formatting and Structured Extraction.

Use these when the output feeds another program, a report generator, a CRM ingest path, or a records normalization pipeline. These tasks are especially useful for validating MVP 3 semantic mutations because they benefit from clearer formatting constraints and layout-aware prompt changes.

### 4.3 Numeric reasoning with strict downstream parsing

Primary tasks: Arithmetic Reasoning and Word Math Problems.

Use these when the answer will be consumed by code, a spreadsheet, a rules engine, or a validation step that cannot tolerate surrounding prose.

---

## 5. Assessment of Current Task Set

The current task bundle is good enough to validate MVP 3 on three important axes:

- Closed-label discipline
- Structured output compliance
- Numeric answer discipline

The current task bundle is still narrow in several ways:

- It does not cover long-context extraction where irrelevant text competes with the target facts.
- It does not cover optional or missing fields, which is common in real extraction pipelines.
- It does not cover multi-label classification, hierarchical labels, or ambiguous intent routing.
- It does not cover multilingual prompts or locale-sensitive outputs.
- It does not cover agent-like tasks where the right answer is a tool choice, action plan, or schema-constrained command.

---

## 6. Recommended MVP 3 Expansions

The next task additions should stay inside MVP 3’s current architecture: semantic mutations, diversity control, adaptive schedules, and ablation tracking.

### 6.1 Schema-constrained extraction with missing fields

Add a task where the model must output JSON with required keys plus explicit `null` values for missing information.

Why it matters: this tests whether semantic mutations can learn field-presence directives and whether proxy evaluators should move from fuzzy string similarity toward schema-aware scoring.

### 6.2 Intent routing with hierarchical labels

Add a task with labels such as `billing/refund`, `billing/invoice`, `support/bug`, and `support/access`.

Why it matters: it expands classification beyond shallow sentiment and tests whether label constraints remain useful when the label vocabulary becomes less obvious.

### 6.3 Long-context extraction

Add examples where relevant facts are buried inside longer passages with distractor information.

Why it matters: this will expose whether current mutations improve focus and instruction clarity or only help on short prompts.

### 6.4 Multilingual classification and extraction

Add a task mix with English plus at least one other language.

Why it matters: this stresses evaluator assumptions, exact-match normalization rules, and prompt robustness under different token distributions.

### 6.5 Multi-step reasoning with structured output

Add tasks that require reasoning but must emit a compact JSON object or tuple-like answer.

Why it matters: this combines reasoning and formatting pressure, which is closer to real production workflows than pure math or pure extraction alone.

### 6.6 Staged evaluation pipelines for complex tasks

Introduce new tasks only when paired with a stronger evaluator stack such as:

- JSON parse filter
- Schema compliance scorer
- Exact or fuzzy value scorer

Why it matters: MVP 3 can optimize prompts well, but complex workloads need better evaluation signal than a single flat metric.

---

## 7. Guidance for Adding New Tasks

When adding a task, define these properties explicitly in the task file and review:

- Whether the task is strict-output or tolerant-format.
- Whether the evaluator should reward gradual improvement or enforce a hard contract.
- Whether the task is intended for benchmark gating or exploratory validation only.
- What real-world downstream consumer would break if the output included extra prose.

If the downstream system expects machine-readable output, prefer strict scoring or staged proxy metrics. If the downstream system can tolerate mild formatting drift, use fuzzy or schema-based scoring to preserve optimization signal.

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-08 | GitHub Copilot | Added bundled task inventory, audit corrections, and MVP 3 expansion guidance. |
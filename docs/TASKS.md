# PromptFoundry Task Catalog

> **Version:** 1.2.0
> **Status:** Active
> **Last Updated:** 2026-03-12
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
| Schema Extraction With Missing Fields | [examples/schema_extraction_task.yaml](../examples/schema_extraction_task.yaml) | `pipeline` (`json_parse` → `json_schema` → `fuzzy_match`) | CRM intake, contact normalization, ticket ingestion, and workflows where absent fields must be explicit `null` values |
| Hierarchical Intent Routing | [examples/hierarchical_intent_task.yaml](../examples/hierarchical_intent_task.yaml) | `label_answer` with strict bare-label perfect scoring and conservative partial credit for verbose label selection | Queue routing, workflow dispatch, escalation categorization, and multi-tenant support triage |
| Long-Context Incident Extraction | [examples/long_context_extraction_task.yaml](../examples/long_context_extraction_task.yaml) | `pipeline` (`json_value_coverage` → `json_parse` → `json_schema` → `fuzzy_match`) | Incident management, audit summaries, long-form operational notes, and extraction from noisy timelines |
| Multilingual Routing | [examples/multilingual_routing_task.yaml](../examples/multilingual_routing_task.yaml) | `label_answer` with strict canonical-label scoring across mixed-language inputs | Global support triage, multilingual helpdesk queues, and locale-shifted routing workflows |
| Multilingual Incident Extraction | [examples/multilingual_incident_extraction_task.yaml](../examples/multilingual_incident_extraction_task.yaml) | `pipeline` (`json_value_coverage` → `json_parse` → `json_schema` → `fuzzy_match`) | Cross-region incident reporting, multilingual NOC handoffs, and mixed-language ops summaries |
| Ambiguous Intent Routing | [examples/ambiguous_intent_routing_task.yaml](../examples/ambiguous_intent_routing_task.yaml) | `label_answer` with an explicit `escalate/ambiguous` abstain label | Safety-focused triage, escalation gating, and workflows that must avoid overconfident misrouting |
| Tool Action Schema | [examples/tool_action_schema_task.yaml](../examples/tool_action_schema_task.yaml) | `pipeline` (`json_value_coverage` → `json_parse` → `json_schema` → `fuzzy_match`) | Tool selection, action planning, and machine-readable orchestration commands |
| Arithmetic Reasoning | [examples/arithmetic_task.yaml](../examples/arithmetic_task.yaml) | `numeric_answer` with strict numeric-only perfect scoring and prose partial credit | Calculator-style assistants, invoice math verification, quantity checks, routing tasks that require a bare numeric result |
| Word Math Problems | [examples/word_problems_task.yaml](../examples/word_problems_task.yaml) | `numeric_answer` with strict numeric-only perfect scoring and prose partial credit | Higher-variance quantitative reasoning where prompt optimization must discover answer-only formatting |

---

## 3. Evaluation Contracts

### 3.1 Strict-output tasks

The following tasks are strict by design and should reject explanatory completions as perfect answers even when the semantic answer is correct:

- Sentiment Classification
- Hierarchical Intent Routing
- Multilingual Routing
- Ambiguous Intent Routing
- Arithmetic Reasoning
- Word Math Problems

These tasks are used to measure whether optimization discovers instructions like “return exactly one label” or “output only the final answer.” Arithmetic, word-math, and the routing tasks now preserve only conservative partial credit for verbose local-model outputs so the optimizer keeps a usable gradient without treating explanations as perfect answers.

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

Hierarchical Intent Routing extends the same contract to multi-level labels where downstream queue assignment depends on the exact slash-delimited route.

Multilingual Routing adds locale shifts without changing the canonical output labels, while Ambiguous Intent Routing adds a calibrated abstain path through `escalate/ambiguous` when safe routing is not possible.

### 4.2 Structured extraction from semi-structured text

Primary tasks: JSON Formatting and Structured Extraction.

Long-Context Incident Extraction, Multilingual Incident Extraction, Schema Extraction With Missing Fields, and Tool Action Schema extend this group to pipeline-scored JSON tasks where retrieval focus, locale shifts, explicit missing-field handling, and machine-readable action objects all matter. The extraction tasks now start with a cheap value-recovery stage so verbose outputs still produce signal before the optimizer solves strict JSON formatting.

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
- Long-context focus under distractors
- Multilingual robustness across routing and extraction
- Machine-readable tool/action selection

The new schema-missing-fields task adds a fourth axis:

- Schema completeness with explicit null handling

The current task bundle is still narrow in several ways:

- It now covers multilingual prompts, but only with a small European-language mix and canonical English output labels.
- It now covers ambiguous routing, but abstention quality is still the weakest benchmark under the bundled slow-local budget.
- It now covers tool-choice objects, but not longer multi-step action plans with dependencies.
- It does not cover locale-specific numeric/date formatting or multilingual output schemas.
- It does not cover agent-like tasks where multiple actions or tool sequences must be planned together.

---

## 6. Recommended MVP 3 Expansions

The next task additions should stay inside MVP 3’s current architecture: semantic mutations, diversity control, adaptive schedules, and ablation tracking.

### 6.1 Schema-constrained extraction with missing fields

Implemented in [examples/schema_extraction_task.yaml](../examples/schema_extraction_task.yaml).

Why it matters: this tests whether semantic mutations can learn field-presence directives and uses the first bundled YAML-configured staged evaluator stack to preserve optimization signal without over-rewarding malformed outputs.

### 6.2 Intent routing with hierarchical labels

Implemented in [examples/hierarchical_intent_task.yaml](../examples/hierarchical_intent_task.yaml).

Why it matters: it expands classification beyond shallow sentiment and tests whether label constraints remain useful when the label vocabulary becomes less obvious.

### 6.3 Long-context extraction

Implemented in [examples/long_context_extraction_task.yaml](../examples/long_context_extraction_task.yaml).

Why it matters: this now exposes whether current mutations improve focus and instruction clarity or only help on short prompts.

### 6.4 Multilingual classification and extraction

Implemented in [examples/multilingual_routing_task.yaml](../examples/multilingual_routing_task.yaml) and [examples/multilingual_incident_extraction_task.yaml](../examples/multilingual_incident_extraction_task.yaml).

Why it matters: these tasks stress evaluator assumptions, exact-match normalization rules, and prompt robustness under different token distributions while keeping canonical output contracts stable.

### 6.5 Multi-step reasoning with structured output

Implemented in [examples/tool_action_schema_task.yaml](../examples/tool_action_schema_task.yaml).

Why it matters: this combines routing-style interpretation with strict machine-readable output, which is closer to real production workflows than pure math or pure extraction alone.

### 6.6 Staged evaluation pipelines for complex tasks

Implemented across the multilingual extraction and tool-action schema tasks using a stronger evaluator stack such as:

- JSON parse filter
- Schema compliance scorer
- Exact or fuzzy value scorer

Why it matters: MVP 3 can optimize prompts well, but complex workloads need better evaluation signal than a single flat metric.

### 6.7 Stage-aware semantic mutations

Implemented in the optimizer and evolutionary strategy feedback loop.

Why it matters: pipeline-scored tasks now feed the dominant failing stage back into operator selection, so structural failures bias toward layout/constraint repairs while quality failures bias toward verification and cleanup.

### 6.8 Ambiguous routing with explicit abstention

Implemented in [examples/ambiguous_intent_routing_task.yaml](../examples/ambiguous_intent_routing_task.yaml).

Why it matters: this tests whether the optimizer can preserve strict label discipline without forcing overconfident guesses when the safest behavior is escalation.

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
| 1.1.0 | 2026-03-12 | GitHub Copilot | Added hierarchical routing, long-context extraction, stage-aware mutation feedback notes, and signal-recovery evaluator contracts for verbose local-model outputs. |
| 1.2.0 | 2026-03-12 | GitHub Copilot | Added multilingual routing/extraction, ambiguous abstain-capable routing, tool-action schema tasks, and updated current coverage limits. |
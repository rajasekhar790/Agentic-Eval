# Agentic-Eval

Evaluation framework for agentic AI systems, measuring **reasoning**, **tool usage**, **workflow reliability**, and **task success**.

---

## Overview

Agentic-Eval provides a structured way to evaluate the behaviour of AI agents across four key dimensions:

| Dimension | What it measures |
|---|---|
| **Reasoning** | Chain-of-thought completeness, logical consistency, intermediate-step correctness, conclusion accuracy |
| **Tool Usage** | Tool-selection accuracy, parameter correctness, redundant calls, tool-call success rate |
| **Workflow Reliability** | Step-completion rate, error recovery, state consistency, retry efficiency |
| **Task Success** | Binary success, partial completion, sub-goal achievement, efficiency vs. budget |

Each dimension returns a normalised score in **[0, 1]**.  The overall score is the mean across all four dimensions.

---

## Installation

```bash
pip install -e .            # editable / development install
pip install -e ".[dev]"     # also installs pytest for running tests
```

Python ≥ 3.9 is required.  The framework has **no runtime dependencies** beyond the Python standard library.

---

## Quick Start

### Python API

```python
from agentic_eval import EvaluationSuite

suite = EvaluationSuite()

sample = {
    # --- Reasoning ---
    "reasoning_steps": ["identify facts", "apply logic", "derive answer"],
    "expected_reasoning_steps": ["identify facts", "apply logic"],
    "conclusion": "Paris",
    "expected_conclusion": "Paris",

    # --- Tool usage ---
    "tool_calls": [
        {"tool": "knowledge_base", "parameters": {"query": "capital of France"}, "status": "success"}
    ],
    "expected_tool_calls": [
        {"tool": "knowledge_base", "parameters": {"query": "capital of France"}}
    ],

    # --- Workflow ---
    "steps": [
        {"id": "retrieve", "status": "completed", "retries": 0, "state": {"answer": "Paris"}},
    ],
    "total_steps": 1,
    "expected_states": {"retrieve": {"answer": "Paris"}},

    # --- Task success ---
    "status": "success",
    "goals": [{"id": "answer_question", "achieved": True}],
    "steps_taken": 1,
    "step_budget": 5,
}

report = suite.evaluate(sample)
report.print_summary()   # pretty console output
print(report.to_json())  # full JSON report
```

**Console output:**

```
============================================================
  Agentic-Eval Report  |  run: run-20260405T000000-abc123
  2026-04-05T00:00:00.000000+00:00
============================================================
  ✓ reasoning              [████████████████████]  1.0000
  ✓ tool_usage             [████████████████████]  1.0000
  ✓ workflow               [████████████████████]  1.0000
  ✓ task_success           [████████████████████]  1.0000
------------------------------------------------------------
  Overall score: 1.0000   →  PASS ✓
============================================================
```

### Batch evaluation

```python
reports = suite.evaluate_batch([sample_a, sample_b, sample_c])

# Or aggregate into a single averaged report:
agg_report = suite.aggregate_batch([sample_a, sample_b, sample_c])
agg_report.print_summary()
```

### CLI

```bash
# Evaluate a single sample
python -m agentic_eval evaluate examples/sample.json

# Evaluate a batch and aggregate
python -m agentic_eval evaluate-batch examples/batch_samples.json

# Save the JSON report to a file
python -m agentic_eval evaluate examples/sample.json --output report.json

# Show help
python -m agentic_eval --help
```

---

## Sample Format

A sample is a plain Python `dict` (or JSON object).  All keys are optional; omitting a key gracefully disables the corresponding sub-metric.

| Key | Type | Used by | Description |
|---|---|---|---|
| `reasoning_steps` | `list[str]` | Reasoning | Steps produced by the agent |
| `expected_reasoning_steps` | `list[str]` | Reasoning | Reference steps for CoT completeness |
| `intermediate_answers` | `list` | Reasoning | Intermediate answers |
| `expected_intermediate_answers` | `list` | Reasoning | Ground-truth intermediate answers |
| `conclusion` | `any` | Reasoning | Agent's final answer |
| `expected_conclusion` | `any` | Reasoning | Ground-truth final answer |
| `contradictions` | `list[str]` | Reasoning | Explicit contradiction flags |
| `tool_calls` | `list[dict]` | Tool Usage | Tool calls made (`tool`, `parameters`, `status`) |
| `expected_tool_calls` | `list[dict]` | Tool Usage | Reference tool calls |
| `steps` | `list[dict]` | Workflow | Workflow steps (`id`, `status`, `retries`, `state`, `recovered`) |
| `total_steps` | `int` | Workflow | Total expected steps in the workflow |
| `expected_states` | `dict` | Workflow | Mapping of step id → expected state dict |
| `status` | `str` | Task Success | `"success"` / `"partial"` / `"failure"` |
| `completion_score` | `float` | Task Success | Explicit partial-completion score [0, 1] |
| `goals` | `list[dict]` | Task Success | Sub-goals (`id`, `achieved`) |
| `steps_taken` | `int` | Task Success | Number of steps used by the agent |
| `step_budget` | `int` | Task Success | Allowed step budget |
| `time_taken` | `float` | Task Success | Seconds used |
| `time_budget` | `float` | Task Success | Allowed time budget (seconds) |

---

## Architecture

```
agentic_eval/
├── __init__.py           # Public API exports
├── suite.py              # EvaluationSuite orchestrator
├── cli.py                # Command-line interface
├── evaluators/
│   ├── base.py           # BaseEvaluator + EvaluationResult
│   ├── reasoning.py      # ReasoningEvaluator
│   ├── tool_usage.py     # ToolUsageEvaluator
│   ├── workflow.py       # WorkflowEvaluator
│   └── task_success.py   # TaskSuccessEvaluator
├── metrics/
│   └── metrics.py        # EvaluationMetrics (aggregation)
└── reporters/
    └── report.py         # EvaluationReport (console + JSON)
```

### Extending the framework

Implement `BaseEvaluator` and pass your custom evaluator to `EvaluationSuite`:

```python
from agentic_eval.evaluators.base import BaseEvaluator, EvaluationResult

class MyCustomEvaluator(BaseEvaluator):
    name = "my_dimension"

    def _evaluate(self, sample):
        score = ...  # compute your score
        return EvaluationResult(evaluator=self.name, score=score)

from agentic_eval import EvaluationSuite
suite = EvaluationSuite(evaluators=[MyCustomEvaluator()])
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

All 97 tests should pass.

---

## License

MIT

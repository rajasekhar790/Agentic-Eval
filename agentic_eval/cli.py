"""Command-line interface for Agentic-Eval.

Usage::

    python -m agentic_eval.cli --help
    python -m agentic_eval.cli evaluate sample.json
    python -m agentic_eval.cli evaluate sample.json --output report.json
    python -m agentic_eval.cli evaluate-batch samples.json --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-eval",
        description="Evaluate agentic AI systems using Agentic-Eval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- evaluate ----
    ev_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a single sample from a JSON file.",
    )
    ev_parser.add_argument(
        "sample",
        metavar="SAMPLE_FILE",
        help="Path to a JSON file containing the agent sample.",
    )
    ev_parser.add_argument(
        "--output", "-o",
        metavar="OUTPUT_FILE",
        default=None,
        help="Write the JSON report to this file instead of stdout.",
    )
    ev_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier.",
    )

    # ---- evaluate-batch ----
    batch_parser = subparsers.add_parser(
        "evaluate-batch",
        help="Evaluate a batch of samples and output an aggregated report.",
    )
    batch_parser.add_argument(
        "samples",
        metavar="SAMPLES_FILE",
        help="Path to a JSON file containing a list of agent samples.",
    )
    batch_parser.add_argument(
        "--output", "-o",
        metavar="OUTPUT_FILE",
        default=None,
        help="Write the JSON report to this file instead of stdout.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    from agentic_eval.suite import EvaluationSuite

    parser = _build_parser()
    args = parser.parse_args(argv)

    suite = EvaluationSuite()

    if args.command == "evaluate":
        sample_path = Path(args.sample)
        if not sample_path.exists():
            print(f"Error: file not found: {sample_path}", file=sys.stderr)
            return 1
        sample = json.loads(sample_path.read_text())
        report = suite.evaluate(sample, run_id=args.run_id)
        report.print_summary()
        _write_or_print(report.to_json(), args.output)

    elif args.command == "evaluate-batch":
        samples_path = Path(args.samples)
        if not samples_path.exists():
            print(f"Error: file not found: {samples_path}", file=sys.stderr)
            return 1
        samples = json.loads(samples_path.read_text())
        if not isinstance(samples, list):
            print("Error: samples file must contain a JSON array.", file=sys.stderr)
            return 1
        report = suite.aggregate_batch(samples)
        report.print_summary()
        _write_or_print(report.to_json(), args.output)

    return 0


def _write_or_print(content: str, output: str | None) -> None:
    if output:
        Path(output).write_text(content)
        print(f"\nReport written to: {output}")
    else:
        print(content)


if __name__ == "__main__":
    sys.exit(main())

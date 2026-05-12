"""CI threshold gate for Phase A RAGAS summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "phase-a" / "ragas_summary.json"
DEFAULT_REPORT = ROOT / "phase-a" / "eval_gate_report.md"


def parse_threshold(item: str) -> tuple[str, float]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"Expected metric=value, got {item!r}")
    metric, value = item.split("=", 1)
    metric = metric.strip()
    if not metric:
        raise argparse.ArgumentTypeError(f"Missing metric name in {item!r}")
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid threshold value in {item!r}") from exc
    return metric, threshold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Path to ragas_summary.json")
    parser.add_argument(
        "--threshold",
        action="append",
        type=parse_threshold,
        required=True,
        help="Metric gate in metric=value format. Can be repeated.",
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Markdown report path")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Missing summary file: {summary_path}", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[tuple[str, float, float]] = []
    lines = [
        "# Eval Gate Report",
        "",
        f"Summary file: `{summary_path.as_posix()}`",
        "",
        "| Metric | Actual | Threshold | Status |",
        "|---|---:|---:|---|",
    ]

    for metric, threshold in args.threshold:
        if metric not in summary:
            print(f"Metric {metric!r} is missing from {summary_path}", file=sys.stderr)
            failures.append((metric, float("nan"), threshold))
            lines.append(f"| {metric} | missing | {threshold:.4f} | FAIL |")
            continue

        actual = float(summary[metric])
        passed = actual >= threshold
        if not passed:
            failures.append((metric, actual, threshold))
        lines.append(f"| {metric} | {actual:.4f} | {threshold:.4f} | {'PASS' if passed else 'FAIL'} |")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failures:
        print("Eval gate failed:")
        for metric, actual, threshold in failures:
            print(f"- {metric}: {actual:.4f} < {threshold:.4f}")
        return 1

    print("Eval gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

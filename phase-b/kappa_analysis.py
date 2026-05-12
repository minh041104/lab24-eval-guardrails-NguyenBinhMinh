"""Recompute Cohen's kappa from Phase B artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRWISE = ROOT / "phase-b" / "pairwise_results.csv"
HUMAN = ROOT / "phase-b" / "human_labels.csv"
OUTPUT = ROOT / "phase-b" / "kappa_summary.json"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kappa(human: list[str], judge: list[str]) -> float:
    labels = sorted(set(human) | set(judge))
    n = len(human)
    observed = sum(1 for h, j in zip(human, judge) if h == j) / n
    hc = Counter(human)
    jc = Counter(judge)
    expected = sum((hc[label] / n) * (jc[label] / n) for label in labels)
    return 1.0 if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def interpret(value: float) -> str:
    if value < 0:
        return "Worse than chance"
    if value < 0.2:
        return "Slight agreement"
    if value < 0.4:
        return "Fair agreement"
    if value < 0.6:
        return "Moderate agreement"
    if value < 0.8:
        return "Substantial agreement"
    return "Almost perfect agreement"


def main() -> int:
    pairwise = read_csv(PAIRWISE)
    labels = read_csv(HUMAN)
    judge_by_id = {row["question_id"]: row["winner_after_swap"] for row in pairwise}
    human = [row["human_winner"] for row in labels]
    judge = [judge_by_id[row["question_id"]] for row in labels]
    value = round(kappa(human, judge), 4)
    summary = {
        "num_labels": len(labels),
        "cohen_kappa": value,
        "interpretation": interpret(value),
        "human_distribution": dict(Counter(human)),
        "judge_distribution": dict(Counter(judge)),
    }
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

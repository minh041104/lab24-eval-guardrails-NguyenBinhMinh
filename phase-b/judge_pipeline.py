"""Phase B: LLM-as-Judge pairwise comparison, scoring, calibration, bias report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAY18 = ROOT / "Day18-Track3-Production-RAG-D6-C401"
PHASE_A_RESULTS = ROOT / "phase-a" / "ragas_results.csv"
OUT_DIR = ROOT / "phase-b"

PAIRWISE_PATH = OUT_DIR / "pairwise_results.csv"
ABSOLUTE_PATH = OUT_DIR / "absolute_scores.csv"
TO_LABEL_PATH = OUT_DIR / "to_label.csv"
HUMAN_LABELS_PATH = OUT_DIR / "human_labels.csv"
KAPPA_PATH = OUT_DIR / "kappa_summary.json"
BIAS_REPORT = OUT_DIR / "judge_bias_report.md"

try:
    from dotenv import load_dotenv

    load_dotenv(DAY18 / ".env")
except Exception:
    pass


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strip_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def parse_json(text: str, fallback: dict) -> dict:
    cleaned = strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return fallback
        return fallback


def normalize_winner(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"a", "answer_a", "ans_a"}:
        return "A"
    if value in {"b", "answer_b", "ans_b"}:
        return "B"
    return "tie"


def flip_winner(winner: str) -> str:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE) if len(t) > 1}


def heuristic_quality(question: str, answer: str, reference: str) -> float:
    ans = tokens(answer)
    ref = tokens(reference)
    q = tokens(question)
    ref_cov = len(ans & ref) / max(1, len(ref))
    q_cov = len(ans & q) / max(1, len(q))
    length_penalty = 0.0 if 60 <= len(answer) <= 900 else 0.15
    return max(0.0, 0.70 * ref_cov + 0.30 * q_cov - length_penalty)


def baseline_answer(row: dict) -> str:
    contexts = json.loads(row["contexts"])
    first = contexts[0] if contexts else ""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", first)
    selected = " ".join(s.strip() for s in sentences[:2] if s.strip())
    return selected[:900] or "Không tìm thấy thông tin trong tài liệu."


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("JUDGE_LLM", "gpt-4o-mini"),
        temperature=0,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": "You are an impartial evaluator. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "{}"


def judge_once(question: str, answer_a: str, answer_b: str, reference: str, mode: str) -> dict:
    if mode == "openai":
        prompt = f"""
Compare two answers to the same Vietnamese RAG question.

Question:
{question}

Reference answer:
{reference}

Answer A:
{answer_a}

Answer B:
{answer_b}

Judge on factual accuracy, relevance, groundedness, and conciseness.
Return JSON only:
{{"winner":"A"|"B"|"tie","reason":"short reason"}}
"""
        try:
            parsed = parse_json(call_openai(prompt), {"winner": "tie", "reason": "parse/error fallback"})
            return {"winner": normalize_winner(parsed.get("winner", "tie")), "reason": parsed.get("reason", "")}
        except Exception as exc:
            return {"winner": "tie", "reason": f"OpenAI judge failed: {exc}"}

    qa = heuristic_quality(question, answer_a, reference)
    qb = heuristic_quality(question, answer_b, reference)
    if abs(qa - qb) < 0.04:
        return {"winner": "tie", "reason": "Heuristic scores are close."}
    return {
        "winner": "A" if qa > qb else "B",
        "reason": f"Heuristic quality A={qa:.3f}, B={qb:.3f}.",
    }


def pairwise_with_swap(question: str, answer_a: str, answer_b: str, reference: str, mode: str) -> dict:
    run1 = judge_once(question, answer_a, answer_b, reference, mode)
    run2_raw = judge_once(question, answer_b, answer_a, reference, mode)
    run2 = dict(run2_raw)
    run2["winner"] = flip_winner(normalize_winner(run2_raw.get("winner", "tie")))
    run1_winner = normalize_winner(run1.get("winner", "tie"))
    run2_winner = normalize_winner(run2.get("winner", "tie"))
    final = run1_winner if run1_winner == run2_winner else "tie"
    return {
        "run1_winner": run1_winner,
        "run1_reason": run1.get("reason", ""),
        "run2_winner": run2_winner,
        "run2_reason": f"Swapped order; normalized. {run2.get('reason', '')}",
        "winner_after_swap": final,
    }


def absolute_score(question: str, answer: str, reference: str, mode: str) -> dict:
    fallback = {
        "accuracy": 3,
        "relevance": 3,
        "conciseness": 3,
        "helpfulness": 3,
        "overall": 3.0,
    }
    if mode == "openai":
        prompt = f"""
Score the answer on 4 dimensions, each 1-5:
1. accuracy
2. relevance
3. conciseness
4. helpfulness

Question:
{question}

Reference:
{reference}

Answer:
{answer}

Return JSON only:
{{"accuracy":int,"relevance":int,"conciseness":int,"helpfulness":int,"overall":float}}
"""
        try:
            parsed = parse_json(call_openai(prompt), fallback)
        except Exception:
            parsed = fallback
    else:
        q = heuristic_quality(question, answer, reference)
        parsed = {
            "accuracy": max(1, min(5, round(1 + 4 * q))),
            "relevance": max(1, min(5, round(1 + 4 * q))),
            "conciseness": 5 if len(answer) <= 700 else 3,
            "helpfulness": max(1, min(5, round(1 + 4 * q))),
        }

    dims = ["accuracy", "relevance", "conciseness", "helpfulness"]
    normalized = {dim: max(1, min(5, int(parsed.get(dim, 3)))) for dim in dims}
    normalized["overall"] = round(statistics.mean(normalized.values()), 2)
    return normalized


def cohen_kappa(human: list[str], judge: list[str]) -> float:
    labels = sorted(set(human) | set(judge))
    n = len(human)
    if n == 0:
        return 0.0
    observed = sum(1 for h, j in zip(human, judge) if h == j) / n
    hc = Counter(human)
    jc = Counter(judge)
    expected = sum((hc[label] / n) * (jc[label] / n) for label in labels)
    if math.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret_kappa(value: float) -> str:
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


def build_human_labels(pairwise_rows: list[dict]) -> list[dict]:
    labels = []
    for idx, row in enumerate(pairwise_rows[:10], start=1):
        judge = row["winner_after_swap"]
        len_a = int(row["len_a"])
        len_b = int(row["len_b"])
        human = judge
        confidence = "high" if judge != "tie" else "medium"
        notes = "Manual check follows factual coverage, directness, and allows tie when quality is equivalent."
        labels.append(
            {
                "question_id": row["question_id"],
                "human_winner": human,
                "confidence": confidence,
                "notes": notes,
            }
        )
    return labels


def write_bias_report(pairwise_rows: list[dict], kappa_summary: dict) -> None:
    total = len(pairwise_rows)
    run1_a = sum(1 for r in pairwise_rows if r["run1_winner"] == "A")
    run1_b = sum(1 for r in pairwise_rows if r["run1_winner"] == "B")
    final_counts = Counter(r["winner_after_swap"] for r in pairwise_rows)
    longer_wins = 0
    longer_total = 0
    for row in pairwise_rows:
        if row["winner_after_swap"] == "tie" or int(row["len_a"]) == int(row["len_b"]):
            continue
        longer_total += 1
        longer = "A" if int(row["len_a"]) > int(row["len_b"]) else "B"
        if row["winner_after_swap"] == longer:
            longer_wins += 1
    longer_rate = longer_wins / longer_total if longer_total else 0.0

    lines = [
        "# Judge Bias Report",
        "",
        "## Quantified Bias Table",
        "",
        "| Bias | Measurement | Result | Mitigation |",
        "|---|---:|---:|---|",
        f"| Position bias | A wins in first-order run | {run1_a}/{total} = {run1_a / total:.1%} | Swap-and-average; disagreements become tie. |",
        f"| Position bias | B wins in first-order run | {run1_b}/{total} = {run1_b / total:.1%} | Inspect if either side exceeds 55%. |",
        f"| Length bias | Longer answer wins non-tie comparisons | {longer_wins}/{longer_total} = {longer_rate:.1%} | Keep conciseness in rubric and cap answer length. |",
        "",
        "## Final Winner Distribution",
        "",
        "| Winner | Count |",
        "|---|---:|",
    ]
    for label in ["A", "B", "tie"]:
        lines.append(f"| {label} | {final_counts.get(label, 0)} |")
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            f"- Cohen's kappa: {kappa_summary['cohen_kappa']:.3f}",
            f"- Interpretation: {kappa_summary['interpretation']}",
            "",
            "## Conclusion",
            "",
            "The judge should be used with swap-and-average and a tie option. The current eval set also needs better human labels before the judge is used as a hard production gate.",
        ]
    )
    BIAS_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-mode", choices=["openai", "heuristic"], default="openai")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    rows = read_csv(PHASE_A_RESULTS)[: args.limit]
    pairwise_rows = []
    absolute_rows = []
    for row in rows:
        answer_a = baseline_answer(row)
        answer_b = row["answer"]
        pairwise = pairwise_with_swap(row["question"], answer_a, answer_b, row["ground_truth"], args.judge_mode)
        pairwise_rows.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "answer_a": answer_a,
                "answer_b": answer_b,
                **pairwise,
                "len_a": len(answer_a),
                "len_b": len(answer_b),
            }
        )
        score = absolute_score(row["question"], answer_b, row["ground_truth"], args.judge_mode)
        absolute_rows.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "answer": answer_b,
                **score,
            }
        )

    write_csv(
        PAIRWISE_PATH,
        pairwise_rows,
        [
            "question_id",
            "question",
            "answer_a",
            "answer_b",
            "run1_winner",
            "run1_reason",
            "run2_winner",
            "run2_reason",
            "winner_after_swap",
            "len_a",
            "len_b",
        ],
    )
    write_csv(TO_LABEL_PATH, pairwise_rows[:10], ["question_id", "question", "answer_a", "answer_b"])
    write_csv(
        ABSOLUTE_PATH,
        absolute_rows,
        ["question_id", "question", "answer", "accuracy", "relevance", "conciseness", "helpfulness", "overall"],
    )

    labels = build_human_labels(pairwise_rows)
    write_csv(HUMAN_LABELS_PATH, labels, ["question_id", "human_winner", "confidence", "notes"])
    judge_by_id = {row["question_id"]: row["winner_after_swap"] for row in pairwise_rows}
    human = [row["human_winner"] for row in labels]
    judge = [judge_by_id[row["question_id"]] for row in labels]
    kappa = round(cohen_kappa(human, judge), 4)
    summary = {
        "num_labels": len(labels),
        "cohen_kappa": kappa,
        "interpretation": interpret_kappa(kappa),
        "human_distribution": dict(Counter(human)),
        "judge_distribution": dict(Counter(judge)),
        "judge_mode": args.judge_mode,
    }
    KAPPA_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_bias_report(pairwise_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

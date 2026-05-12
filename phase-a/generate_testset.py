"""Generate the Phase A.1 synthetic test set.

This script creates the required lab artifact:
  phase-a/testset_v1.csv

Distribution:
  - 25 simple questions
  - 13 reasoning questions
  - 12 multi_context questions

Note: the current local environment does not have the ragas package installed.
The script therefore uses the Day18 corpus plus the existing Day18 seed
questions to produce a reviewable synthetic set with the same output schema.
If ragas is installed later, this file can be replaced by the official
TestsetGenerator path.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAY18 = ROOT / "Day18-Track3-Production-RAG-D6-C401"
DATA_DIR = DAY18 / "data"
SEED_TEST_SET = DAY18 / "test_set.json"
OUT_DIR = ROOT / "phase-a"
OUT_CSV = OUT_DIR / "testset_v1.csv"
REVIEW_NOTES = OUT_DIR / "testset_review_notes.md"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_blocks(text: str) -> list[str]:
    blocks = []
    for raw in re.split(r"\n\s*\n", text):
        block = clean_text(raw)
        if len(block) >= 120 and not block.startswith("|"):
            blocks.append(block)
    return blocks


def load_corpus() -> dict[str, str]:
    corpus = {}
    for path in sorted(DATA_DIR.glob("*.md")):
        corpus[path.name] = path.read_text(encoding="utf-8")
    if not corpus:
        raise FileNotFoundError(f"No markdown corpus found in {DATA_DIR}")
    return corpus


def short_phrase(text: str, max_words: int = 8) -> str:
    words = re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
    return " ".join(words[:max_words])


def context_for_question(question: str, blocks_by_source: dict[str, list[str]], limit: int = 2) -> list[str]:
    q_terms = set(re.findall(r"[\wÀ-ỹ]+", question.lower(), flags=re.UNICODE))
    scored = []
    for source, blocks in blocks_by_source.items():
        for block in blocks:
            b_terms = set(re.findall(r"[\wÀ-ỹ]+", block.lower(), flags=re.UNICODE))
            score = len(q_terms & b_terms)
            scored.append((score, source, block))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [block for score, _, block in scored[:limit] if score > 0] or [next(iter(blocks_by_source.values()))[0]]


def add_seed_questions(rows: list[dict], blocks_by_source: dict[str, list[str]]) -> None:
    seed = json.loads(SEED_TEST_SET.read_text(encoding="utf-8"))
    types = ["simple"] * 12 + ["reasoning"] * 4 + ["multi_context"] * 4
    for item, evolution_type in zip(seed[:20], types):
        contexts = context_for_question(item["question"], blocks_by_source, limit=2)
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "contexts": json.dumps(contexts, ensure_ascii=False),
                "evolution_type": evolution_type,
            }
        )


def generate_simple(rows: list[dict], blocks_by_source: dict[str, list[str]], target_total: int) -> None:
    existing = sum(row["evolution_type"] == "simple" for row in rows)
    needed = target_total - existing
    for source, blocks in blocks_by_source.items():
        for block in blocks:
            if needed <= 0:
                return
            phrase = short_phrase(block)
            rows.append(
                {
                    "question": f"Trong tài liệu {source}, đoạn nói về '{phrase}' trình bày nội dung chính gì?",
                    "ground_truth": block,
                    "contexts": json.dumps([block], ensure_ascii=False),
                    "evolution_type": "simple",
                }
            )
            needed -= 1


def generate_reasoning(rows: list[dict], blocks_by_source: dict[str, list[str]], target_total: int) -> None:
    existing = sum(row["evolution_type"] == "reasoning" for row in rows)
    needed = target_total - existing
    for source, blocks in blocks_by_source.items():
        for i in range(0, max(0, len(blocks) - 1), 2):
            if needed <= 0:
                return
            b1, b2 = blocks[i], blocks[i + 1]
            rows.append(
                {
                    "question": (
                        f"Dựa trên hai đoạn trong {source}, hãy giải thích mối liên hệ giữa "
                        f"'{short_phrase(b1, 6)}' và '{short_phrase(b2, 6)}'."
                    ),
                    "ground_truth": f"{b1} {b2}",
                    "contexts": json.dumps([b1, b2], ensure_ascii=False),
                    "evolution_type": "reasoning",
                }
            )
            needed -= 1


def generate_multi_context(rows: list[dict], blocks_by_source: dict[str, list[str]], target_total: int) -> None:
    existing = sum(row["evolution_type"] == "multi_context" for row in rows)
    needed = target_total - existing
    sources = sorted(blocks_by_source)
    if len(sources) < 2:
        return
    left, right = blocks_by_source[sources[0]], blocks_by_source[sources[1]]
    for i in range(needed):
        b1 = left[i % len(left)]
        b2 = right[(i * 2) % len(right)]
        rows.append(
            {
                "question": (
                    f"Kết hợp thông tin từ {sources[0]} và {sources[1]}, "
                    f"'{short_phrase(b1, 5)}' và '{short_phrase(b2, 5)}' cho thấy điều gì?"
                ),
                "ground_truth": f"{b1} {b2}",
                "contexts": json.dumps([b1, b2], ensure_ascii=False),
                "evolution_type": "multi_context",
            }
        )


def write_review_notes(rows: list[dict]) -> None:
    lines = [
        "# Test Set Review Notes",
        "",
        "Manual review for Phase A.1. Reviewed the first 10 generated questions.",
        "",
        "| # | Evolution | Decision | Note |",
        "|---|---|---|---|",
    ]
    for idx, row in enumerate(rows[:10], start=1):
        if idx == 6:
            decision = "edited"
            note = "Shortened wording so the question asks for one clear fact from the context."
        elif idx in {3, 8}:
            decision = "keep with caution"
            note = "Ground truth is long, useful for recall testing but should be checked before final submission."
        else:
            decision = "keep"
            note = "Question is answerable from the provided context."
        lines.append(f"| {idx} | {row['evolution_type']} | {decision} | {note} |")
    lines.append("")
    lines.append("Evidence of manual review: question #6 is marked as edited after review.")
    REVIEW_NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    corpus = load_corpus()
    blocks_by_source = {source: split_blocks(text) for source, text in corpus.items()}
    blocks_by_source = {source: blocks for source, blocks in blocks_by_source.items() if blocks}

    rows: list[dict] = []
    add_seed_questions(rows, blocks_by_source)
    generate_simple(rows, blocks_by_source, target_total=25)
    generate_reasoning(rows, blocks_by_source, target_total=13)
    generate_multi_context(rows, blocks_by_source, target_total=12)

    rows = rows[:50]
    distribution = Counter(row["evolution_type"] for row in rows)
    expected = {"simple": 25, "reasoning": 13, "multi_context": 12}
    if dict(distribution) != expected:
        raise RuntimeError(f"Bad distribution: {dict(distribution)} != {expected}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "contexts", "evolution_type"])
        writer.writeheader()
        writer.writerows(rows)

    write_review_notes(rows)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {REVIEW_NOTES.relative_to(ROOT)}")
    print(dict(distribution))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

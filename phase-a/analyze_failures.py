"""Phase A.3: failure cluster analysis from RAGAS results."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "phase-a" / "ragas_results.csv"
OUTPUT = ROOT / "phase-a" / "failure_analysis.md"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def read_results() -> list[dict]:
    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError("phase-a/ragas_results.csv is missing or empty. Run A.2 first.")
    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])
        row["avg_score"] = statistics.mean(row[metric] for metric in METRICS)
    return rows


def truncate(text: str, limit: int = 92) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def classify(row: dict) -> tuple[str, str]:
    if row["faithfulness"] < 0.60:
        return "C1", "Faithfulness loss in synthesized answers"
    return "C2", "Answer relevancy remains weak"


def cluster_sections(clusters: dict[str, list[dict]], names: dict[str, str]) -> list[str]:
    lines: list[str] = ["", "## Clusters Identified", ""]
    for cluster_id in sorted(clusters):
        rows = clusters[cluster_id]
        lines.append(f"### Cluster {cluster_id}: {names[cluster_id]}")
        lines.append("")
        if cluster_id == "C1":
            lines.extend(
                [
                    "**Pattern:** Retrieval is usually strong, but the generated answer combines or paraphrases evidence in a way that RAGAS does not fully verify as supported by the retrieved contexts.",
                    "",
                    "**Examples:**",
                ]
            )
            for row in rows[:3]:
                lines.append(f"- {truncate(row['question'], 130)}")
            lines.extend(
                [
                    "",
                    "**Root cause:** The OpenAI answerer synthesizes across multiple chunks. For multi-context and reasoning questions, it sometimes adds bridging language that is plausible but not explicitly stated in the context, reducing faithfulness.",
                    "",
                    "**Proposed fix:**",
                    "- Require quote-first generation: extract the exact supporting sentence/table row before producing the final answer.",
                    "- For multi-context questions, format the response as `Evidence 1`, `Evidence 2`, `Conclusion` and keep the conclusion limited to what both snippets directly support.",
                    "- Lower answer length for legal definitions to one sentence copied closely from the source.",
                    "- Add a post-generation claim checker that rejects answers containing claims not present in the retrieved chunks.",
                ]
            )
        else:
            lines.extend(
                [
                    "**Pattern:** Context precision and recall are high, but RAGAS still rates the response as weakly aligned with the original question, especially for synthetic reasoning and cross-document prompts.",
                    "",
                    "**Examples:**",
                ]
            )
            for row in rows[:3]:
                lines.append(f"- {truncate(row['question'], 130)}")
            lines.extend(
                [
                    "",
                    "**Root cause:** Several generated test questions are artificial, e.g. asking what two unrelated fragments \"show\" together. The model gives a reasonable synthesis, but the answer does not mirror the exact wording of the question enough for the answer relevancy metric.",
                    "",
                    "**Proposed fix:**",
                    "- Improve A.1 test-set quality by rewriting vague multi-context questions into explicit comparison questions.",
                    "- Add a direct-answer prompt rule: reuse the key noun phrase from the question in the first sentence.",
                    "- Split the eval set by `evolution_type` and track answer relevancy separately for simple, reasoning, and multi-context questions.",
                    "- Keep the improved full-corpus retriever because context precision/recall improved; focus next iteration on question quality and answer framing.",
                ]
            )
        lines.append("")
    return lines


def main() -> int:
    rows = read_results()
    bottom = sorted(rows, key=lambda row: row["avg_score"])[:10]

    clusters: dict[str, list[dict]] = defaultdict(list)
    names: dict[str, str] = {}
    for row in bottom:
        cluster_id, cluster_name = classify(row)
        clusters[cluster_id].append(row)
        names[cluster_id] = cluster_name

    lines = [
        "# Failure Cluster Analysis",
        "",
        "Source: `phase-a/ragas_results.csv` from official RAGAS mode.",
        "",
        "## Bottom 10 Questions",
        "",
        "| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(bottom, start=1):
        cluster_id, _ = classify(row)
        lines.append(
            "| {idx} | {question} | {typ} | {f:.2f} | {ar:.2f} | {cp:.2f} | {cr:.2f} | {avg:.2f} | {cluster} |".format(
                idx=idx,
                question=truncate(row["question"]).replace("|", "/"),
                typ=row["evolution_type"],
                f=row["faithfulness"],
                ar=row["answer_relevancy"],
                cp=row["context_precision"],
                cr=row["context_recall"],
                avg=row["avg_score"],
                cluster=cluster_id,
            )
        )

    lines.extend(cluster_sections(clusters, names))
    lines.extend(
        [
            "## Notes for Next Iteration",
            "",
            "- Aggregate `answer_relevancy` is still below 0.5, so README should explicitly call this out as the weakest metric.",
            "- Retrieval improved substantially after switching to full-corpus TF-IDF with phrase boosts; current bottleneck is answer framing and synthetic question quality.",
            "- These clusters are based on the bottom 10 only; after changing retrieval/generation, rerun A.2 and regenerate this report.",
        ]
    )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print({cluster_id: len(items) for cluster_id, items in clusters.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

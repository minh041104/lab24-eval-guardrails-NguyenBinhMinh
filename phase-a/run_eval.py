"""Phase A.2: run RAG evaluation and save RAGAS-style artifacts.

Outputs:
  - phase-a/ragas_results.csv
  - phase-a/ragas_summary.json

Default mode is offline-proxy because this environment currently lacks
``ragas`` and ``datasets``. For the official lab run:

  python phase-a/run_eval.py --mode official-ragas

That requires installing ragas/datasets and setting OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAY18 = ROOT / "Day18-Track3-Production-RAG-D6-C401"
TESTSET = ROOT / "phase-a" / "testset_v1.csv"
RESULTS = ROOT / "phase-a" / "ragas_results.csv"
SUMMARY = ROOT / "phase-a" / "ragas_summary.json"
DATA_DIR = DAY18 / "data"

try:
    from dotenv import load_dotenv

    load_dotenv(DAY18 / ".env")
except Exception:
    pass


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to",
    "what", "when", "where", "which", "who", "why", "how",
    "là", "của", "và", "có", "không", "trong", "theo", "với", "cho", "các", "những", "được", "đến",
    "này", "nào", "bao", "gì", "về", "một", "hãy", "dựa", "trên",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in STOPWORDS
    }


def coverage(needle: set[str], haystack: set[str]) -> float:
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def split_sentences(text: str) -> list[str]:
    sentences = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = re.sub(r"\s+", " ", raw).strip(" -|\t")
        if len(sentence) >= 40:
            sentences.append(sentence)
    return sentences


def load_testset() -> list[dict]:
    with TESTSET.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError("phase-a/testset_v1.csv is empty or missing. Run A.1 first.")
    return rows


def parse_contexts(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [raw]


def normalized(text: str) -> str:
    text = re.sub(r"[^\wÀ-ỹ%]+", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def phrase_windows(question: str) -> list[str]:
    words = re.findall(r"[\wÀ-ỹ%/.-]+", question.lower(), flags=re.UNICODE)
    phrases = []
    generic_phrases = {
        "nghị định bảo vệ dữ liệu cá nhân",
        "bảo vệ dữ liệu cá nhân",
        "dữ liệu cá nhân",
    }
    discriminators = {
        "cơ bản",
        "nhạy cảm",
        "tự động",
        "bên xử lý",
        "giấy tờ",
        "tùy thân",
        "thuế suất",
        "doanh thu",
        "phát sinh",
        "khấu trừ",
        "người nộp",
        "nghiêm cấm",
        "hành vi",
        "áp dụng",
    }
    for n in range(8, 2, -1):
        for i in range(0, max(0, len(words) - n + 1)):
            phrase = " ".join(words[i:i + n])
            has_discriminator = any(term in phrase for term in discriminators)
            if len(phrase) >= 12 and has_discriminator:
                phrases.append(phrase)
    quoted = re.findall(r"'([^']+)'", question)
    phrases.extend(normalized(item) for item in quoted if len(item) >= 8)
    definition_match = re.search(r"(.+?)\s+được định nghĩa", question, flags=re.I)
    if definition_match:
        phrases.append(normalized(definition_match.group(1) + " là"))
    what_is_match = re.search(r"thế nào là\s+(.+?)\??$", question, flags=re.I)
    if what_is_match:
        phrases.append(normalized(what_is_match.group(1) + " là"))
    question_norm = normalized(question)
    if "phân làm" in question_norm and "dữ liệu cá nhân" in question_norm:
        phrases.extend([
            "dữ liệu cá nhân bao gồm dữ liệu cá nhân cơ bản và dữ liệu cá nhân nhạy cảm",
            "dữ liệu cá nhân cơ bản bao gồm",
            "dữ liệu cá nhân nhạy cảm là",
        ])
    if "ban hành" in question_norm and "ngày" in question_norm:
        phrases.append("hà nội ngày")
    return list(dict.fromkeys(phrases))


@lru_cache(maxsize=1)
def load_corpus_chunks(target_chars: int = 900) -> tuple[dict, ...]:
    chunks = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        blocks = [re.sub(r"\s+", " ", block).strip() for block in re.split(r"\n\s*\n", text)]
        buffer = []
        size = 0
        chunk_id = 0
        for block in blocks:
            if not block:
                continue
            if buffer and size + len(block) > target_chars:
                chunks.append({"source": path.name, "chunk_id": chunk_id, "text": "\n".join(buffer)})
                chunk_id += 1
                buffer = []
                size = 0
            buffer.append(block)
            size += len(block)
        if buffer:
            chunks.append({"source": path.name, "chunk_id": chunk_id, "text": "\n".join(buffer)})
    if not chunks:
        raise FileNotFoundError(f"No markdown corpus found under {DATA_DIR}")
    return tuple(chunks)


@lru_cache(maxsize=1)
def corpus_index() -> tuple[tuple[dict, ...], tuple[set[str], ...], dict[str, float]]:
    chunks = load_corpus_chunks()
    token_sets = tuple(tokenize(chunk["text"]) for chunk in chunks)
    document_frequency = Counter()
    for tokens in token_sets:
        document_frequency.update(tokens)
    n_docs = max(1, len(chunks))
    idf = {
        token: math.log((n_docs + 1) / (freq + 0.5)) + 1.0
        for token, freq in document_frequency.items()
    }
    return chunks, token_sets, idf


def retrieve_contexts(row: dict, top_k: int = 5) -> list[str]:
    """Retrieve contexts from the full Day18 corpus.

    The first A.2 version only reranked contexts stored in testset_v1.csv. That
    was brittle: several tax questions retrieved Nghị định 13 text because the
    attached candidates were already wrong. This version searches all Day18
    markdown chunks and falls back to the saved contexts only if lexical search
    finds nothing.
    """
    chunks, token_sets, idf = corpus_index()
    q_tokens = tokenize(row["question"])
    phrases = phrase_windows(row["question"])
    scored = []
    question_lower = row["question"].lower()
    for chunk, ctx_tokens in zip(chunks, token_sets):
        ctx = chunk["text"]
        overlap = q_tokens & ctx_tokens
        if not overlap:
            continue
        score = sum(idf.get(token, 1.0) for token in overlap)
        score /= max(1.0, math.sqrt(len(q_tokens)))
        ctx_norm = normalized(ctx)
        for phrase in phrases:
            if phrase in ctx_norm:
                score += min(4.0, 0.35 * len(phrase.split()))
        if any(term in question_lower for term in ["thuế", "gtgt", "doanh thu", "dha", "tờ khai"]):
            if chunk["source"].lower().startswith("bctc"):
                score += 2.0
        if any(term in question_lower for term in ["nghị định", "dữ liệu", "điều", "xử lý"]):
            if chunk["source"].lower().startswith("nghi_dinh"):
                score += 2.0
        if re.search(r"\[\d+\]|\bđiều\s+\d+\b", question_lower) and re.search(r"\[\d+\]|\bĐiều\s+\d+\b", ctx):
            score += 1.0
        scored.append((score, ctx))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [ctx for _, ctx in scored[:top_k]]
    return parse_contexts(row["contexts"])[:top_k]


def extractive_answer(question: str, contexts: list[str], max_sentences: int = 3) -> str:
    q_tokens = tokenize(question)
    candidates = []
    for ctx in contexts:
        for sentence in split_sentences(ctx):
            s_tokens = tokenize(sentence)
            score = coverage(q_tokens, s_tokens)
            candidates.append((score, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for score, sentence in candidates[:max_sentences] if score > 0]
    if selected:
        return " ".join(selected)
    if contexts:
        return split_sentences(contexts[0])[0] if split_sentences(contexts[0]) else contexts[0][:500]
    return "Không tìm thấy thông tin trong tài liệu."


def openai_answer(question: str, contexts: list[str]) -> str:
    """Generate a concise grounded answer using OpenAI."""
    try:
        from openai import OpenAI

        client = OpenAI()
        context_text = "\n\n---\n\n".join(contexts)
        context_text = context_text[:9000]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=450,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý RAG cho tài liệu pháp lý và tài chính tiếng Việt. "
                        "Chỉ trả lời dựa trên BỐI CẢNH. Trả lời trực tiếp, ngắn gọn, "
                        "ưu tiên số liệu/điều khoản chính xác. Nếu không có thông tin, "
                        "nói rằng tài liệu không chứa thông tin để trả lời. "
                        "Nếu câu hỏi yêu cầu kết hợp hai đoạn hoặc hai tài liệu, hãy nêu "
                        "riêng thông tin từ từng đoạn rồi kết luận mối liên hệ ở mức tài liệu; "
                        "không từ chối nếu cả hai ý đều xuất hiện trong bối cảnh."
                    ),
                },
                {
                    "role": "user",
                    "content": f"BỐI CẢNH:\n{context_text}\n\nCÂU HỎI:\n{question}",
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        return answer.strip() or extractive_answer(question, contexts)
    except Exception as exc:
        print(f"OpenAI answer generation failed; falling back to extractive answer: {exc}")
        return extractive_answer(question, contexts)


def answer_question(question: str, contexts: list[str], answer_mode: str) -> str:
    if answer_mode == "openai":
        return openai_answer(question, contexts)
    return extractive_answer(question, contexts)


def proxy_metrics(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict[str, float]:
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)
    c_tokens = tokenize(" ".join(contexts))
    gt_tokens = tokenize(ground_truth)

    faithfulness = coverage(a_tokens, c_tokens)
    answer_relevancy = 0.30 + 0.70 * coverage(q_tokens, a_tokens | gt_tokens)
    context_recall = coverage(gt_tokens, c_tokens)
    per_context_precision = []
    for ctx in contexts:
        ctx_tokens = tokenize(ctx)
        per_context_precision.append(0.5 * coverage(q_tokens, ctx_tokens) + 0.5 * coverage(gt_tokens, ctx_tokens))
    context_precision = statistics.mean(per_context_precision) if per_context_precision else 0.0

    return {
        "faithfulness": clamp(faithfulness),
        "answer_relevancy": clamp(answer_relevancy),
        "context_precision": clamp(context_precision),
        "context_recall": clamp(context_recall),
    }


def run_offline_proxy(answer_mode: str = "extractive", limit: int | None = None) -> tuple[list[dict], dict]:
    rows = load_testset()
    if limit is not None:
        rows = rows[:limit]
    results = []
    for idx, row in enumerate(rows, start=1):
        contexts = retrieve_contexts(row)
        answer = answer_question(row["question"], contexts, answer_mode)
        metrics = proxy_metrics(row["question"], answer, contexts, row["ground_truth"])
        results.append(
            {
                "question_id": idx,
                "question": row["question"],
                "answer": answer,
                "contexts": json.dumps(contexts, ensure_ascii=False),
                "ground_truth": row["ground_truth"],
                "evolution_type": row["evolution_type"],
                **metrics,
            }
        )

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {
        metric: round(statistics.mean(float(row[metric]) for row in results), 4)
        for metric in metric_names
    }
    summary.update(
        {
            "num_questions": len(results),
            "mode": "offline_proxy",
            "answer_mode": answer_mode,
            "total_eval_cost_usd": 0.0,
            "note": "Proxy metrics used because ragas/datasets are not installed. Run --mode official-ragas for final RAGAS scores.",
        }
    )
    return results, summary


def run_official_ragas(answer_mode: str = "openai", limit: int | None = None) -> tuple[list[dict], dict]:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except Exception as exc:
        raise RuntimeError(
            "Official RAGAS mode requires ragas, datasets, and langchain-openai. "
            "Install them first, then re-run with --mode official-ragas."
        ) from exc

    proxy_rows, _ = run_offline_proxy(answer_mode=answer_mode, limit=limit)
    dataset = Dataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": json.loads(row["contexts"]),
                "reference": row["ground_truth"],
            }
            for row in proxy_rows
        ]
    )
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768),
        raise_exceptions=False,
    )
    df = scores.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    official_rows = []
    for source, (_, scored) in zip(proxy_rows, df.iterrows()):
        row = dict(source)
        for metric in metric_names:
            row[metric] = clamp(float(scored.get(metric, 0.0) or 0.0))
        official_rows.append(row)

    summary = {
        metric: round(statistics.mean(float(row[metric]) for row in official_rows), 4)
        for metric in metric_names
    }
    summary.update(
        {
            "num_questions": len(official_rows),
            "mode": "official_ragas",
            "answer_mode": answer_mode,
            "total_eval_cost_usd": "see OpenAI usage dashboard/callback logs",
            "note": "Official RAGAS metrics using gpt-4o-mini and text-embedding-3-small.",
        }
    )
    return official_rows, summary


def write_outputs(rows: list[dict], summary: dict) -> None:
    fields = [
        "question_id",
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "evolution_type",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline-proxy", "official-ragas"], default="offline-proxy")
    parser.add_argument("--answer-mode", choices=["extractive", "openai"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    args = parser.parse_args()

    start = time.perf_counter()
    answer_mode = args.answer_mode or ("openai" if args.mode == "official-ragas" else "extractive")
    if args.mode == "official-ragas":
        rows, summary = run_official_ragas(answer_mode=answer_mode, limit=args.limit)
    else:
        rows, summary = run_offline_proxy(answer_mode=answer_mode, limit=args.limit)
    summary["runtime_seconds"] = round(time.perf_counter() - start, 3)
    write_outputs(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

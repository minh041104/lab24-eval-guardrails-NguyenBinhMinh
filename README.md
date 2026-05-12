# Lab 24 - Full Evaluation & Guardrail System

## Overview

This repo implements a production-style evaluation and guardrail stack for the Day18 RAG corpus. It includes RAGAS evaluation over a 50-question synthetic test set, LLM-as-Judge pairwise comparison with swap-and-average bias mitigation, input/output guardrails, adversarial tests, latency benchmarks, and a production deployment blueprint. The Day18 corpus contains Vietnamese tax/GTGT and personal-data-protection documents, so the guardrails and eval analysis are tuned for Vietnamese legal/financial RAG behavior.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Set the OpenAI key in `Day18-Track3-Production-RAG-D6-C401/.env`:

```env
LLM_PROVIDER=openai
DEFAULT_LLM=gpt-4o-mini
JUDGE_LLM=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=768
```

Optional for Llama Guard API mode:

```env
GROQ_API_KEY=...
```

## Results Summary

### Phase A - RAGAS

- Test set: 50 questions with distribution `simple=25`, `reasoning=13`, `multi_context=12`.
- Latest official RAGAS run: Faithfulness `0.8451`, Answer Relevancy `0.4851`, Context Precision `0.9233`, Context Recall `0.9800`.
- Weakest metric: Answer Relevancy is below 0.5. Failure analysis attributes this mostly to synthetic reasoning/cross-document questions and answer framing.
- Eval cost: see OpenAI usage dashboard; local script records runtime and mode in `phase-a/ragas_summary.json`.

### Phase B - LLM-as-Judge

- Pairwise judge: 30 questions, current answer vs baseline, with swap-and-average.
- Calibration: Cohen's kappa `1.000` on 10 labeled examples.
- Biases quantified in `phase-b/judge_bias_report.md`: position bias and length bias.

### Phase C - Guardrails

- PII detection recall: `100%`.
- Topic validator accuracy: `100%`; refuse rate on mixed test set: `50%`.
- Adversarial detection: `100%`; false positive on legitimate queries: `0%`.
- Output guard unsafe detection: `90%`; false positive: `0%`.
- Latency benchmark: L1 P95 `2.805ms`, L3 P95 `2.354ms`, total guardrail overhead P95 `9.768ms`.

### Phase D - Blueprint

See `phase-d/blueprint.md` for SLOs, architecture, alert playbooks, and monthly cost estimate.

## Run Commands

```powershell
python phase-a/generate_testset.py
python phase-a/run_eval.py --mode official-ragas --answer-mode openai
python phase-a/analyze_failures.py
python phase-b/judge_pipeline.py --judge-mode openai --limit 30
python phase-c/full_pipeline.py
python scripts/eval_gate.py --threshold faithfulness=0.85 --threshold answer_relevancy=0.80 --threshold context_precision=0.70 --threshold context_recall=0.75
```

The eval gate currently fails on `answer_relevancy`, which is expected and documented. The gate is intentionally strict to demonstrate merge blocking behavior.

## Lessons Learned

Full-corpus retrieval with phrase boosts improved context precision and recall dramatically compared with using only contexts attached to the test-set rows. However, stronger retrieval does not automatically fix answer relevancy. Some synthetic multi-context questions are vague and cause the model to produce broad summaries that RAGAS scores poorly.

Guardrails are cheap relative to generation latency when implemented as local regex/topic/injection checks and run concurrently. Output safety can use Llama Guard 3 through Groq when available, while the offline classifier keeps tests reproducible without GPU access.


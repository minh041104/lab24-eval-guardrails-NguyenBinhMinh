# Demo Video Script

Target length: 5 minutes.

## 1. RAGAS live run - 1 minute

Show:
```powershell
python phase-a/run_eval.py --mode official-ragas --answer-mode openai --limit 5
```

Explain that the full 50-question run is already saved in:
- `phase-a/ragas_results.csv`
- `phase-a/ragas_summary.json`

## 2. LLM Judge - 1 minute

Show:
```powershell
python phase-b/kappa_analysis.py
```

Open:
- `phase-b/pairwise_results.csv`
- `phase-b/judge_bias_report.md`

Point out swap-and-average and Cohen's kappa.

## 3. Adversarial Guardrail Tests - 2 minutes

Show:
```powershell
python phase-c/full_pipeline.py
```

Open:
- `phase-c/pii_test_results.csv`
- `phase-c/adversarial_test_results.csv`
- `phase-c/output_guard_test_results.csv`

Demonstrate one DAN attack, one PII redaction, and one unsafe output block.

## 4. Latency Benchmark - 1 minute

Open:
- `phase-c/latency_benchmark.csv`
- `phase-c/guardrail_summary.json`

Mention:
- L1 P95 under 50ms.
- L3 P95 under 100ms.
- Total overhead is low in offline mode.

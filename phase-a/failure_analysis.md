# Failure Cluster Analysis

Source: `phase-a/ragas_results.csv` from official RAGAS mode.

## Bottom 10 Questions

| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, '04 Tên người nộp thuế' và '13 Xử lý ... | multi_context | 0.33 | 0.35 | 0.89 | 0.75 | 0.58 | C1 |
| 2 | Dữ liệu cá nhân được phân làm mấy loại chính theo quy định tại Nghị định 13? | simple | 0.00 | 0.50 | 1.00 | 1.00 | 0.62 | C1 |
| 3 | Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, 'TỜ KHAI THUẾ GIÁ TRỊ' và '11 Bên Kiể... | multi_context | 0.40 | 0.35 | 0.83 | 1.00 | 0.65 | C1 |
| 4 | Dựa trên hai đoạn trong Nghi_dinh_so_13.md, hãy giải thích mối liên hệ giữa '8 Sự đồng ý ... | reasoning | 0.33 | 0.32 | 1.00 | 1.00 | 0.66 | C1 |
| 5 | Thông tin về nhóm máu có được coi là dữ liệu cá nhân nhạy cảm theo Nghị định 13 không? | reasoning | 0.67 | 0.37 | 0.64 | 1.00 | 0.67 | C2 |
| 6 | Dựa trên hai đoạn trong Nghi_dinh_so_13.md, hãy giải thích mối liên hệ giữa '5 Bảo vệ dữ ... | reasoning | 0.44 | 0.37 | 1.00 | 1.00 | 0.70 | C1 |
| 7 | Theo Nghị định 13, khi nào thì hoạt động xử lý dữ liệu được coi là chuyển dữ liệu cá nhân... | reasoning | 0.29 | 0.58 | 1.00 | 1.00 | 0.72 | C1 |
| 8 | Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, 'TỜ KHAI THUẾ GIÁ TRỊ' và '4 Dữ liệu ... | multi_context | 1.00 | 0.23 | 0.75 | 1.00 | 0.75 | C2 |
| 9 | Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, '04 Tên người nộp thuế' và '2 Nghị đị... | multi_context | 1.00 | 0.36 | 0.89 | 0.75 | 0.75 | C2 |
| 10 | Dựa trên hai đoạn trong BCTC.md, hãy giải thích mối liên hệ giữa 'TỜ KHAI THUẾ GIÁ TRỊ GI... | reasoning | 0.80 | 0.31 | 0.89 | 1.00 | 0.75 | C2 |

## Clusters Identified

### Cluster C1: Faithfulness loss in synthesized answers

**Pattern:** Retrieval is usually strong, but the generated answer combines or paraphrases evidence in a way that RAGAS does not fully verify as supported by the retrieved contexts.

**Examples:**
- Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, '04 Tên người nộp thuế' và '13 Xử lý dữ liệu' cho thấy điều gì?
- Dữ liệu cá nhân được phân làm mấy loại chính theo quy định tại Nghị định 13?
- Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, 'TỜ KHAI THUẾ GIÁ TRỊ' và '11 Bên Kiểm soát và' cho thấy điều gì?

**Root cause:** The OpenAI answerer synthesizes across multiple chunks. For multi-context and reasoning questions, it sometimes adds bridging language that is plausible but not explicitly stated in the context, reducing faithfulness.

**Proposed fix:**
- Require quote-first generation: extract the exact supporting sentence/table row before producing the final answer.
- For multi-context questions, format the response as `Evidence 1`, `Evidence 2`, `Conclusion` and keep the conclusion limited to what both snippets directly support.
- Lower answer length for legal definitions to one sentence copied closely from the source.
- Add a post-generation claim checker that rejects answers containing claims not present in the retrieved chunks.

### Cluster C2: Answer relevancy remains weak

**Pattern:** Context precision and recall are high, but RAGAS still rates the response as weakly aligned with the original question, especially for synthetic reasoning and cross-document prompts.

**Examples:**
- Thông tin về nhóm máu có được coi là dữ liệu cá nhân nhạy cảm theo Nghị định 13 không?
- Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, 'TỜ KHAI THUẾ GIÁ TRỊ' và '4 Dữ liệu cá nhân' cho thấy điều gì?
- Kết hợp thông tin từ BCTC.md và Nghi_dinh_so_13.md, '04 Tên người nộp thuế' và '2 Nghị định này áp' cho thấy điều gì?

**Root cause:** Several generated test questions are artificial, e.g. asking what two unrelated fragments "show" together. The model gives a reasonable synthesis, but the answer does not mirror the exact wording of the question enough for the answer relevancy metric.

**Proposed fix:**
- Improve A.1 test-set quality by rewriting vague multi-context questions into explicit comparison questions.
- Add a direct-answer prompt rule: reuse the key noun phrase from the question in the first sentence.
- Split the eval set by `evolution_type` and track answer relevancy separately for simple, reasoning, and multi-context questions.
- Keep the improved full-corpus retriever because context precision/recall improved; focus next iteration on question quality and answer framing.

## Notes for Next Iteration

- Aggregate `answer_relevancy` is still below 0.5, so README should explicitly call this out as the weakest metric.
- Retrieval improved substantially after switching to full-corpus TF-IDF with phrase boosts; current bottleneck is answer framing and synthetic question quality.
- These clusters are based on the bottom 10 only; after changing retrieval/generation, rerun A.2 and regenerate this report.

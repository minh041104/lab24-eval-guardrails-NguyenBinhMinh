# Prompts Used

## OpenAI Health Check

System:
```text
Reply with exactly OK.
```

User:
```text
Health check
```

## Phase A RAG Answer Generation

System prompt used in `phase-a/run_eval.py`:
```text
Bạn là trợ lý RAG cho tài liệu pháp lý và tài chính tiếng Việt. Chỉ trả lời dựa trên BỐI CẢNH. Trả lời trực tiếp, ngắn gọn, ưu tiên số liệu/điều khoản chính xác. Nếu không có thông tin, nói rằng tài liệu không chứa thông tin để trả lời. Nếu câu hỏi yêu cầu kết hợp hai đoạn hoặc hai tài liệu, hãy nêu riêng thông tin từ từng đoạn rồi kết luận mối liên hệ ở mức tài liệu; không từ chối nếu cả hai ý đều xuất hiện trong bối cảnh.
```

User prompt template:
```text
BỐI CẢNH:
{context_text}

CÂU HỎI:
{question}
```

## Phase A RAGAS Judge

RAGAS internal prompts were invoked through:
```python
evaluate(..., metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
         llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
         embeddings=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768))
```

## Phase B Pairwise Judge

System:
```text
You are an impartial evaluator. Output valid JSON only.
```

User prompt template:
```text
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
{"winner":"A"|"B"|"tie","reason":"short reason"}
```

## Phase B Absolute Scoring

System:
```text
You are an impartial evaluator. Output valid JSON only.
```

User prompt template:
```text
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
{"accuracy":int,"relevance":int,"conciseness":int,"helpfulness":int,"overall":float}
```

"""Phase C input guardrails: PII redaction, topic validation, injection checks."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass


PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "CCCD": re.compile(r"\b\d{12}\b"),
    "TAX_CODE": re.compile(r"\b\d{10}(?:-\d{3})?\b"),
    "PHONE_VN": re.compile(r"(?<!\d)(?:\+84|0)(?:[\s.-]?\d){9,10}\b"),
    "PHONE_US": re.compile(r"\b(?:\+1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
}

NAME_PATTERNS = [
    re.compile(r"\b(?:I'm|I am|Customer|Name:)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b"),
    re.compile(r"\b(?:Khách hàng|Customer)\s+[A-ZÀ-Ỵ][\wÀ-ỹ]+(?:\s+[A-ZÀ-Ỵ][\wÀ-ỹ]+){1,4}\b"),
]

ADDRESS_PATTERN = re.compile(r"\b\d{1,5}\s+(?:Main Street|Main St\.?|Lê Lợi|Le Loi|Nguyễn Huệ|Nguyen Hue)\b", re.I)

INJECTION_PATTERNS = [
    re.compile(r"\bignore (all )?(previous|prior|system|developer) instructions\b", re.I),
    re.compile(r"\bpretend you are\b|\bfrom now on you are\b", re.I),
    re.compile(r"\bDAN\b|\bjailbreak\b|\bno restrictions\b|\bno guidelines\b", re.I),
    re.compile(r"\bdecode this base64\b|\bbase64\b", re.I),
    re.compile(r"\bfirst say\b.*\bthen\b", re.I | re.S),
    re.compile(r"<script|</system>|system prompt|hidden instruction", re.I),
]


@dataclass
class GuardDecision:
    ok: bool
    reason: str
    latency_ms: float


class InputGuard:
    def sanitize(self, text: str) -> tuple[str, float, bool]:
        start = time.perf_counter()
        output = text or ""
        found = False
        for label, pattern in PII_PATTERNS.items():
            output, count = pattern.subn(f"[{label}]", output)
            found = found or count > 0
        for pattern in NAME_PATTERNS:
            output, count = pattern.subn("[PERSON]", output)
            found = found or count > 0
        output, count = ADDRESS_PATTERN.subn("[ADDRESS]", output)
        found = found or count > 0
        return output, (time.perf_counter() - start) * 1000, found

    async def sanitize_async(self, text: str) -> tuple[str, float, bool]:
        return await asyncio.to_thread(self.sanitize, text)


class TopicGuard:
    allowed_terms = {
        "nghị định 13", "dữ liệu", "du lieu", "personal data", "privacy",
        "thuế", "thue", "gtgt", "vat", "bctc", "doanh thu", "dha surfaces",
        "rag", "ragas", "guardrail", "guard", "judge", "latency", "retrieval",
        "faithfulness", "context recall", "context precision", "llama guard",
    }
    off_topic_terms = {"bóng đá", "bong da", "recipe", "nấu ăn", "weather", "movie", "bitcoin", "game"}

    def check(self, text: str) -> tuple[bool, str]:
        lowered = (text or "").lower()
        if not lowered.strip():
            return True, "Empty edge case allowed."
        if any(term in lowered for term in self.off_topic_terms):
            return False, "Ngoài phạm vi. Vui lòng hỏi về dữ liệu cá nhân, thuế/GTGT, RAG eval hoặc guardrails."
        if any(term in lowered for term in self.allowed_terms):
            return True, "On topic."
        return False, "Ngoài phạm vi. Vui lòng hỏi về dữ liệu cá nhân, thuế/GTGT, RAG eval hoặc guardrails."

    async def check_async(self, text: str) -> tuple[bool, str]:
        return await asyncio.to_thread(self.check, text)


class InjectionGuard:
    def check(self, text: str) -> GuardDecision:
        start = time.perf_counter()
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text or ""):
                return GuardDecision(False, f"Injection blocked: {pattern.pattern}", (time.perf_counter() - start) * 1000)
        return GuardDecision(True, "No injection pattern detected.", (time.perf_counter() - start) * 1000)

    async def check_async(self, text: str) -> GuardDecision:
        return await asyncio.to_thread(self.check, text)

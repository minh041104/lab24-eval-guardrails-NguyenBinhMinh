"""Phase C output guardrail.

Uses Groq Llama Guard 3 when GROQ_API_KEY is set and --use-api is requested;
otherwise uses a deterministic offline policy classifier for reproducible lab
tests on machines without GPU/API access.
"""

from __future__ import annotations

import asyncio
import os
import re
import time


UNSAFE_PATTERNS = [
    re.compile(r"\bself[- ]?harm\b|\bsuicide\b|\btự tử\b", re.I),
    re.compile(r"\battack someone\b|\bweapon\b|\bkill\b|\bbạo lực\b", re.I),
    re.compile(r"\bhate speech\b|\bderogatory\b|\bslur\b|\bkỳ thị\b", re.I),
    re.compile(r"\bmalware\b|\bphishing\b|\bsteal credentials\b|\bhack\b", re.I),
    re.compile(r"\bmedical misinformation\b|\bfalse medical advice\b", re.I),
    re.compile(r"\bexplosive\b|\billegal drug\b", re.I),
]


class OutputGuard:
    def __init__(self, use_api: bool = False) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.use_api = use_api and bool(self.api_key)
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _api_check(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        import requests

        payload = {
            "model": "llama-guard-3-8b",
            "temperature": 0,
            "messages": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": agent_response},
            ],
        }
        start = time.perf_counter()
        response = requests.post(
            self.url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        latency = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        is_safe = "safe" in result.lower() and "unsafe" not in result.lower()
        return is_safe, result, latency

    def _offline_check(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        start = time.perf_counter()
        combined = f"{user_input}\n{agent_response}"
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(combined):
                return False, f"unsafe: matched {pattern.pattern}", (time.perf_counter() - start) * 1000
        return True, "safe: offline policy classifier", (time.perf_counter() - start) * 1000

    def check(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        if self.use_api:
            try:
                return self._api_check(user_input, agent_response)
            except Exception as exc:
                safe, reason, latency = self._offline_check(user_input, agent_response)
                return safe, f"{reason}; api_fallback={exc}", latency
        return self._offline_check(user_input, agent_response)

    async def check_async(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        return await asyncio.to_thread(self.check, user_input, agent_response)

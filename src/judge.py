"""Judge adapter: lets DeepEval use a Groq (or any OpenAI-compatible) model.

Setup (PowerShell):
    $env:LLM_API_KEY = "your-groq-key"
    $env:JUDGE_MODEL = "qwen/qwen3.8-27b"     # must differ from the answering model
    $env:DEEPEVAL_TELEMETRY_OPT_OUT = "YES"
"""
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx

import env_loader  # noqa: F401  (loads .env)

try:                                   # import path differs between versions
    from deepeval.models import DeepEvalBaseLLM
except ImportError:
    from deepeval.models.base_model import DeepEvalBaseLLM

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MIN_INTERVAL = float(os.environ.get("JUDGE_MIN_INTERVAL", "2.5"))  # seconds between calls
CACHE_FILE = Path("output/judge_cache.jsonl")


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model=None):
        self.model_name = model or os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")
        self.last_call = 0.0
        self.cache = {}
        if CACHE_FILE.exists():            # reuse earlier answers, saves free quota
            for line in open(CACHE_FILE, encoding="utf-8"):
                row = json.loads(line)
                self.cache[row["key"]] = row["text"]

    # --- required by DeepEval -------------------------------------------
    def load_model(self):
        return self.model_name

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt, schema=None):
        text = self._ask(prompt, schema)
        if schema is None:
            return text
        data = json.loads(self._extract_json(text))
        return schema(**data)               # DeepEval wants the schema object back

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    # --- helpers ---------------------------------------------------------
    def _ask(self, prompt, schema):
        if schema is not None:
            prompt += ("\n\nReply with valid JSON only, no other text. "
                       f"JSON schema: {json.dumps(schema.model_json_schema())}")
        key = hashlib.sha256((self.model_name + prompt).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]

        body = {"model": self.model_name, "temperature": 0, "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {API_KEY}"}
        for attempt in range(6):
            wait = MIN_INTERVAL - (time.time() - self.last_call)
            if wait > 0:
                time.sleep(wait)             # stay under the free rate limit
            self.last_call = time.time()
            r = httpx.post(f"{BASE_URL}/chat/completions", json=body,
                           headers=headers, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(6 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Judge API error {r.status_code}: {r.text}")
            break
        else:
            raise RuntimeError("Judge API kept failing (rate limit?)")

        text = r.json()["choices"][0]["message"]["content"] or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        self.cache[key] = text
        CACHE_FILE.parent.mkdir(exist_ok=True)
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "text": text}) + "\n")
        return text

    @staticmethod
    def _extract_json(text):
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        start = min([i for i in (text.find("{"), text.find("[")) if i != -1],
                    default=0)
        end = max(text.rfind("}"), text.rfind("]")) + 1
        return text[start:end] if end > start else text
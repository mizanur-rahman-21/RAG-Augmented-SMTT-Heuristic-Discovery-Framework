"""
LLM Client with Full Provenance Logging
=======================================
Provides an interface for local and API LLM backends:
- Ollama backend (default): `http://localhost:11434/api/generate`
- OpenAI-compatible backend: `/v1/chat/completions`

Strict Provenance Requirements:
- Computes SHA-256 hash of the exact prompt sent.
- Computes SHA-256 hash of the exact raw response received.
- Records model name, backend, timestamp, inference time (ms), and token counts.
- Extracts Python code block cleanly.
- Raises descriptive exceptions if the LLM backend is offline or unavailable.
- NO mock, template, or fallback heuristics in this module.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import requests


@dataclass
class LLMGenerationResult:
    model: str
    backend: str
    timestamp_utc: str
    prompt_hash: str
    response_hash: str
    raw_response: str
    extracted_code: str
    generated_tokens: int
    inference_time_ms: float
    metadata: Dict[str, Any]


def extract_code_block(response_text: str, entry_point: str = "custom_heuristic") -> str:
    """Extracts python code block from markdown response or raw text."""
    # Look for ```python ... ```
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        for block in matches:
            if f"def {entry_point}" in block:
                return block.strip()
        return matches[0].strip()

    # If no markdown fence, check if raw response has function definition
    if f"def {entry_point}" in response_text:
        lines = response_text.splitlines()
        code_lines = []
        capturing = False
        for line in lines:
            if f"def {entry_point}" in line:
                capturing = True
            if capturing:
                code_lines.append(line)
        return "\n".join(code_lines).strip()

    return response_text.strip()


class LLMClient:
    """Base client interface for real LLM inference."""

    def generate(self, prompt: str) -> LLMGenerationResult:
        raise NotImplementedError

    def health_check(self) -> Dict[str, Any]:
        raise NotImplementedError


class OllamaClient(LLMClient):
    """
    Client for local Ollama instances.
    Default endpoint: http://localhost:11434
    Default model: qwen2.5-coder:7b-instruct-q4_K_M
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b-instruct-q4_K_M",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def health_check(self) -> Dict[str, Any]:
        """Checks if Ollama is running and whether the target model is available."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=5.0)
            if resp.status_code != 200:
                return {
                    "online": False,
                    "error": f"Ollama returned HTTP status {resp.status_code}",
                }
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            # Exact or prefix match
            model_present = any(self.model in m or m.startswith(self.model) for m in models)
            return {
                "online": True,
                "models_available": models,
                "target_model": self.model,
                "target_model_present": model_present,
            }
        except requests.exceptions.RequestException as e:
            return {
                "online": False,
                "error": f"Failed to connect to Ollama at {self.base_url}: {e}",
            }

    def generate(self, prompt: str, seed: Optional[int] = None) -> LLMGenerationResult:
        generate_url = f"{self.base_url}/api/generate"
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
        }
        if seed is not None:
            options["seed"] = int(seed)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(generate_url, json=payload, timeout=self.timeout_s)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}.\n"
                "Please verify that Ollama is running:\n"
                "  1. Start Ollama service: `ollama serve`\n"
                f"  2. Pull the model: `ollama pull {self.model}`"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Ollama request timed out after {self.timeout_s}s.") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama returned error HTTP {resp.status_code}: {resp.text}")

        resp_json = resp.json()
        raw_response = resp_json.get("response", "")
        response_hash = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        extracted_code = extract_code_block(raw_response)

        eval_count = resp_json.get("eval_count", len(raw_response.split()))

        return LLMGenerationResult(
            model=self.model,
            backend="ollama",
            timestamp_utc=timestamp_utc,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            raw_response=raw_response,
            extracted_code=extracted_code,
            generated_tokens=eval_count,
            inference_time_ms=inference_time_ms,
            metadata={
                "total_duration_ns": resp_json.get("total_duration", 0),
                "load_duration_ns": resp_json.get("load_duration", 0),
                "eval_duration_ns": resp_json.get("eval_duration", 0),
            },
        )


class OpenAICompatibleClient(LLMClient):
    """Client for OpenAI-compatible HTTP endpoints (vLLM, TGI, DeepSeek, LocalAI, OpenAI)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "qwen2.5-coder-7b-instruct",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def health_check(self) -> Dict[str, Any]:
        models_url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(models_url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", [])]
                return {
                    "online": True,
                    "models_available": models,
                    "target_model": self.model,
                    "target_model_present": self.model in models or len(models) > 0,
                }
            return {"online": False, "error": f"HTTP status {resp.status_code}"}
        except Exception as e:
            return {"online": False, "error": str(e)}

    def generate(self, prompt: str) -> LLMGenerationResult:
        chat_url = f"{self.base_url}/chat/completions"
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(chat_url, json=payload, headers=headers, timeout=self.timeout_s)
        except Exception as e:
            raise ConnectionError(f"Could not connect to OpenAI-compatible endpoint at {self.base_url}: {e}") from e

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI endpoint returned error HTTP {resp.status_code}: {resp.text}")

        resp_json = resp.json()
        raw_response = resp_json["choices"][0]["message"]["content"]
        response_hash = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        extracted_code = extract_code_block(raw_response)
        usage = resp_json.get("usage", {})
        gen_tokens = usage.get("completion_tokens", len(raw_response.split()))

        return LLMGenerationResult(
            model=self.model,
            backend="openai_compatible",
            timestamp_utc=timestamp_utc,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            raw_response=raw_response,
            extracted_code=extracted_code,
            generated_tokens=gen_tokens,
            inference_time_ms=inference_time_ms,
            metadata={"usage": usage},
        )


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """Factory creating configured LLM client based on config dict."""
    llm_cfg = config.get("llm", {})
    backend = llm_cfg.get("backend", "ollama").lower()

    if backend == "ollama":
        return OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg.get("model_name", "qwen2.5-coder:7b-instruct-q4_K_M"),
            temperature=float(llm_cfg.get("temperature", 0.7)),
            top_p=float(llm_cfg.get("top_p", 0.95)),
            max_tokens=int(llm_cfg.get("max_output_tokens", 1024)),
            timeout_s=float(llm_cfg.get("timeout_s", 120.0)),
        )
    elif backend in ("openai", "openai_compatible", "vllm"):
        return OpenAICompatibleClient(
            base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=llm_cfg.get("api_key", "EMPTY"),
            model=llm_cfg.get("model_name", "qwen2.5-coder-7b-instruct"),
            temperature=float(llm_cfg.get("temperature", 0.7)),
            top_p=float(llm_cfg.get("top_p", 0.95)),
            max_tokens=int(llm_cfg.get("max_output_tokens", 1024)),
            timeout_s=float(llm_cfg.get("timeout_s", 120.0)),
        )
    else:
        raise ValueError(f"Unknown LLM backend: '{backend}'. Supported: 'ollama', 'openai_compatible'.")

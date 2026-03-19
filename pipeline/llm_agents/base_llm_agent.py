"""
pipeline/llm_agents/base_llm_agent.py
--------------------------------------
Shared GateDecision dataclass and BaseLLMAgent with lazy anthropic init
and exponential backoff.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GateDecision:
    approved: bool
    confidence: str                          # "HIGH" | "MEDIUM" | "LOW"
    reasoning: str                           # 1-3 sentences
    action: str                              # "PROCEED" | "PROCEED_WITH_WARNING" | "HALT"
    warning: Optional[str] = None
    issues: Optional[list] = None            # gate2 extra field
    suspicious_patterns: Optional[list] = None  # gate3 extra field


class BaseLLMAgent:
    def __init__(self, model: str = "claude-opus-4-5", max_tokens: int = 2048,
                 max_retries: int = 3):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = None  # NOT initialized here

    def _get_client(self):
        """Lazy init — import anthropic only on first call."""
        if self._client is None:
            import anthropic                   # deferred: keeps pipeline importable without dep
            from dotenv import load_dotenv
            load_dotenv()                      # loads .env at CWD/repo root
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def call(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM with exponential backoff. Returns raw text response."""
        client = self._get_client()
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text
            except Exception as exc:
                last_exc = exc
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_exc}"
        )

    def call_structured(self, system_prompt: str, user_message: str,
                        output_schema: dict) -> dict:
        """Call LLM expecting JSON output matching output_schema.

        Appends JSON schema instruction to system prompt.
        Retries once with a correction prompt on parse failure.
        """
        schema_instruction = (
            "\n\nYou MUST respond with ONLY valid JSON matching this schema:\n"
            + json.dumps(output_schema, indent=2)
            + "\n\nDo not include any text outside the JSON object."
        )
        augmented_system = system_prompt + schema_instruction

        raw = self.call(augmented_system, user_message)

        # First parse attempt
        try:
            return _parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            pass

        # One retry with a correction prompt
        correction_msg = (
            f"Your previous response could not be parsed as JSON.\n"
            f"Response was:\n{raw}\n\n"
            f"Please respond with ONLY valid JSON matching the schema. "
            f"No markdown, no explanation."
        )
        raw2 = self.call(augmented_system, correction_msg)
        return _parse_json(raw2)


def _parse_json(text: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence lines
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        text = inner.strip()
    return json.loads(text)

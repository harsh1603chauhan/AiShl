from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.core.config import settings


class LLMClient(Protocol):
    def generate(self, prompt: str, system_prompt: str | None = None) -> str: ...


@dataclass(slots=True)
class NullLLMClient:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return ""


@dataclass(slots=True)
class OpenAIChatClient:
    model: str
    api_key: str

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=settings.llm_temperature)
        return response.choices[0].message.content or ""


@dataclass(slots=True)
class GeminiChatClient:
    model: str
    api_key: str

    def __post_init__(self) -> None:
        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        config = None
        if system_prompt:
            from google.genai import types

            config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=settings.llm_temperature)
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return getattr(response, "text", "") or ""


@dataclass(slots=True)
class GroqChatClient:
    model: str
    api_key: str

    def __post_init__(self) -> None:
        from groq import Groq

        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=settings.llm_temperature)
        return response.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "openai" and settings.openai_api_key and settings.llm_model:
        return OpenAIChatClient(model=settings.llm_model, api_key=settings.openai_api_key)
    if settings.llm_provider == "gemini" and settings.gemini_api_key and settings.llm_model:
        return GeminiChatClient(model=settings.llm_model, api_key=settings.gemini_api_key)
    if settings.llm_provider == "groq" and settings.groq_api_key and settings.llm_model:
        return GroqChatClient(model=settings.llm_model, api_key=settings.groq_api_key)
    return NullLLMClient()

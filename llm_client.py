"""Minimal OpenAI Chat Completions / Gemini client."""

import os
from collections.abc import Iterator, Sequence
from typing import Literal, TypedDict

from google import genai
from google.genai import types
from openai import OpenAI


class Message(TypedDict):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class LLMClient:
    def __init__(
        self,
        provider: Literal["openai", "gemini"],
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60,
    ) -> None:
        if provider not in ("openai", "gemini"):
            raise ValueError("provider must be 'openai' or 'gemini'")

        env_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        api_key = api_key or os.getenv(env_name)
        if not api_key:
            raise ValueError(f"missing API key: pass api_key or set {env_name}")

        self.model = model
        self._openai: OpenAI | None = None
        self._gemini: genai.Client | None = None

        if provider == "openai":
            self._openai = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            http_options = types.HttpOptions(timeout=int(timeout * 1000))
            if base_url:
                http_options.base_url = base_url
            self._gemini = genai.Client(api_key=api_key, http_options=http_options)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        **params: object,
    ) -> str:
        """Return the complete response text."""
        if self._openai:
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            response = self._openai.chat.completions.create(
                model=self.model,
                messages=self._openai_messages(messages, json_mode),
                **params,  # type: ignore[arg-type]
            )
            return response.choices[0].message.content or ""

        response = self._gemini.models.generate_content(  # type: ignore[union-attr]
            model=self.model,
            contents=self._gemini_contents(messages),
            config=self._gemini_config(messages, json_mode, params),
        )
        return response.text or ""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        **params: object,
    ) -> Iterator[str]:
        """Yield response text chunks."""
        if self._openai:
            params["stream"] = True
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            chunks = self._openai.chat.completions.create(
                model=self.model,
                messages=self._openai_messages(messages, json_mode),
                **params,  # type: ignore[arg-type]
            )
            for chunk in chunks:  # type: ignore[union-attr]
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    yield text
            return

        chunks = self._gemini.models.generate_content_stream(  # type: ignore[union-attr]
            model=self.model,
            contents=self._gemini_contents(messages),
            config=self._gemini_config(messages, json_mode, params),
        )
        for chunk in chunks:
            if chunk.text:
                yield chunk.text

    def close(self) -> None:
        client = self._openai or self._gemini
        if client:
            client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _openai_messages(
        messages: Sequence[Message], json_mode: bool
    ) -> list[Message]:
        result = list(messages)
        if json_mode:
            result.insert(0, {"role": "system", "content": "Return a valid JSON object."})
        return result

    @staticmethod
    def _gemini_contents(messages: Sequence[Message]) -> list[dict[str, object]]:
        return [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in messages
            if message["role"] not in ("system", "developer")
        ]

    @staticmethod
    def _gemini_config(
        messages: Sequence[Message], json_mode: bool, params: dict[str, object]
    ) -> types.GenerateContentConfig | None:
        config = dict(params)
        system = [
            message["content"]
            for message in messages
            if message["role"] in ("system", "developer")
        ]
        if system:
            config["system_instruction"] = "\n".join(system)
        if json_mode:
            config["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**config) if config else None

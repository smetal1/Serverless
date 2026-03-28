"""REST client for the Podstack gateway.

Provides a high-level Python interface to the Podstack Inference OS,
compatible with the OpenAI API format::

    from podstack import PodstackClient

    client = PodstackClient(
        base_url="https://api.podstack.example.com",
        api_key="ps-xxxxxxxxxxxx",
    )

    # Chat completion
    response = client.chat(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)

    # Streaming
    for chunk in client.chat(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True,
    ):
        print(chunk.choices[0].delta.content, end="")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import httpx

from .types import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelListResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
)

logger = logging.getLogger(__name__)


class PodstackClient:
    """Synchronous REST client for the Podstack Inference OS gateway.

    Parameters
    ----------
    base_url : str
        Base URL of the Podstack gateway (e.g. ``https://api.podstack.example.com``).
        Falls back to the ``PODSTACK_BASE_URL`` environment variable.
    api_key : str
        API key for authentication.  Falls back to ``PODSTACK_API_KEY``.
    timeout : float
        Default request timeout in seconds.
    max_retries : int
        Number of retries for transient failures.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (base_url or os.environ.get("PODSTACK_BASE_URL", "http://localhost:4000")).rstrip("/")
        self.api_key = api_key or os.environ.get("PODSTACK_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

        transport = httpx.HTTPTransport(retries=max_retries)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers=self._default_headers(),
        )

    def _default_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "podstack-sdk/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> PodstackClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Chat Completion
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse | Iterator[ChatCompletionChunk]:
        """Create a chat completion.

        If ``stream=True``, returns an iterator of ``ChatCompletionChunk``
        objects.  Otherwise returns a ``ChatCompletionResponse``.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        payload.update(kwargs)

        if stream:
            return self._stream_chat(payload)

        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return ChatCompletionResponse.model_validate(resp.json())

    def _stream_chat(self, payload: dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        """Stream chat completion via SSE."""
        with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data)
                        yield ChatCompletionChunk.model_validate(chunk_data)
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.warning("Failed to parse SSE chunk: %s", exc)

    # ------------------------------------------------------------------
    # Text Completion
    # ------------------------------------------------------------------

    def complete(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Create a text completion."""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(kwargs)

        resp = self._client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        return CompletionResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create embeddings for the given input."""
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        payload.update(kwargs)

        resp = self._client.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        return EmbeddingResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Image Generation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        model: str,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        payload.update(kwargs)

        resp = self._client.post("/v1/images/generations", json=payload)
        resp.raise_for_status()
        return ImageGenerationResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Audio - Transcription (ASR)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        model: str,
        audio_file: str | Path,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        **kwargs: Any,
    ) -> TranscriptionResponse:
        """Transcribe an audio file to text."""
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Multipart form data
        files = {
            "file": (audio_path.name, open(audio_path, "rb"), "audio/mpeg"),
        }
        data: dict[str, str] = {
            "model": model,
            "response_format": response_format,
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        data.update({k: str(v) for k, v in kwargs.items()})

        # Use a separate request without JSON content-type
        resp = self._client.post(
            "/v1/audio/transcriptions",
            files=files,
            data=data,
        )
        resp.raise_for_status()
        return TranscriptionResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Audio - Speech (TTS)
    # ------------------------------------------------------------------

    def synthesize(
        self,
        model: str,
        text: str,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
        output_file: str | Path | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Synthesize speech from text.

        Returns the raw audio bytes.  If *output_file* is specified,
        also writes the audio to disk.
        """
        payload: dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        payload.update(kwargs)

        resp = self._client.post("/v1/audio/speech", json=payload)
        resp.raise_for_status()

        audio_bytes = resp.content
        if output_file:
            Path(output_file).write_bytes(audio_bytes)
            logger.info("Audio written to %s", output_file)

        return audio_bytes

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def list_models(self) -> ModelListResponse:
        """List all available models."""
        resp = self._client.get("/v1/models")
        resp.raise_for_status()
        return ModelListResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Check gateway health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

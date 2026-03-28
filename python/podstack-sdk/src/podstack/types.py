"""Pydantic models for request/response types matching the OpenAI API.

These types are used by the ``PodstackClient`` and can also serve as
validation schemas in proxy/gateway code.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ======================================================================
# Chat Completion
# ======================================================================


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool", "function"] = "user"
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """Request body for POST /v1/chat/completions."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response body for POST /v1/chat/completions."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta = Field(default_factory=ChatCompletionChunkDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChunkChoice] = Field(default_factory=list)


# ======================================================================
# Text Completion
# ======================================================================


class CompletionRequest(BaseModel):
    """Request body for POST /v1/completions."""

    model: str
    prompt: str | list[str]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class CompletionChoice(BaseModel):
    """A single choice in a completion response."""

    index: int = 0
    text: str = ""
    finish_reason: str | None = "stop"
    logprobs: Any | None = None


class CompletionResponse(BaseModel):
    """Response body for POST /v1/completions."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


# ======================================================================
# Embeddings
# ======================================================================


class EmbeddingRequest(BaseModel):
    """Request body for POST /v1/embeddings."""

    model: str
    input: str | list[str]
    encoding_format: str = "float"
    user: str | None = None


class EmbeddingData(BaseModel):
    """A single embedding vector."""

    object: str = "embedding"
    index: int = 0
    embedding: list[float] = Field(default_factory=list)


class EmbeddingResponse(BaseModel):
    """Response body for POST /v1/embeddings."""

    object: str = "list"
    model: str = ""
    data: list[EmbeddingData] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


# ======================================================================
# Image Generation
# ======================================================================


class ImageGenerationRequest(BaseModel):
    """Request body for POST /v1/images/generations."""

    model: str = ""
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    response_format: str = "url"
    style: str = "vivid"
    user: str | None = None


class ImageData(BaseModel):
    """A single generated image."""

    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    """Response body for POST /v1/images/generations."""

    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData] = Field(default_factory=list)


# ======================================================================
# Audio - Transcription (ASR)
# ======================================================================


class TranscriptionRequest(BaseModel):
    """Request metadata for POST /v1/audio/transcriptions.

    The actual audio file is sent as multipart form data.
    """

    model: str = "whisper-1"
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float = 0.0


class TranscriptionResponse(BaseModel):
    """Response body for POST /v1/audio/transcriptions."""

    text: str = ""
    task: str = "transcribe"
    language: str | None = None
    duration: float | None = None
    segments: list[dict[str, Any]] | None = None


# ======================================================================
# Audio - Speech (TTS)
# ======================================================================


class SpeechRequest(BaseModel):
    """Request body for POST /v1/audio/speech."""

    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0


class SpeechResponse(BaseModel):
    """Metadata for a speech synthesis response.

    The actual audio bytes are returned as the HTTP response body.
    """

    content_type: str = "audio/mpeg"
    duration_seconds: float | None = None


# ======================================================================
# Models list
# ======================================================================


class ModelInfo(BaseModel):
    """A single model entry in the models list."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "podstack"
    permission: list[dict[str, Any]] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    """Response body for GET /v1/models."""

    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)

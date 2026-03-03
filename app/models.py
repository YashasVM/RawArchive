from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CanonicalMessage(BaseModel):
    conversation_id: str
    timestamp_ms: int
    sender_id: str
    sender_display: str
    text: str


class DatasetStats(BaseModel):
    messages_total: int
    target_user_messages: int = 0
    avg_reply_gap_sec: float | None = None
    conversations_total: int


class UploadResponse(BaseModel):
    dataset_id: str
    stats: DatasetStats
    warnings: list[str]
    participants: list[str]


class BuildRequest(BaseModel):
    target_user_id: str
    context_turns: int = Field(default=8, ge=1, le=30)
    min_reply_chars: int = Field(default=2, ge=1, le=200)
    max_samples: int = Field(default=20000, ge=100, le=100000)
    val_ratio: float = Field(default=0.1, gt=0.0, lt=0.5)


class BuildResponse(BaseModel):
    bundle_id: str
    train_examples: int
    val_examples: int
    download_url: str


class ColabLaunchResponse(BaseModel):
    notebook_url: str
    env: dict[str, str]


class RegisterModelRequest(BaseModel):
    bundle_id: str
    adapter_uri: str
    base_model: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class RegisterModelResponse(BaseModel):
    model_id: str
    status: str


class TrainingExample(BaseModel):
    conversation_id: str
    input_text: str
    target_text: str

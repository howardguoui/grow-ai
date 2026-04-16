"""Pydantic response schemas for the grow-ai API."""
from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel


class InsightOut(BaseModel):
    id: int
    compressed: str
    framework_tags: list[str]
    quality_score: int
    reinforcement_count: int
    error_recovery: bool
    created_at: str
    similarity: float | None = None


class InsightDetail(InsightOut):
    full_context: str
    lora_pair: dict | None = None


class InsightPage(BaseModel):
    items: list[InsightOut]
    total: int
    page: int
    pages: int


class SearchResult(BaseModel):
    items: list[InsightOut]
    query: str
    mode: str  # "semantic" | "keyword"
    count: int


class GrowthEntry(BaseModel):
    question_id: int
    question: str
    response: str
    recorded_at: str


class GrowthSnapshot(BaseModel):
    model_version: str | None
    insight_count: int
    entries: list[GrowthEntry]
    recorded_at: str


class BatchOut(BaseModel):
    id: int
    status: str
    insight_count: int
    triggered_at: str
    completed_at: str | None
    model_version: str | None


class StatsOut(BaseModel):
    total_insights: int
    avg_quality_score: float
    high_quality_count: int
    error_recovery_count: int
    reinforced_count: int
    top_frameworks: list[dict]
    pending_expansion: int
    queued_batches: int
    completed_batches: int
    last_batch_version: str | None


class HealthOut(BaseModel):
    ollama: bool
    db: bool
    last_model_version: str | None
    total_insights: int


class FinetuneOut(BaseModel):
    status: str
    message: str

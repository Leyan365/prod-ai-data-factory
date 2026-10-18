"""Pydantic schemas for the web API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    version: str
    default_provider: str
    supported_exports: List[str]
    storage_dir: str
    persisted_datasets: int
    persisted_jobs: int


class ExamplePreview(BaseModel):
    input_text: str
    output_text: str
    task_type: str
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    quality_approved: Optional[bool] = None


class DatasetSummary(BaseModel):
    id: str
    name: str
    description: str
    example_count: int
    created_at: str
    export_format: Optional[str] = None
    export_path: Optional[str] = None


class DatasetDetail(BaseModel):
    id: str
    name: str
    description: str
    example_count: int
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    examples: List[ExamplePreview] = Field(default_factory=list)


class JobSummary(BaseModel):
    id: str
    name: str
    status: str
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    error_message: Optional[str] = None
    dataset_id: Optional[str] = None
    created_at: str
    output_data: Dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str

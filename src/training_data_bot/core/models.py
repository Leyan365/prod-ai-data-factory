"""
Core data models for training data bot.

This module defines Pydantic models for all data structures used throughout
the application , ensuring type safety and validation.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

try:
    from pydantic import BaseModel, Field, field_validator
    try:
        from pydantic import ConfigDict
    except ImportError:
        ConfigDict = None
except ImportError:
    from enum import Enum as _Enum
    import json as _json

    class _FieldInfo:
        def __init__(self, default: Any = None, default_factory=None):
            self.default = default
            self.default_factory = default_factory

        def get_default(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            return self.default

    def Field(default: Any = None, default_factory=None, **kwargs):
        return _FieldInfo(default=default, default_factory=default_factory)

    def field_validator(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate

    class BaseModel:
        """Small fallback used when pydantic is not installed."""

        def __init__(self, **data: Any):
            fields: Dict[str, Any] = {}
            for cls in reversed(self.__class__.mro()):
                fields.update(getattr(cls, "__annotations__", {}))

            for name in fields:
                if name in data:
                    value = data.pop(name)
                elif hasattr(self.__class__, name):
                    default = getattr(self.__class__, name)
                    value = default.get_default() if isinstance(default, _FieldInfo) else default
                else:
                    value = None
                setattr(self, name, value)

            for name, value in data.items():
                setattr(self, name, value)

            self._apply_fallback_calculations()

        def _apply_fallback_calculations(self) -> None:
            if self.__class__.__name__ == "Document" and getattr(self, "content", None):
                if getattr(self, "word_count", 0) == 0:
                    self.word_count = len(self.content.split())
                if getattr(self, "char_count", 0) == 0:
                    self.char_count = len(self.content)
            elif self.__class__.__name__ == "TextChunk" and getattr(self, "content", None):
                if getattr(self, "token_count", 0) == 0:
                    self.token_count = len(self.content) // 4
            elif self.__class__.__name__ == "Dataset":
                if getattr(self, "total_examples", 0) == 0 and getattr(self, "examples", None) is not None:
                    self.total_examples = len(self.examples)

        def model_dump(self, mode: str = "python", **kwargs) -> Dict[str, Any]:
            data = {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            }
            if mode == "json":
                return self._jsonable(data)
            return data

        def model_dump_json(self, **kwargs) -> str:
            return _json.dumps(self.model_dump(mode="json"))

        @classmethod
        def model_validate(cls, data: Any):
            if isinstance(data, cls):
                return data
            if not isinstance(data, dict):
                raise TypeError("model_validate expects a dictionary")
            return cls(**data)

        def dict(self) -> Dict[str, Any]:
            return self.model_dump()

        def json(self) -> str:
            return self.model_dump_json()

        def _raw_dict(self) -> Dict[str, Any]:
            return {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            }

        @classmethod
        def _jsonable(cls, value: Any) -> Any:
            if isinstance(value, BaseModel):
                return cls._jsonable(value.model_dump())
            if isinstance(value, dict):
                return {cls._jsonable(k): cls._jsonable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cls._jsonable(item) for item in value]
            if isinstance(value, (UUID, datetime, Path)):
                return str(value)
            if isinstance(value, _Enum):
                return value.value
            return value

    ConfigDict = None


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class BaseEntity(BaseModel):
    """Base class for all entites with common fields."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    if ConfigDict is not None:
        model_config = ConfigDict(
            use_enum_values=True,
            populate_by_name=True,
            arbitrary_types_allowed=True,
        )
    else:
        class Config:
            use_enum_values = True
            allow_population_by_field_name = True
            arbitrary_types_allowed = True


# Enums
class DocumentType(str, Enum):
    """Supported Documents Types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    URL = "url"


class TaskType(str, Enum):
    QA_GENERATION = "qa_generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    NER = "named_entity_recognition"
    RED_TEAMING = "red_teaming"
    INSTRUCTION_RESPONSE = "instruction_response"


class QualityMetric(str, Enum):
    """Quality asseseement metrics"""
    TOXICITY = "toxicity"
    BIAS = "bias"
    DIVERSITY = "diversity"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"


class ProcessingStatus(str, Enum):
    """Processing status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    """Export format options"""
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


# Documents models
class Document(BaseEntity):
    """Represents a source document"""
    title: str
    content: str
    source: str # file path, URL, etc.
    doc_type: DocumentType
    language: Optional[str] = "en"
    encoding: Optional[str] = "utf-8"
    size: int = 0 # bytes
    word_count: int = Field(default=0, validate_default=True)
    char_count: int = Field(default=0, validate_default=True)

    # processing info
    extraction_method: Optional[str] = None
    processing_time: Optional[float] = None # Changed type to float for consistency

    @field_validator("word_count", mode="before")
    def calculate_word_count(cls, v, info):
        content = getattr(info, "data", {}).get("content")
        if content is not None and (v == 0 or v is None):
            return len(content.split())
        return v
    
    @field_validator("char_count", mode="before")
    def calculate_char_count(cls, v, info):
        content = getattr(info, "data", {}).get("content")
        if content is not None and (v == 0 or v is None):
            return len(content)
        return v
    

class TextChunk(BaseEntity):
    document_id: UUID
    content: str
    start_index: int
    end_index: int
    chunk_index: int
    token_count: int = Field(default=0, validate_default=True)

    # Context preservation
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None

    # Semantic info
    embeddings: Optional[List[float]] = None
    topics: List[str] = Field(default_factory=list) # Changed hint to List[str]

    @field_validator("token_count", mode="before")
    def calculate_token_count(cls, v, info):
        content = getattr(info, "data", {}).get("content")
        if content is not None and (v == 0 or v is None):
            return len(content) // 4
        return v
    

# Task Models
class TaskTemplate(BaseEntity):
    """Represents a task template"""
    name: str
    task_type: TaskType
    description: str
    prompt_template: str
    output_format: Optional[str] = None

    # Task_specific configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality requirements
    min_output_length: int = 10
    max_output_length: int = 2000
    quality_thresholds: Dict[QualityMetric, float] = Field(default_factory=dict)

    # Performance settings
    timeout: int = 60
    max_retries: int = 3


class TaskResult(BaseEntity):
    """Result of a task execution"""
    task_id: UUID
    # FIX: Corrected field definition with an assignment
    template_id: UUID 
    input_chunk_id: UUID

    # Output
    output: str
    confidence: Optional[float] = None

    # Quality scores
    # FIX: Corrected field definition with an assignment
    quality_scores: Dict[QualityMetric, float] = Field(default_factory=dict)

    # Processing info
    processing_time: float
    token_usage: int = 0
    cost: Optional[float] = None

    # Status
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    raw_output: Optional[Dict[str, Any]] = None


# Training Data Models
class TrainingExample(BaseEntity):
    """A single training example"""
    input_text: str
    output_text: str
    task_type: TaskType

    # Source tracking
    source_document_id: UUID
    source_chunk_id: Optional[UUID] = None
    template_id: Optional[UUID] = None

    # Quality assessment
    quality_scores: Dict[QualityMetric, float] = Field(default_factory=dict)
    quality_approved: Optional[bool] = None

    # Additional fields for different formats
    instruction: Optional[str] = None # For instruction-following datasets
    context: Optional[str] = None  # For context-based tasks
    category: Optional[str] = None # For classification tasks


class Dataset(BaseEntity):
    """A collection of training examples"""
    name: str
    description: str
    version: str = "1.0.0"

    # Content
    # FIX: Changed default_factory to list
    examples: List[TrainingExample] = Field(default_factory=list) 

    # Statistics
    total_examples: int = Field(default=0, validate_default=True)
    task_type_counts: Dict[TaskType, int] = Field(default_factory=dict)
    quality_stats: Dict[QualityMetric, Dict[str, float]] = Field(default_factory=dict)

    # Splits
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1

    # Export info
    export_format: ExportFormat = ExportFormat.JSONL
    exported_at: Optional[datetime] = None
    export_path: Optional[Path] = None

    @field_validator("total_examples", mode="before")
    def calculate_total_examples(cls, v, info):
        examples = getattr(info, "data", {}).get("examples")
        if examples is not None and (v == 0 or v is None):
            return len(examples)
        return v


# Quality Models
class QualityReport(BaseEntity):
    """Quality assessment result for an example or dataset."""

    target_id: UUID
    overall_score: float = 1.0
    passed: bool = True
    metric_scores: Dict[QualityMetric, float] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)


# Operation Models
class ProcessingJob(BaseEntity):
    """Tracks progress for a processing operation."""

    name: str
    job_type: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    started_at: datetime = Field(default_factory=utc_now)
    estimated_completion: Optional[datetime] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    raw_output: Optional[Dict[str, Any]] = None

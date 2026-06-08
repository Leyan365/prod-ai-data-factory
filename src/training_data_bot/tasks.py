"""
Task template rendering and execution.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

from .ai import AIClient
from .core.config import settings
from .core.exceptions import TaskExecutionError, TemplateRenderError
from .core.models import ProcessingStatus, TaskResult, TaskTemplate, TaskType, TextChunk


class StrictTemplateContext(dict):
    """Dictionary that raises a domain error for missing template variables."""

    def __missing__(self, key):
        raise TemplateRenderError(f"Missing prompt template variable: {key}")


class TemplateRenderer:
    """Render simple named placeholders without expression execution."""

    def render(self, template: TaskTemplate, context: Mapping[str, Any]) -> str:
        try:
            return template.prompt_template.format_map(StrictTemplateContext(context))
        except TemplateRenderError:
            raise
        except Exception as exc:
            raise TemplateRenderError(
                f"Failed to render prompt template '{template.name}'",
                cause=exc,
            ) from exc


@dataclass
class TaskGenerator:
    """Compatibility wrapper for generator-style imports."""

    task_type: TaskType


class TaskManager:
    """Resolve task templates, render prompts, and call an AI client."""

    def __init__(
        self,
        templates: Optional[Mapping[TaskType, TaskTemplate]] = None,
        renderer: Optional[TemplateRenderer] = None,
    ):
        self.templates: Dict[TaskType, TaskTemplate] = (
            self.default_templates() if templates is None else dict(templates)
        )
        self.renderer = renderer or TemplateRenderer()

    @staticmethod
    def default_templates() -> Dict[TaskType, TaskTemplate]:
        """Return built-in templates for every supported task type."""

        return {
            TaskType.QA_GENERATION: TaskTemplate(
                name="Basic QA Generation",
                task_type=TaskType.QA_GENERATION,
                description="Generate one question and answer from a chunk.",
                prompt_template=(
                    "Create one concise question and answer from this source text.\n"
                    "Source title: {source_document_title}\n"
                    "Text:\n{content}"
                ),
                output_format="question_answer",
                timeout=settings.ai_timeout,
                max_retries=settings.ai_max_retries,
            ),
            TaskType.CLASSIFICATION: TaskTemplate(
                name="Basic Classification",
                task_type=TaskType.CLASSIFICATION,
                description="Classify a chunk into a short label.",
                prompt_template=(
                    "Classify the following text into one short category label.\n"
                    "Text:\n{content}"
                ),
                output_format="label",
                timeout=settings.ai_timeout,
                max_retries=settings.ai_max_retries,
            ),
            TaskType.SUMMARIZATION: TaskTemplate(
                name="Basic Summarization",
                task_type=TaskType.SUMMARIZATION,
                description="Summarize a chunk.",
                prompt_template="Summarize the following text clearly and briefly.\nText:\n{content}",
                output_format="summary",
                timeout=settings.ai_timeout,
                max_retries=settings.ai_max_retries,
            ),
            TaskType.NER: TaskTemplate(
                name="Basic Entity Extraction",
                task_type=TaskType.NER,
                description="Extract named entities as a simple list.",
                prompt_template=(
                    "Extract named entities from the text as a simple JSON-like list.\n"
                    "Text:\n{content}"
                ),
                output_format="entity_list",
                timeout=settings.ai_timeout,
                max_retries=settings.ai_max_retries,
            ),
            TaskType.RED_TEAMING: TaskTemplate(
                name="Basic Red Teaming",
                task_type=TaskType.RED_TEAMING,
                description="Identify possible unsafe or problematic instructions.",
                prompt_template=(
                    "Identify possible unsafe or problematic instructions in this text.\n"
                    "Text:\n{content}"
                ),
                output_format="safety_notes",
                timeout=settings.ai_timeout,
                max_retries=settings.ai_max_retries,
            ),
            TaskType.INSTRUCTION_RESPONSE: TaskTemplate(
                name="Basic Instruction Response",
                task_type=TaskType.INSTRUCTION_RESPONSE,
                description="Create one instruction and response pair from a chunk.",
                prompt_template=(
                    "Create one instruction and response pair based on this text.\n"
                    "Document: {source_document_title}\n"
                    "Text:\n{content}"
                ),
                output_format="instruction_response",
                timeout=settings.ai_timeout,
                max_retries=settings.ai_max_retries,
            ),
        }

    async def execute_task(
        self,
        task_type: TaskType,
        input_chunk: TextChunk,
        client=None,
        *,
        raise_on_error: bool = True,
    ) -> TaskResult:
        """Execute one task type against one preprocessed chunk."""

        started = perf_counter()
        template = self.get_template(task_type)
        prompt = self.renderer.render(template, self.build_context(input_chunk))
        ai_client = client or AIClient(max_retries=template.max_retries, timeout=template.timeout)

        try:
            response = await ai_client.generate(prompt, timeout=template.timeout)
        except Exception as exc:
            if not raise_on_error:
                return TaskResult(
                    task_id=uuid4(),
                    template_id=template.id,
                    input_chunk_id=input_chunk.id,
                    output="",
                    processing_time=perf_counter() - started,
                    token_usage=0,
                    status=ProcessingStatus.FAILED,
                    error_message=str(exc),
                )
            raise TaskExecutionError(
                f"Failed to execute task '{template.name}'",
                cause=exc,
            ) from exc

        return TaskResult(
            task_id=uuid4(),
            template_id=template.id,
            input_chunk_id=input_chunk.id,
            output=response.text,
            confidence=None,
            processing_time=perf_counter() - started,
            token_usage=response.token_usage,
            cost=response.cost,
            status=ProcessingStatus.COMPLETED,
            raw_output=response.raw_response,
        )

    def get_template(self, task_type: TaskType) -> TaskTemplate:
        try:
            return self.templates[task_type]
        except KeyError as exc:
            raise TaskExecutionError(f"No task template registered for task type: {task_type}") from exc

    def build_context(self, input_chunk: TextChunk) -> Dict[str, Any]:
        metadata = input_chunk.metadata or {}
        return {
            "content": input_chunk.content,
            "chunk_index": input_chunk.chunk_index,
            "document_id": str(input_chunk.document_id),
            "source_document_title": metadata.get("source_document_title", ""),
            "source_document_type": metadata.get("source_document_type", ""),
            "source_document_source": metadata.get("source_document_source", ""),
        }


class QAGenerator(TaskGenerator):
    def __init__(self):
        super().__init__(TaskType.QA_GENERATION)


class ClassificationGenerator(TaskGenerator):
    def __init__(self):
        super().__init__(TaskType.CLASSIFICATION)


class SummarizationGenerator(TaskGenerator):
    def __init__(self):
        super().__init__(TaskType.SUMMARIZATION)

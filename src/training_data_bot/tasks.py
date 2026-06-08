"""
Deterministic task generation for the current tutorial slice.
"""

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from .core.models import ProcessingStatus, TaskResult, TaskTemplate, TaskType, TextChunk


@dataclass
class GeneratedTask:
    """Internal task metadata."""

    template_id: object
    name: str


class TaskManager:
    """Execute simple task templates without external AI calls."""

    def __init__(self):
        self.templates = {
            TaskType.QA_GENERATION: GeneratedTask(uuid4(), "Basic QA"),
            TaskType.CLASSIFICATION: GeneratedTask(uuid4(), "Basic Classification"),
            TaskType.SUMMARIZATION: GeneratedTask(uuid4(), "Basic Summary"),
            TaskType.NER: GeneratedTask(uuid4(), "Basic Entity Extraction"),
            TaskType.RED_TEAMING: GeneratedTask(uuid4(), "Basic Safety Prompt"),
            TaskType.INSTRUCTION_RESPONSE: GeneratedTask(uuid4(), "Basic Instruction Response"),
        }

    async def execute_task(self, task_type: TaskType, input_chunk: TextChunk, client=None) -> TaskResult:
        started = perf_counter()
        template = self.templates[task_type]
        output = self._generate_output(task_type, input_chunk.content)

        return TaskResult(
            task_id=uuid4(),
            template_id=template.template_id,
            input_chunk_id=input_chunk.id,
            output=output,
            confidence=1.0,
            processing_time=perf_counter() - started,
            token_usage=input_chunk.token_count,
            status=ProcessingStatus.COMPLETED,
        )

    def _generate_output(self, task_type: TaskType, content: str) -> str:
        preview = " ".join(content.split())[:500]

        if task_type == TaskType.QA_GENERATION:
            return f"Question: What is the main idea of this passage?\nAnswer: {preview}"
        if task_type == TaskType.CLASSIFICATION:
            return "general"
        if task_type == TaskType.SUMMARIZATION:
            return preview
        if task_type == TaskType.NER:
            return "[]"
        if task_type == TaskType.RED_TEAMING:
            return f"Review this passage for safety concerns: {preview}"
        if task_type == TaskType.INSTRUCTION_RESPONSE:
            return f"Instruction: Explain the passage.\nResponse: {preview}"
        return preview


class QAGenerator:
    pass


class ClassificationGenerator:
    pass


class SummarizationGenerator:
    pass

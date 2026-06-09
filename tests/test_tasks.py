"""Tests for task templates and execution."""

import asyncio
from uuid import uuid4

import pytest

from training_data_bot.ai import AIClient, AIResponse, MockAIProvider
from training_data_bot.bot import TrainingDataBot
from training_data_bot.core.exceptions import TaskExecutionError, TemplateRenderError
from training_data_bot.core.models import Document, DocumentType, ProcessingStatus, TaskTemplate, TaskType, TextChunk
from training_data_bot.tasks import TaskManager, TemplateRenderer


def run(coro):
    return asyncio.run(coro)


def make_chunk(content="Chunk content"):
    return TextChunk(
        id=uuid4(),
        document_id=uuid4(),
        content=content,
        start_index=0,
        end_index=len(content),
        chunk_index=2,
        metadata={
            "source_document_title": "Doc Title",
            "source_document_type": "txt",
            "source_document_source": "memory://doc",
        },
    )


def test_default_templates_cover_all_task_types():
    templates = TaskManager.default_templates()

    assert set(templates) == set(TaskType)
    assert all(isinstance(template, TaskTemplate) for template in templates.values())


def test_template_renderer_inserts_chunk_context():
    template = TaskTemplate(
        name="Test",
        task_type=TaskType.SUMMARIZATION,
        description="test",
        prompt_template="{source_document_title}: {chunk_index}: {content}",
    )
    context = TaskManager().build_context(make_chunk("hello"))

    rendered = TemplateRenderer().render(template, context)

    assert rendered == "Doc Title: 2: hello"


def test_template_renderer_missing_variable_raises():
    template = TaskTemplate(
        name="Bad",
        task_type=TaskType.SUMMARIZATION,
        description="bad",
        prompt_template="{missing}",
    )

    with pytest.raises(TemplateRenderError):
        TemplateRenderer().render(template, {})


def test_unknown_task_type_raises_task_execution_error():
    manager = TaskManager(templates={})

    with pytest.raises(TaskExecutionError):
        manager.get_template(TaskType.SUMMARIZATION)


def test_execute_task_returns_structured_task_result():
    manager = TaskManager()
    chunk = make_chunk("Useful content")
    result = run(manager.execute_task(TaskType.SUMMARIZATION, chunk, client=AIClient()))

    assert result.template_id == manager.templates[TaskType.SUMMARIZATION].id
    assert result.input_chunk_id == chunk.id
    assert result.status == ProcessingStatus.COMPLETED
    assert result.output.startswith("Mock response:")
    assert result.processing_time >= 0
    assert result.token_usage > 0
    assert result.raw_output == {"provider": "mock"}


def test_execute_task_without_client_uses_offline_mock_provider():
    result = run(TaskManager().execute_task(TaskType.CLASSIFICATION, make_chunk("content")))

    assert result.status == ProcessingStatus.COMPLETED
    assert result.output.startswith("Mock response:")


def test_execute_task_failure_can_return_failed_result():
    class FailingProvider:
        async def generate(self, prompt, *, timeout=None):
            raise RuntimeError("provider failed")

        async def close(self):
            return None

    result = run(
        TaskManager().execute_task(
            TaskType.SUMMARIZATION,
            make_chunk("content"),
            client=AIClient(provider=FailingProvider(), max_retries=0),
            raise_on_error=False,
        )
    )

    assert result.status == ProcessingStatus.FAILED
    assert result.output == ""
    assert "provider failed" in result.error_message


def test_execute_task_failure_raises_by_default():
    class FailingProvider:
        async def generate(self, prompt, *, timeout=None):
            raise RuntimeError("provider failed")

        async def close(self):
            return None

    with pytest.raises(TaskExecutionError):
        run(
            TaskManager().execute_task(
                TaskType.SUMMARIZATION,
                make_chunk("content"),
                client=AIClient(provider=FailingProvider(), max_retries=0),
            )
        )


def test_training_bot_process_documents_uses_mock_task_layer(workspace_tmp):
    bot = TrainingDataBot(config={"storage_dir": workspace_tmp / "storage"})
    document = Document(
        title="Bot Doc",
        content="This is source content for the bot workflow.",
        source="memory://bot",
        doc_type=DocumentType.TXT,
    )

    dataset = run(
        bot.process_documents(
            documents=[document],
            task_types=[TaskType.SUMMARIZATION],
            quality_filter=False,
        )
    )
    run(bot.cleanup())

    assert len(dataset.examples) == 1
    assert dataset.examples[0].output_text.startswith("Mock response:")
    assert dataset.examples[0].source_document_id == document.id

"""Tests for deterministic quality evaluation."""

import asyncio
from uuid import uuid4

from training_data_bot.ai import AIClient, AIResponse
from training_data_bot.bot import TrainingDataBot
from training_data_bot.core.models import (
    Dataset,
    Document,
    DocumentType,
    QualityMetric,
    TaskType,
    TrainingExample,
)
from training_data_bot.evaluation import QualityEvaluator


def run(coro):
    return asyncio.run(coro)


DEFAULT_INPUT = "The solar system includes planets orbiting the sun."
DEFAULT_OUTPUT = "The answer explains that planets orbit the sun in the solar system."


def make_example(input_text=None, output_text=None):
    return TrainingExample(
        input_text=DEFAULT_INPUT if input_text is None else input_text,
        output_text=DEFAULT_OUTPUT if output_text is None else output_text,
        task_type=TaskType.SUMMARIZATION,
        source_document_id=uuid4(),
    )


def score(report, metric):
    return report.metric_scores.get(metric, report.metric_scores.get(metric.value))


def test_passing_example_has_populated_metric_scores():
    report = run(QualityEvaluator().evaluate(make_example()))

    assert report.passed is True
    assert report.overall_score >= 0.6
    assert set(report.metric_scores) == {metric.value for metric in QualityMetric}
    assert score(report, QualityMetric.RELEVANCE) > 0
    assert score(report, QualityMetric.COHERENCE) == 1.0
    assert report.issues == []


def test_empty_input_and_output_fail_with_actionable_reasons():
    report = run(QualityEvaluator().evaluate(make_example(input_text="", output_text="")))

    assert report.passed is False
    assert "Input text is too short" in report.issues
    assert "Output text is too short" in report.issues
    assert report.reasons
    assert score(report, QualityMetric.RELEVANCE) == 0.0
    assert score(report, QualityMetric.COHERENCE) == 0.0


def test_short_or_malformed_output_fails_coherence():
    short_report = run(QualityEvaluator().evaluate(make_example(output_text="ok")))
    malformed_report = run(
        QualityEvaluator().evaluate(
            make_example(output_text="Solar system planets orbit sun !!!!!!!!!!")
        )
    )

    assert short_report.passed is False
    assert "Output text is too short" in short_report.issues
    assert malformed_report.passed is False
    assert "Output appears incoherent or malformed" in malformed_report.issues


def test_blocked_terms_fail_toxicity_and_bias():
    report = run(
        QualityEvaluator(blocked_terms=("forbidden",)).evaluate(
            make_example(output_text="The solar system answer includes a forbidden term.")
        )
    )

    assert report.passed is False
    assert "Output contains blocked terms: forbidden" in report.issues
    assert score(report, QualityMetric.TOXICITY) == 0.0
    assert score(report, QualityMetric.BIAS) == 0.0


def test_unrelated_output_fails_relevance():
    report = run(
        QualityEvaluator(min_relevance_overlap=0.2).evaluate(
            make_example(
                input_text="Photosynthesis uses sunlight to make plant energy.",
                output_text="Airplanes require runways, pilots, engines, and careful navigation.",
            )
        )
    )

    assert report.passed is False
    assert "Output appears weakly related to input" in report.issues


def test_dataset_duplicate_outputs_reduce_diversity_and_warn():
    duplicate_output = "Planets orbit the sun in the solar system."
    dataset = Dataset(
        name="duplicates",
        description="duplicate outputs",
        examples=[
            make_example(output_text=duplicate_output),
            make_example(output_text=duplicate_output),
            make_example(output_text="The moon orbits Earth and reflects sunlight."),
        ],
    )

    report = run(QualityEvaluator().evaluate_dataset(dataset))

    assert score(report, QualityMetric.DIVERSITY) < 1.0
    assert any("near-duplicate" in warning for warning in report.warnings)


def test_empty_dataset_fails_with_zero_metric_scores():
    report = run(QualityEvaluator().evaluate_dataset(Dataset(name="empty", description="empty")))

    assert report.passed is False
    assert report.overall_score == 0.0
    assert "Dataset has no examples" in report.issues
    assert all(value == 0.0 for value in report.metric_scores.values())


def test_threshold_behavior_can_be_configured():
    strict = QualityEvaluator(
        min_overall_score=0.95,
        min_metric_score=0.95,
        min_relevance_overlap=0.5,
    )

    report = run(
        strict.evaluate(
            make_example(
                input_text="Cats sleep on warm windowsills.",
                output_text="Cats rest near quiet furniture and cushions.",
            )
        )
    )

    assert report.passed is False
    assert any("below threshold" in issue for issue in report.issues)


class StaticProvider:
    def __init__(self, text):
        self.text = text

    async def generate(self, prompt, *, timeout=None):
        return AIResponse(text=self.text, token_usage=1)

    async def close(self):
        return None


def test_training_bot_quality_filter_filters_failed_and_approves_passing_examples(workspace_tmp):
    document = Document(
        title="Quality Doc",
        content="The solar system has planets orbiting the sun.",
        source="memory://quality",
        doc_type=DocumentType.TXT,
    )

    failing_bot = TrainingDataBot(config={"storage_dir": workspace_tmp / "failing-storage"})
    failing_bot.ai_client = AIClient(provider=StaticProvider("x"), max_retries=0)
    failing_dataset = run(
        failing_bot.process_documents(
            documents=[document],
            task_types=[TaskType.SUMMARIZATION],
            quality_filter=True,
        )
    )
    run(failing_bot.cleanup())

    passing_bot = TrainingDataBot(config={"storage_dir": workspace_tmp / "passing-storage"})
    passing_bot.ai_client = AIClient(
        provider=StaticProvider("The solar system has planets orbiting the sun."),
        max_retries=0,
    )
    passing_dataset = run(
        passing_bot.process_documents(
            documents=[document],
            task_types=[TaskType.SUMMARIZATION],
            quality_filter=True,
        )
    )
    run(passing_bot.cleanup())

    assert failing_dataset.examples == []
    assert len(passing_dataset.examples) == 1
    assert passing_dataset.examples[0].quality_approved is True

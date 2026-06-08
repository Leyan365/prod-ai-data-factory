"""
Lightweight quality evaluation for generated examples and datasets.
"""

from .core.models import Dataset, QualityReport, TrainingExample


class QualityEvaluator:
    """Evaluate examples with simple length-based checks."""

    async def evaluate(self, example: TrainingExample) -> QualityReport:
        issues = []
        if not example.input_text.strip():
            issues.append("Input text is empty")
        if len(example.output_text.strip()) < 3:
            issues.append("Output text is too short")

        passed = not issues
        return QualityReport(
            target_id=example.id,
            overall_score=1.0 if passed else 0.0,
            passed=passed,
            issues=issues,
            reasons=issues,
        )

    async def evaluate_dataset(self, dataset: Dataset, detailed: bool = True) -> QualityReport:
        total = len(dataset.examples)
        approved = len([ex for ex in dataset.examples if ex.quality_approved is not False])
        score = approved / total if total else 0.0
        issues = [] if total else ["Dataset has no examples"]

        return QualityReport(
            target_id=dataset.id,
            overall_score=score,
            passed=score > 0,
            issues=issues,
            reasons=issues,
        )

"""
Rule-based quality evaluation for generated training examples.
"""

import re
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .core.config import settings
from .core.models import Dataset, QualityMetric, QualityReport, TrainingExample


class QualityEvaluator:
    """Evaluate examples and datasets with deterministic offline checks."""

    def __init__(
        self,
        *,
        min_input_length: int = settings.quality_min_input_length,
        min_output_length: int = settings.quality_min_output_length,
        min_overall_score: float = settings.quality_min_overall_score,
        min_metric_score: float = settings.quality_min_metric_score,
        min_relevance_overlap: float = settings.quality_min_relevance_overlap,
        duplicate_threshold: float = settings.quality_duplicate_threshold,
        blocked_terms: Sequence[str] = settings.quality_blocked_terms,
    ):
        self.min_input_length = min_input_length
        self.min_output_length = min_output_length
        self.min_overall_score = min_overall_score
        self.min_metric_score = min_metric_score
        self.min_relevance_overlap = min_relevance_overlap
        self.duplicate_threshold = duplicate_threshold
        self.blocked_terms = tuple(term.lower() for term in blocked_terms)

    async def evaluate(self, example: TrainingExample) -> QualityReport:
        """Evaluate a single training example."""

        input_text = example.input_text or ""
        output_text = example.output_text or ""
        issues: List[str] = []
        warnings: List[str] = []

        metric_scores = {
            QualityMetric.RELEVANCE: self._score_relevance(input_text, output_text),
            QualityMetric.COHERENCE: self._score_coherence(output_text),
            QualityMetric.TOXICITY: self._score_blocked_terms(output_text),
            QualityMetric.BIAS: self._score_blocked_terms(output_text),
            QualityMetric.DIVERSITY: 1.0,
        }

        if len(input_text.strip()) < self.min_input_length:
            issues.append("Input text is too short")
            metric_scores[QualityMetric.RELEVANCE] = 0.0
        if len(output_text.strip()) < self.min_output_length:
            issues.append("Output text is too short")
            metric_scores[QualityMetric.COHERENCE] = min(metric_scores[QualityMetric.COHERENCE], 0.0)

        blocked = self._find_blocked_terms(output_text)
        if blocked:
            joined = ", ".join(blocked)
            issues.append(f"Output contains blocked terms: {joined}")
            metric_scores[QualityMetric.TOXICITY] = 0.0
            metric_scores[QualityMetric.BIAS] = 0.0

        if metric_scores[QualityMetric.RELEVANCE] < self.min_metric_score:
            issues.append("Output appears weakly related to input")
        if metric_scores[QualityMetric.COHERENCE] < self.min_metric_score:
            issues.append("Output appears incoherent or malformed")

        overall_score = self._overall_score(metric_scores.values())
        if overall_score < self.min_overall_score:
            issues.append(
                f"Overall quality score {overall_score:.2f} is below threshold {self.min_overall_score:.2f}"
            )

        passed = not issues
        return QualityReport(
            target_id=example.id,
            overall_score=overall_score,
            passed=passed,
            metric_scores=metric_scores,
            issues=self._deduplicate(issues),
            warnings=warnings,
            reasons=self._deduplicate(issues + warnings),
        )

    async def evaluate_dataset(self, dataset: Dataset, detailed: bool = True) -> QualityReport:
        """Evaluate aggregate quality signals for a dataset."""

        if not dataset.examples:
            issue = "Dataset has no examples"
            return QualityReport(
                target_id=dataset.id,
                overall_score=0.0,
                passed=False,
                metric_scores={metric: 0.0 for metric in QualityMetric},
                issues=[issue],
                reasons=[issue],
            )

        example_reports = [await self.evaluate(example) for example in dataset.examples]
        metric_scores = self._average_metric_scores(example_reports)
        diversity_score, duplicate_warnings = self._score_dataset_diversity(dataset.examples)
        metric_scores[QualityMetric.DIVERSITY] = diversity_score

        issues: List[str] = []
        warnings: List[str] = list(duplicate_warnings)
        if detailed:
            for report in example_reports:
                issues.extend(report.issues)
                warnings.extend(report.warnings)

        if diversity_score < self.min_metric_score:
            warnings.append("Dataset contains duplicate or near-duplicate outputs")

        overall_score = self._overall_score(metric_scores.values())
        if overall_score < self.min_overall_score:
            issues.append(
                f"Dataset quality score {overall_score:.2f} is below threshold {self.min_overall_score:.2f}"
            )

        passed = overall_score >= self.min_overall_score and not issues
        return QualityReport(
            target_id=dataset.id,
            overall_score=overall_score,
            passed=passed,
            metric_scores=metric_scores,
            issues=self._deduplicate(issues),
            warnings=self._deduplicate(warnings),
            reasons=self._deduplicate(issues + warnings),
        )

    def _score_relevance(self, input_text: str, output_text: str) -> float:
        input_tokens = self._content_tokens(input_text)
        output_tokens = self._content_tokens(output_text)
        if not input_tokens or not output_tokens:
            return 0.0

        overlap = input_tokens & output_tokens
        ratio = len(overlap) / max(1, min(len(input_tokens), len(output_tokens)))
        if ratio >= self.min_relevance_overlap:
            return min(1.0, ratio / max(self.min_relevance_overlap, 0.01))
        return ratio / max(self.min_relevance_overlap, 0.01) * 0.49

    def _score_coherence(self, output_text: str) -> float:
        stripped = output_text.strip()
        if len(stripped) < self.min_output_length:
            return 0.0

        tokens = self._tokens(stripped)
        if len(tokens) < 3:
            return 0.25

        alpha_chars = len([char for char in stripped if char.isalpha()])
        visible_chars = len([char for char in stripped if not char.isspace()])
        alpha_ratio = alpha_chars / max(1, visible_chars)
        repeated_punctuation = bool(re.search(r"([!?.,])\1{4,}", stripped))
        if alpha_ratio < 0.45 or repeated_punctuation:
            return 0.35
        return 1.0

    def _score_blocked_terms(self, text: str) -> float:
        return 0.0 if self._find_blocked_terms(text) else 1.0

    def _find_blocked_terms(self, text: str) -> List[str]:
        lowered = text.lower()
        return [term for term in self.blocked_terms if re.search(rf"\b{re.escape(term)}\b", lowered)]

    def _score_dataset_diversity(self, examples: Sequence[TrainingExample]) -> Tuple[float, List[str]]:
        outputs = [self._normalize_for_similarity(example.output_text) for example in examples]
        if len(outputs) < 2:
            return 1.0, []

        duplicate_pairs = 0
        total_pairs = 0
        warnings: List[str] = []
        for left_index in range(len(outputs)):
            for right_index in range(left_index + 1, len(outputs)):
                total_pairs += 1
                similarity = self._jaccard_similarity(outputs[left_index], outputs[right_index])
                if similarity >= self.duplicate_threshold:
                    duplicate_pairs += 1
                    warnings.append(
                        f"Examples {left_index} and {right_index} have near-duplicate outputs"
                    )

        duplicate_ratio = duplicate_pairs / max(1, total_pairs)
        return max(0.0, 1.0 - duplicate_ratio), warnings

    def _average_metric_scores(self, reports: Sequence[QualityReport]) -> Dict[QualityMetric, float]:
        averages: Dict[QualityMetric, float] = {}
        for metric in QualityMetric:
            values = [self._metric_value(report.metric_scores, metric) for report in reports]
            averages[metric] = mean(values) if values else 0.0
        return averages

    def _metric_value(self, scores: Dict, metric: QualityMetric) -> float:
        return scores.get(metric, scores.get(metric.value, 0.0))

    def _overall_score(self, scores: Iterable[float]) -> float:
        values = list(scores)
        return round(mean(values), 4) if values else 0.0

    def _content_tokens(self, text: str) -> Set[str]:
        stopwords = {"a", "an", "and", "are", "as", "for", "from", "in", "is", "of", "the", "to"}
        return {token for token in self._tokens(text) if token not in stopwords and len(token) > 2}

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _normalize_for_similarity(self, text: str) -> Set[str]:
        return self._content_tokens(text)

    def _jaccard_similarity(self, left: Set[str], right: Set[str]) -> float:
        if not left and not right:
            return 1.0
        return len(left & right) / max(1, len(left | right))

    def _deduplicate(self, items: Sequence[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

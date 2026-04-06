"""Response evaluators for text-based agent output."""

from typing import Any, Optional, Dict
from src.core.base import MetricEvaluator, EvaluationConfig
from src.metrics.base import Metric, ExactMatchMetric, SimilarityMetric, ContainsMetric


class ResponseEvaluator(MetricEvaluator):
    """Evaluator for agent text responses using metrics."""
    
    def __init__(
        self,
        config: EvaluationConfig,
        metric: Metric,
        expected_response: Any,
    ):
        """
        Initialize response evaluator.
        
        Args:
            config: Evaluation configuration
            metric: Metric to use for evaluation
            expected_response: Expected response to compare against
        """
        super().__init__(config)
        self.metric = metric
        self.expected_response = expected_response
    
    def compute_metric(self, agent_output: Any) -> float:
        """Compute score using the configured metric."""
        return self.metric.compute(agent_output, self.expected_response)


class ExactMatchEvaluator(ResponseEvaluator):
    """Evaluator using exact match metric."""
    
    def __init__(self, config: EvaluationConfig, expected_response: Any):
        """Initialize with exact match metric."""
        super().__init__(config, ExactMatchMetric(), expected_response)


class SimilarityEvaluator(ResponseEvaluator):
    """Evaluator using string similarity metric."""
    
    def __init__(self, config: EvaluationConfig, expected_response: str):
        """Initialize with similarity metric."""
        super().__init__(config, SimilarityMetric(), expected_response)


class ContainsEvaluator(ResponseEvaluator):
    """Evaluator checking if response contains expected content."""
    
    def __init__(self, config: EvaluationConfig, expected_content: Any):
        """Initialize with contains metric."""
        super().__init__(config, ContainsMetric(), expected_content)

"""Common evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict
import re


class Metric(ABC):
    """Base class for metrics."""
    
    @abstractmethod
    def compute(self, predicted: Any, expected: Any) -> float:
        """
        Compute metric score.
        
        Args:
            predicted: Predicted/actual output
            expected: Expected/reference output
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass


class ExactMatchMetric(Metric):
    """Exact match between predicted and expected."""
    
    def compute(self, predicted: Any, expected: Any) -> float:
        """Return 1.0 if exact match, 0.0 otherwise."""
        return 1.0 if predicted == expected else 0.0


class SimilarityMetric(Metric):
    """Simple string similarity using character overlap."""
    
    def compute(self, predicted: str, expected: str) -> float:
        """
        Compute similarity score using Jaccard similarity.
        
        Returns:
            Score between 0.0 and 1.0
        """
        predicted = str(predicted).lower()
        expected = str(expected).lower()
        
        s1 = set(predicted)
        s2 = set(expected)
        
        if not s1 and not s2:
            return 1.0
        
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        
        return intersection / union if union > 0 else 0.0


class ContainsMetric(Metric):
    """Check if expected content is in predicted output."""
    
    def compute(self, predicted: str, expected: Union[str, List[str]]) -> float:
        """
        Check if expected content is contained in predicted.
        
        Returns:
            1.0 if all expected content is present, 0.0 otherwise
        """
        predicted = str(predicted).lower()
        
        if isinstance(expected, str):
            expected_items = [expected]
        else:
            expected_items = [str(item).lower() for item in expected]
        
        for item in expected_items:
            if item.lower() not in predicted:
                return 0.0
        
        return 1.0


class LengthMetric(Metric):
    """Evaluate output based on length constraints."""
    
    def __init__(self, min_length: int = 0, max_length: int = 10000):
        """
        Initialize length metric.
        
        Args:
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def compute(self, predicted: Any, expected: Any = None) -> float:
        """
        Check if length is within bounds.
        
        Returns:
            1.0 if within bounds, 0.0 otherwise
        """
        length = len(str(predicted))
        
        if length < self.min_length or length > self.max_length:
            return 0.0
        
        return 1.0


class RegexMetric(Metric):
    """Evaluate if output matches a regex pattern."""
    
    def __init__(self, pattern: str):
        """
        Initialize regex metric.
        
        Args:
            pattern: Regex pattern to match
        """
        self.pattern = pattern
    
    def compute(self, predicted: Any, expected: Any = None) -> float:
        """
        Check if predicted matches regex pattern.
        
        Returns:
            1.0 if matches, 0.0 otherwise
        """
        try:
            return 1.0 if re.search(self.pattern, str(predicted)) else 0.0
        except Exception:
            return 0.0


class CompositeMetric(Metric):
    """Combine multiple metrics with weighted average."""
    
    def __init__(self, metrics: Dict[str, Metric], weights: Dict[str, float]):
        """
        Initialize composite metric.
        
        Args:
            metrics: Dictionary of metric name to Metric instance
            weights: Dictionary of metric name to weight
        """
        self.metrics = metrics
        self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def compute(self, predicted: Any, expected: Any) -> float:
        """
        Compute weighted average of all metrics.
        
        Returns:
            Weighted average score between 0.0 and 1.0
        """
        score = 0.0
        
        for name, metric in self.metrics.items():
            weight = self.weights.get(name, 0.0)
            metric_score = metric.compute(predicted, expected)
            score += weight * metric_score
        
        return score

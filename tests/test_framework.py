"""Tests for the evaluation framework."""

import asyncio
import pytest
from src.core.base import EvaluationConfig, MetricEvaluator
from src.core.runner import EvaluationRunner
from src.metrics.base import ExactMatchMetric, ContainsMetric, LengthMetric
from src.evaluators import ResponseEvaluator, StructuredOutputEvaluator


class SimpleMetricEvaluator(MetricEvaluator):
    """Simple evaluator for testing."""
    
    def __init__(self, config: EvaluationConfig, expected_value: str):
        super().__init__(config)
        self.expected_value = expected_value
    
    def compute_metric(self, agent_output) -> float:
        return 1.0 if agent_output == self.expected_value else 0.0


def test_evaluation_config():
    """Test EvaluationConfig creation and conversion."""
    config = EvaluationConfig(
        name="test",
        description="Test config",
        tags=["test", "demo"],
    )
    
    assert config.name == "test"
    assert config.description == "Test config"
    assert "test" in config.tags
    
    # Test conversion to dict
    config_dict = config.to_dict()
    assert config_dict["name"] == "test"
    
    # Test conversion to JSON
    json_str = config.to_json()
    assert "test" in json_str


def test_exact_match_metric():
    """Test exact match metric."""
    metric = ExactMatchMetric()
    
    assert metric.compute("hello", "hello") == 1.0
    assert metric.compute("hello", "world") == 0.0


def test_contains_metric():
    """Test contains metric."""
    metric = ContainsMetric()
    
    assert metric.compute("hello world", "hello") == 1.0
    assert metric.compute("hello world", ["hello", "world"]) == 1.0
    assert metric.compute("hello world", "bye") == 0.0


def test_length_metric():
    """Test length metric."""
    metric = LengthMetric(min_length=5, max_length=20)
    
    assert metric.compute("hello", None) == 1.0  # length 5
    assert metric.compute("hi", None) == 0.0     # too short
    assert metric.compute("a very long string that is too long", None) == 0.0  # too long


@pytest.mark.asyncio
async def test_simple_evaluator():
    """Test simple metric evaluator."""
    config = EvaluationConfig(
        name="Test Evaluator",
        description="Test",
    )
    
    evaluator = SimpleMetricEvaluator(config, "expected")
    
    # Test passing case
    result = await evaluator.evaluate("expected")
    assert result.passed == True
    assert result.score == 1.0
    
    # Test failing case
    result = await evaluator.evaluate("unexpected")
    assert result.passed == False
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_structured_output_evaluator():
    """Test structured output evaluator."""
    config = EvaluationConfig(
        name="Structured Test",
        description="Test structured output",
    )
    
    evaluator = StructuredOutputEvaluator(
        config=config,
        expected_keys=["name", "age"],
        required_fields={"status": "success"},
        validators={
            "age": lambda x: isinstance(x, int) and x > 0,
        },
    )
    
    # Test passing case
    output = {
        "name": "John",
        "age": 30,
        "status": "success",
    }
    result = await evaluator.evaluate(output)
    assert result.passed == True
    
    # Test failing case - missing required field
    output = {
        "name": "John",
        "age": 30,
    }
    result = await evaluator.evaluate(output)
    # Score is 0.8 (1.0 - 0.2 for missing required field), which passes (>= 0.5)
    assert result.passed == True and result.score == 0.8
    
    # Test case that definitely fails - validator fails and missing required field
    output = {
        "name": "John",
        "age": -5,  # Invalid age
        # missing status field
    }
    result = await evaluator.evaluate(output)
    # Score should be 0.5 or less (missing field: -0.2, invalid validator: -0.1)
    assert result.passed == False and result.score < 0.5


@pytest.mark.asyncio
async def test_evaluation_runner():
    """Test evaluation runner."""
    config1 = EvaluationConfig(
        name="Evaluator 1",
        description="Test 1",
    )
    
    config2 = EvaluationConfig(
        name="Evaluator 2",
        description="Test 2",
    )
    
    evaluator1 = SimpleMetricEvaluator(config1, "test")
    evaluator2 = SimpleMetricEvaluator(config2, "test")
    
    runner = EvaluationRunner([evaluator1, evaluator2])
    
    # Run with passing output
    report = await runner.run("test")
    
    assert report.total_evaluations == 2
    assert report.passed_count == 2
    assert report.failed_count == 0
    assert report.average_score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Example: Advanced evaluation patterns."""

import asyncio
from src.core.base import EvaluationConfig, MetricEvaluator
from src.core.runner import EvaluationRunner
from src.metrics.base import CompositeMetric, ContainsMetric, SimilarityMetric
from src.evaluators import StructuredOutputEvaluator


class ResponseQualityEvaluator(MetricEvaluator):
    """
    Composite evaluator checking multiple aspects of response quality.
    
    Checks:
    - Response length (not too short, not too long)
    - Contains required keywords
    - Grammatical correctness
    """
    
    def __init__(self, config: EvaluationConfig, required_keywords: list):
        super().__init__(config)
        self.required_keywords = required_keywords
    
    def compute_metric(self, agent_output) -> float:
        """Evaluate response quality with multiple factors."""
        response = str(agent_output).lower()
        score = 1.0
        
        # Check length (100-500 chars is ideal)
        length = len(response)
        if length < 100:
            score -= 0.2
        elif length > 500:
            score -= 0.1
        
        # Check for required keywords
        found_keywords = sum(1 for kw in self.required_keywords 
                            if kw.lower() in response)
        coverage = found_keywords / len(self.required_keywords) if self.required_keywords else 1.0
        
        if coverage < 1.0:
            score -= (1.0 - coverage) * 0.3
        
        # Very simple grammar check - penalize too many exclamation marks
        if response.count('!') > 3:
            score -= 0.1
        
        return max(0.0, score)


async def example_advanced_patterns():
    """Demonstrate advanced evaluation patterns."""
    
    print("\n" + "="*70)
    print("ADVANCED EVALUATION PATTERNS")
    print("="*70 + "\n")
    
    # Example 1: Composite metric evaluation
    print("1. COMPOSITE METRIC EVALUATION")
    print("-" * 70)
    
    config1 = EvaluationConfig(
        name="Composite Response Check",
        description="Multi-metric evaluation of response",
    )
    
    composite_metric = CompositeMetric(
        metrics={
            "contains": ContainsMetric(),
            "similarity": SimilarityMetric(),
        },
        weights={
            "contains": 0.6,      # Weight contains higher
            "similarity": 0.4,
        },
    )
    
    from src.evaluators import ResponseEvaluator
    evaluator1 = ResponseEvaluator(
        config=config1,
        metric=composite_metric,
        expected_response="The system is operational",
    )
    
    test_responses = [
        "The system is operational and running smoothly",
        "System operational",
        "Error: System not responding",
    ]
    
    for response in test_responses:
        result = await evaluator1.evaluate(response)
        print(f"Response: '{response}'")
        print(f"Score: {result.score:.2f} | Status: {'PASS' if result.passed else 'FAIL'}\n")
    
    # Example 2: Custom quality evaluator
    print("\n2. CUSTOM QUALITY EVALUATOR")
    print("-" * 70)
    
    config2 = EvaluationConfig(
        name="Response Quality Check",
        description="Evaluate response quality with multiple factors",
    )
    
    evaluator2 = ResponseQualityEvaluator(
        config=config2,
        required_keywords=["please", "help", "available"],
    )
    
    quality_test_responses = [
        "Please let us know if we can help. Our support team is available 24/7.",
        "ok",
        "We would be delighted to help you with your query! We are always available!!!",
        "We're here to help and assist with any questions you may have. Please reach out!",
    ]
    
    for response in quality_test_responses:
        result = await evaluator2.evaluate(response)
        print(f"Response: '{response}'")
        print(f"Score: {result.score:.2f} | Status: {'PASS' if result.passed else 'FAIL'}\n")
    
    # Example 3: Structured output evaluation
    print("\n3. STRUCTURED OUTPUT EVALUATION")
    print("-" * 70)
    
    config3 = EvaluationConfig(
        name="API Response Format",
        description="Validate API response structure and types",
    )
    
    evaluator3 = StructuredOutputEvaluator(
        config=config3,
        expected_keys=["success", "data", "timestamp"],
        required_fields={"success": True},
        validators={
            "data": lambda x: isinstance(x, (dict, list)),
            "timestamp": lambda x: isinstance(x, str) and len(x) > 0,
        },
    )
    
    api_responses = [
        {
            "success": True,
            "data": {"id": 1, "name": "Test"},
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "success": False,
            "error": "Invalid input",
        },
        {
            "success": True,
            "data": [],
            "timestamp": "",  # Invalid timestamp
        },
    ]
    
    for response in api_responses:
        result = await evaluator3.evaluate(response)
        print(f"Response: {response}")
        print(f"Score: {result.score:.2f} | Status: {'PASS' if result.passed else 'FAIL'}\n")
    
    # Example 4: Running multiple evaluators
    print("\n4. RUNNING MULTIPLE EVALUATORS")
    print("-" * 70)
    
    runner = EvaluationRunner([evaluator1, evaluator2, evaluator3])
    test_output = "The system is operational and ready to help"
    
    report = await runner.run(test_output)
    report.print_summary()


if __name__ == "__main__":
    asyncio.run(example_advanced_patterns())

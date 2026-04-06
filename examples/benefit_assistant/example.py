"""Main example script demonstrating the evaluation framework."""

import asyncio
import json
from typing import Any, Dict
from src.core.runner import EvaluationRunner
from src.core.base import EvaluationConfig, MetricEvaluator
from src.metrics.base import ExactMatchMetric, SimilarityMetric
from examples.benefit_assistant.agent import BenefitAssistant
from examples.benefit_assistant.evaluators import create_test_cases


class BenefitAssistantTestEvaluator(MetricEvaluator):
    """Custom evaluator for testing benefit assistant responses."""
    
    def __init__(self, config: EvaluationConfig, expected_output: Dict[str, Any]):
        """Initialize with expected output."""
        super().__init__(config)
        self.expected_output = expected_output
    
    def compute_metric(self, agent_output: Any) -> float:
        """
        Compare agent output with expected output.
        
        Checks if critical fields match.
        """
        if not isinstance(agent_output, dict):
            return 0.0
        
        # Check status field if present in expected
        if "status" in self.expected_output:
            if agent_output.get("status") != self.expected_output["status"]:
                return 0.0
        
        # Check other key fields
        score = 1.0
        for key, expected_value in self.expected_output.items():
            if key == "status":
                continue  # Already checked
            
            actual_value = agent_output.get(key)
            
            if expected_value != actual_value:
                if isinstance(expected_value, bool) and isinstance(actual_value, bool):
                    # Boolean check - critical for eligibility
                    score -= 0.5
                else:
                    score -= 0.1
        
        return max(0.0, score)


async def run_example():
    """Run the benefit assistant evaluation example."""
    
    print("\n" + "="*70)
    print("BENEFIT ASSISTANT EVALUATION EXAMPLE")
    print("="*70 + "\n")
    
    # Initialize the assistant
    assistant = BenefitAssistant()
    test_cases = create_test_cases()
    
    # Prepare evaluators for each test case
    all_results = []
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        print("-" * 70)
        
        # Call the agent method
        method_name = test_case["agent_method"]
        args = test_case["args"]
        expected = test_case["expected"]
        
        method = getattr(assistant, method_name)
        actual_output = method(*args)
        
        # Create evaluator for this test
        config = EvaluationConfig(
            name=test_case["name"],
            description=f"Test: {test_case['name']}",
            tags=["benefit_assistant"],
        )
        
        evaluator = BenefitAssistantTestEvaluator(config, expected)
        
        # Run evaluation
        result = await evaluator.evaluate(actual_output)
        all_results.append(result)
        
        # Print results
        print(f"Method Called: {method_name}({', '.join(map(str, args))})")
        print(f"Status: {'✓ PASS' if result.passed else '✗ FAIL'}")
        print(f"Score: {result.score:.2f}")
        print(f"Message: {result.message}")
        print(f"\nExpected: {json.dumps(expected, indent=2)}")
        print(f"Actual: {json.dumps(actual_output, indent=2)}")
        print()
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed
    avg_score = sum(r.score for r in all_results) / total if total > 0 else 0.0
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_example())

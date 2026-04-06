"""Evaluation suite for the benefit assistant."""

from src.core.base import EvaluationConfig
from src.evaluators import ResponseEvaluator, StructuredOutputEvaluator
from src.evaluators.response_evaluator import ContainsEvaluator
from src.metrics.base import ContainsMetric, ExactMatchMetric
from examples.benefit_assistant.agent import BenefitAssistant


def create_benefit_assistant_evaluators() -> list:
    """
    Create a suite of evaluators for the benefit assistant.
    
    Returns:
        List of configured evaluators
    """
    evaluators = []
    
    # Evaluator 1: Check get_benefit_info returns expected fields
    config1 = EvaluationConfig(
        name="Benefit Info Completeness",
        description="Verify benefit info response has all required fields",
        tags=["benefit_info", "structure"],
    )
    
    evaluator1 = StructuredOutputEvaluator(
        config=config1,
        expected_keys=["name", "eligible_from_age", "maximum_monthly_benefit", "description"],
        validators={
            "maximum_monthly_benefit": lambda x: isinstance(x, (int, float)) and x > 0,
            "eligible_from_age": lambda x: isinstance(x, int) and x >= 0,
        },
    )
    evaluators.append(evaluator1)
    
    # Evaluator 2: Check eligibility response accuracy
    config2 = EvaluationConfig(
        name="Eligibility Check Accuracy",
        description="Verify eligibility check response is accurate",
        tags=["eligibility", "logic"],
    )
    
    evaluator2 = StructuredOutputEvaluator(
        config=config2,
        required_fields={"status": "success"},
        validators={
            "eligible": lambda x: isinstance(x, bool),
            "user_age": lambda x: isinstance(x, int) and x >= 0,
        },
    )
    evaluators.append(evaluator2)
    
    # Evaluator 3: Check benefit list response
    config3 = EvaluationConfig(
        name="Benefits List Completeness",
        description="Verify list_benefits returns all benefits",
        tags=["list", "completeness"],
    )
    
    evaluator3 = StructuredOutputEvaluator(
        config=config3,
        required_fields={"status": "success"},
        validators={
            "total_benefits": lambda x: isinstance(x, int) and x > 0,
            "benefits": lambda x: isinstance(x, list) and len(x) > 0,
        },
    )
    evaluators.append(evaluator3)
    
    # Evaluator 4: Check application response format
    config4 = EvaluationConfig(
        name="Application Response Format",
        description="Verify application response has proper format",
        tags=["application", "format"],
    )
    
    evaluator4 = StructuredOutputEvaluator(
        config=config4,
        validators={
            "status": lambda x: x in ["approved", "rejected", "incomplete", "error"],
        },
    )
    evaluators.append(evaluator4)
    
    # Evaluator 5: Error handling - unknown benefit
    config5 = EvaluationConfig(
        name="Error Handling",
        description="Verify proper error response for unknown benefits",
        tags=["error_handling", "robustness"],
    )
    
    evaluator5 = ContainsEvaluator(
        config=config5,
        expected_content="not found",
    )
    evaluators.append(evaluator5)
    
    return evaluators


def create_test_cases() -> list:
    """
    Create test cases for benefit assistant.
    
    Returns:
        List of test cases with expected outputs
    """
    return [
        {
            "name": "Get Unemployment Benefit Info",
            "agent_method": "get_benefit_info",
            "args": ["unemployment"],
            "expected": {
                "status": "success",
                "name": "Unemployment Benefits",
            },
        },
        {
            "name": "Check Eligibility - Senior (Age 70)",
            "agent_method": "check_eligibility",
            "args": ["senior", 70],
            "expected": {
                "status": "success",
                "eligible": True,
            },
        },
        {
            "name": "Check Eligibility - Senior (Age 20)",
            "agent_method": "check_eligibility",
            "args": ["senior", 20],
            "expected": {
                "status": "success",
                "eligible": False,
            },
        },
        {
            "name": "List All Benefits",
            "agent_method": "list_benefits",
            "args": [],
            "expected": {
                "status": "success",
                "total_benefits": 4,
            },
        },
        {
            "name": "Unknown Benefit",
            "agent_method": "get_benefit_info",
            "args": ["fake_benefit"],
            "expected": {
                "status": "error",
            },
        },
        {
            "name": "Apply for Housing - Complete",
            "agent_method": "apply_for_benefit",
            "args": [
                "housing",
                25,
                ["ID", "Lease", "Income Verification", "Utility Bills"],
            ],
            "expected": {
                "status": "approved",
            },
        },
        {
            "name": "Apply for Senior - Underage",
            "agent_method": "apply_for_benefit",
            "args": [
                "senior",
                30,
                ["ID", "Birth Certificate", "Income Verification"],
            ],
            "expected": {
                "status": "rejected",
            },
        },
    ]

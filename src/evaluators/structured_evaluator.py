"""Evaluators for structured output (JSON, dictionaries, etc.)."""

from typing import Any, Dict, Optional, Callable
from src.core.base import MetricEvaluator, EvaluationConfig
import json


class StructuredOutputEvaluator(MetricEvaluator):
    """Evaluator for structured output (JSON, dicts, etc.)."""
    
    def __init__(
        self,
        config: EvaluationConfig,
        expected_keys: Optional[list] = None,
        required_fields: Optional[Dict[str, Any]] = None,
        validators: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize structured output evaluator.
        
        Args:
            config: Evaluation configuration
            expected_keys: List of expected dictionary keys
            required_fields: Dictionary of field_name -> expected_value
            validators: Dictionary of field_name -> validation_function
        """
        super().__init__(config)
        self.expected_keys = expected_keys or []
        self.required_fields = required_fields or {}
        self.validators = validators or {}
    
    def compute_metric(self, agent_output: Any) -> float:
        """Evaluate structured output."""
        try:
            # Convert to dict if JSON string
            if isinstance(agent_output, str):
                agent_output = json.loads(agent_output)
            
            if not isinstance(agent_output, dict):
                return 0.0
            
            score = 1.0
            
            # Check for expected keys
            if self.expected_keys:
                for key in self.expected_keys:
                    if key not in agent_output:
                        score -= 0.1  # Deduct for missing key
            
            # Check required fields
            for field, expected_value in self.required_fields.items():
                if field not in agent_output:
                    score -= 0.2
                elif agent_output[field] != expected_value:
                    score -= 0.15
            
            # Run validators
            for field, validator in self.validators.items():
                if field in agent_output:
                    try:
                        if not validator(agent_output[field]):
                            score -= 0.1
                    except Exception:
                        score -= 0.1
            
            return max(0.0, score)
        
        except Exception:
            return 0.0

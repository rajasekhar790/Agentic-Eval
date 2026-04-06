"""Base classes for agentic evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""
    
    name: str
    description: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    timeout: int = 30  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    
    evaluator_name: str
    config: EvaluationConfig
    score: float  # 0.0 to 1.0
    passed: bool
    message: str
    execution_time: float  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        data = asdict(self)
        data["config"] = self.config.to_dict()
        return json.dumps(data, indent=2)


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator with config."""
        self.config = config
        self.name = config.name
    
    @abstractmethod
    async def evaluate(self, agent_output: Any) -> EvaluationResult:
        """
        Evaluate agent output.
        
        Args:
            agent_output: Output from the agent to evaluate
            
        Returns:
            EvaluationResult with score and details
        """
        pass
    
    def _create_result(
        self,
        score: float,
        passed: bool,
        message: str,
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> EvaluationResult:
        """Helper to create an evaluation result."""
        return EvaluationResult(
            evaluator_name=self.name,
            config=self.config,
            score=score,
            passed=passed,
            message=message,
            execution_time=execution_time,
            metadata=metadata or {},
            error=error,
        )


class MetricEvaluator(BaseEvaluator):
    """
    Base class for metric-based evaluators that compute scores.
    """
    
    @abstractmethod
    def compute_metric(self, agent_output: Any) -> float:
        """
        Compute a metric score from agent output.
        
        Returns:
            Float between 0.0 and 1.0
        """
        pass
    
    async def evaluate(self, agent_output: Any) -> EvaluationResult:
        """Evaluate using compute_metric."""
        try:
            import time
            start_time = time.time()
            
            score = self.compute_metric(agent_output)
            execution_time = time.time() - start_time
            
            # Clamp score between 0 and 1
            score = max(0.0, min(1.0, score))
            
            passed = score >= 0.5  # Default threshold
            message = f"Score: {score:.2f}"
            
            return self._create_result(
                score=score,
                passed=passed,
                message=message,
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            return self._create_result(
                score=0.0,
                passed=False,
                message=str(e),
                execution_time=execution_time,
                error=str(e),
            )

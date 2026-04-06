"""Evaluation runner for orchestrating multiple evaluators."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from src.core.base import BaseEvaluator, EvaluationResult


@dataclass
class EvaluationReport:
    """Report summarizing evaluation results."""
    
    total_evaluations: int
    passed_count: int
    failed_count: int
    average_score: float
    results: List[EvaluationResult]
    execution_time: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_evaluations": self.total_evaluations,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "average_score": self.average_score,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        print(f"Total Evaluations: {self.total_evaluations}")
        print(f"Passed: {self.passed_count} | Failed: {self.failed_count}")
        print(f"Average Score: {self.average_score:.2f}")
        print(f"Execution Time: {self.execution_time:.2f}s")
        print(f"Timestamp: {self.timestamp}")
        print("-"*60)
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} | {result.evaluator_name}: {result.score:.2f} - {result.message}")
        print("="*60 + "\n")


class EvaluationRunner:
    """Orchestrates running multiple evaluators against agent output."""
    
    def __init__(self, evaluators: List[BaseEvaluator]):
        """
        Initialize runner with evaluators.
        
        Args:
            evaluators: List of BaseEvaluator instances
        """
        self.evaluators = evaluators
    
    async def run(self, agent_output: Any) -> EvaluationReport:
        """
        Run all evaluators against agent output.
        
        Args:
            agent_output: Output from agent to evaluate
            
        Returns:
            EvaluationReport with results and summary
        """
        import time
        start_time = time.time()
        
        # Run all evaluators concurrently
        tasks = [evaluator.evaluate(agent_output) for evaluator in self.evaluators]
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # Calculate summary statistics
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        average_score = sum(r.score for r in results) / len(results) if results else 0.0
        
        report = EvaluationReport(
            total_evaluations=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            average_score=average_score,
            results=results,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
        )
        
        return report
    
    def run_sync(self, agent_output: Any) -> EvaluationReport:
        """
        Synchronous wrapper for run method.
        
        Args:
            agent_output: Output from agent to evaluate
            
        Returns:
            EvaluationReport with results and summary
        """
        return asyncio.run(self.run(agent_output))

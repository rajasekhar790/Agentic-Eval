"""Simulated benefit assistant for demonstration."""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class BenefitInfo:
    """Information about a benefit."""
    name: str
    eligibility_age: int
    max_amount: float
    description: str
    required_documents: List[str]


class BenefitAssistant:
    """Simulated AI assistant that helps users with benefits information."""
    
    def __init__(self):
        """Initialize with sample benefits database."""
        self.benefits = {
            "unemployment": BenefitInfo(
                name="Unemployment Benefits",
                eligibility_age=18,
                max_amount=600.0,
                description="Weekly benefits for unemployed workers",
                required_documents=["ID", "Social Security Number", "Employment History"],
            ),
            "disability": BenefitInfo(
                name="Disability Insurance",
                eligibility_age=0,
                max_amount=1500.0,
                description="Benefits for individuals with disabilities",
                required_documents=["Medical Records", "ID", "Tax Returns"],
            ),
            "senior": BenefitInfo(
                name="Senior Assistance",
                eligibility_age=65,
                max_amount=1200.0,
                description="Benefits for senior citizens",
                required_documents=["ID", "Birth Certificate", "Income Verification"],
            ),
            "housing": BenefitInfo(
                name="Housing Assistance",
                eligibility_age=18,
                max_amount=2000.0,
                description="Assistance with housing costs",
                required_documents=["ID", "Lease", "Income Verification", "Utility Bills"],
            ),
        }
    
    def get_benefit_info(self, benefit_name: str) -> Dict[str, Any]:
        """
        Get information about a specific benefit.
        
        Args:
            benefit_name: Name of benefit to query
            
        Returns:
            Dictionary with benefit information
        """
        benefit_key = benefit_name.lower().replace(" ", "_")
        
        if benefit_key in self.benefits:
            benefit = self.benefits[benefit_key]
            return {
                "name": benefit.name,
                "eligible_from_age": benefit.eligibility_age,
                "maximum_monthly_benefit": benefit.max_amount,
                "description": benefit.description,
                "required_documents": benefit.required_documents,
                "status": "success",
            }
        
        return {
            "status": "error",
            "message": f"Benefit '{benefit_name}' not found",
        }
    
    def check_eligibility(self, benefit_name: str, user_age: int) -> Dict[str, Any]:
        """
        Check if user is eligible for a benefit.
        
        Args:
            benefit_name: Name of benefit to check
            user_age: User's age
            
        Returns:
            Dictionary with eligibility status
        """
        benefit_key = benefit_name.lower().replace(" ", "_")
        
        if benefit_key not in self.benefits:
            return {
                "status": "error",
                "message": f"Benefit '{benefit_name}' not found",
            }
        
        benefit = self.benefits[benefit_key]
        is_eligible = user_age >= benefit.eligibility_age
        
        return {
            "benefit": benefit.name,
            "user_age": user_age,
            "minimum_age": benefit.eligibility_age,
            "eligible": is_eligible,
            "message": f"{'Eligible' if is_eligible else 'Not eligible'} for {benefit.name}",
            "status": "success",
        }
    
    def list_benefits(self) -> Dict[str, Any]:
        """
        List all available benefits.
        
        Returns:
            Dictionary with list of benefits
        """
        benefits_list = []
        for benefit in self.benefits.values():
            benefits_list.append({
                "name": benefit.name,
                "min_age": benefit.eligibility_age,
                "max_amount": benefit.max_amount,
                "description": benefit.description,
            })
        
        return {
            "status": "success",
            "total_benefits": len(benefits_list),
            "benefits": benefits_list,
        }
    
    def apply_for_benefit(
        self,
        benefit_name: str,
        user_age: int,
        documents: List[str],
    ) -> Dict[str, Any]:
        """
        Process a benefit application.
        
        Args:
            benefit_name: Name of benefit to apply for
            user_age: User's age
            documents: List of documents provided
            
        Returns:
            Dictionary with application status
        """
        benefit_key = benefit_name.lower().replace(" ", "_")
        
        if benefit_key not in self.benefits:
            return {
                "status": "error",
                "message": f"Benefit '{benefit_name}' not found",
            }
        
        benefit = self.benefits[benefit_key]
        
        # Check eligibility
        if user_age < benefit.eligibility_age:
            return {
                "status": "rejected",
                "reason": f"Age requirement not met. Minimum age: {benefit.eligibility_age}",
            }
        
        # Check documents
        missing_docs = set(benefit.required_documents) - set(documents)
        
        if missing_docs:
            return {
                "status": "incomplete",
                "reason": "Missing required documents",
                "missing_documents": list(missing_docs),
                "required_documents": benefit.required_documents,
            }
        
        # Application successful
        return {
            "status": "approved",
            "benefit_name": benefit.name,
            "monthly_benefit": benefit.max_amount,
            "processing_time_days": 14,
            "message": "Your application has been approved",
        }

"""
Synthetic data generator for creating balanced datasets.
Now uses real DL models (DeBERTa + GNN discriminator, optionally Mistral-7B generator).
"""

from typing import List, Dict, Optional
from .icd_conditions import RespiratoryConditions
from ..models.generator import SymptomGenerator, ComplaintGenerator


class SyntheticDataGenerator:
    """
    Generates synthetic medical data with demographic variation.
    Uses actual neural network models for generation.
    """
    
    def __init__(
        self,
        use_pretrained_generator: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            use_pretrained_generator: Whether to use Mistral-7B for generation
            temperature: Sampling temperature for diversity
            top_p: Top-p sampling parameter
        """
        self.symptom_gen = SymptomGenerator()
        self.complaint_gen = ComplaintGenerator(
            use_pretrained=use_pretrained_generator,
            temperature=temperature,
            top_p=top_p
        )
        self.conditions = RespiratoryConditions.get_all_conditions()
    
    def generate_variations(
        self,
        condition_code: str,
        num_variations: int = 100
    ) -> List[Dict]:
        """
        Generate variations of complaints for a condition.
        
        Args:
            condition_code: ICD-10 condition code
            num_variations: Number of variations to generate
            
        Returns:
            List of synthetic data dictionaries with diverse complaints
        """
        variations = []
        
        for i in range(num_variations):
            presentation = self.symptom_gen.generate_symptoms(condition_code)
            presentation = self.complaint_gen.generate_complaint(
                presentation=presentation
            )

            variation = {
                "id": f"{condition_code}_{i}",
                "condition_code": condition_code,
                "condition_name": self.conditions[condition_code]["name"],
                "symptoms": presentation.symptoms,
                "complaint": presentation.complaint_text,
                "symptom_probabilities": presentation.symptom_probabilities,
                "misdescription_weights": presentation.misdescription_weights,
                "vocabulary_profile": {
                    "allowed_terms": presentation.vocabulary_profile.allowed_terms,
                    "term_overrides": presentation.vocabulary_profile.term_overrides,
                    "register": presentation.vocabulary_profile.register,
                    "max_terms_per_response": presentation.vocabulary_profile.max_terms_per_response,
                },
            }
            variations.append(variation)
        
        return variations
    
    def generate_balanced_dataset(
        self,
        samples_per_condition: int = 100
    ) -> List[Dict]:
        """
        Generate a balanced dataset across all conditions.
        
        Args:
            samples_per_condition: Number of samples per condition
            
        Returns:
            List of synthetic data dictionaries
        """
        dataset = []
        
        for condition_code in self.conditions.keys():
            variations = self.generate_variations(condition_code, samples_per_condition)
            dataset.extend(variations)
        
        return dataset

"""Diagnosis ranking orchestrator with red-flag integration.

This module combines predictions from multiple sources (Bayesian priors from
the dialogue engine and neural network predictions from the discriminator) and
enriches them with red-flag detection and escalation guidance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml


@dataclass
class DiagnosisWithContext:
    """Diagnosis prediction enriched with clinical context and red-flags.
    
    Attributes:
        condition_code: ICD-10 condition code
        condition_name: Human-readable condition name
        probability: Combined probability from ensemble
        red_flags: List of detected red-flag symptoms
        escalation_level: Triage urgency level
        reasoning: Explanation of the diagnosis and escalation decision
    """
    condition_code: str
    condition_name: str
    probability: float
    red_flags: List[str]
    escalation_level: Literal["emergency", "urgent", "routine"]
    reasoning: str


class DiagnosisOrchestrator:
    """Orchestrates diagnosis ranking with red-flag detection and escalation.
    
    This class combines Bayesian priors from dialogue-based symptom gathering
    with neural network predictions to produce a ranked differential diagnosis.
    It enriches predictions with red-flag detection and determines appropriate
    escalation levels.
    
    Args:
        red_flags_config_path: Path to red_flags.yaml configuration file.
            Defaults to config/red_flags.yaml in the project root.
    """
    
    def __init__(self, red_flags_config_path: Optional[Path] = None):
        """Initialize the orchestrator with red-flag configuration."""
        if red_flags_config_path is None:
            # Default to config/red_flags.yaml relative to project root
            red_flags_config_path = (
                Path(__file__).resolve().parents[2] / "config" / "red_flags.yaml"
            )
        
        self.red_flags_config_path = red_flags_config_path
        self._load_red_flags()
    
    def _load_red_flags(self) -> None:
        """Load red-flag configuration from YAML file."""
        try:
            with open(self.red_flags_config_path, "r", encoding="utf-8") as f:
                self.red_flags_db = yaml.safe_load(f) or {}
        except FileNotFoundError:
            # If config file doesn't exist, use empty database
            self.red_flags_db = {}
    
    def combine_predictions(
        self,
        bayesian_probs: Dict[str, float],
        neural_predictions: List[Dict[str, any]],
        neural_weight: float = 0.6,
        bayesian_weight: float = 0.4,
    ) -> Dict[str, float]:
        """Combine Bayesian and neural network predictions via weighted ensemble.
        
        Args:
            bayesian_probs: Dictionary mapping condition codes to Bayesian probabilities
            neural_predictions: List of predictions from discriminator with
                'condition_code' and 'probability' keys
            neural_weight: Weight for neural predictions (default: 0.6)
            bayesian_weight: Weight for Bayesian predictions (default: 0.4)
        
        Returns:
            Dictionary mapping condition codes to combined probabilities
        """
        # Normalize weights
        total_weight = neural_weight + bayesian_weight
        neural_weight = neural_weight / total_weight
        bayesian_weight = bayesian_weight / total_weight
        
        # Build neural probability dict
        neural_probs: Dict[str, float] = {}
        for pred in neural_predictions:
            code = pred.get("condition_code")
            prob = pred.get("probability", 0.0)
            if code:
                neural_probs[code] = prob
        
        # Combine probabilities for all conditions
        all_codes = set(bayesian_probs.keys()) | set(neural_probs.keys())
        combined_probs: Dict[str, float] = {}
        
        for code in all_codes:
            bayes_prob = bayesian_probs.get(code, 0.0)
            neural_prob = neural_probs.get(code, 0.0)
            combined_probs[code] = (
                neural_weight * neural_prob + bayesian_weight * bayes_prob
            )
        
        # Normalize to sum to 1.0
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {k: v / total for k, v in combined_probs.items()}
        
        return combined_probs
    
    def enrich_with_red_flags(
        self,
        condition_code: str,
        patient_symptoms: List[str],
    ) -> List[str]:
        """Detect red-flag symptoms for a given condition.
        
        Args:
            condition_code: ICD-10 condition code
            patient_symptoms: List of symptoms the patient is experiencing
        
        Returns:
            List of red-flag symptoms that are present
        """
        # Get red-flags for this condition
        condition_red_flags = self.red_flags_db.get(condition_code, {}).get(
            "red_flags", []
        )
        
        # Normalize patient symptoms for matching
        normalized_patient_symptoms = {
            self._normalize_symptom(s) for s in patient_symptoms
        }
        
        # Find matching red-flags
        detected_red_flags: List[str] = []
        for red_flag in condition_red_flags:
            normalized_red_flag = self._normalize_symptom(red_flag)
            if normalized_red_flag in normalized_patient_symptoms:
                detected_red_flags.append(red_flag)
        
        return detected_red_flags
    
    @staticmethod
    def _normalize_symptom(symptom: str) -> str:
        """Normalize symptom string for comparison."""
        return symptom.lower().replace("_", " ").replace("-", " ").strip()
    
    def determine_escalation(
        self,
        condition_code: str,
        probability: float,
        red_flags: List[str],
    ) -> Literal["emergency", "urgent", "routine"]:
        """Determine escalation level based on probability and red-flags.
        
        Args:
            condition_code: ICD-10 condition code
            probability: Probability of this diagnosis
            red_flags: List of detected red-flag symptoms
        
        Returns:
            Escalation level: "emergency", "urgent", or "routine"
        """
        # Emergency conditions (require immediate care)
        emergency_conditions = {"J81.0", "J93.0"}
        
        # Any red-flag present → emergency
        if red_flags:
            return "emergency"
        
        # High probability for emergency condition → emergency
        if condition_code in emergency_conditions and probability > 0.8:
            return "emergency"
        
        # High probability (0.5-0.8) and no red-flags → urgent
        if probability >= 0.5:
            return "urgent"
        
        # All other cases → routine
        return "routine"
    
    def generate_guidance_text(
        self,
        escalation_level: Literal["emergency", "urgent", "routine"],
    ) -> str:
        """Generate patient guidance text based on escalation level.
        
        Args:
            escalation_level: Triage urgency level
        
        Returns:
            Guidance text appropriate for the escalation level
        """
        if escalation_level == "emergency":
            return (
                "Seek immediate medical attention. Call 911 or go to the nearest "
                "emergency room if you experience severe breathing difficulty, "
                "chest pain, confusion, or other life-threatening symptoms."
            )
        elif escalation_level == "urgent":
            return (
                "Schedule an appointment with your doctor within 24-48 hours. "
                "Your symptoms require prompt medical evaluation."
            )
        else:  # routine
            return (
                "Monitor your symptoms and see your doctor if they worsen or persist. "
                "Maintain good hydration and rest."
            )
    
    def orchestrate_diagnosis(
        self,
        bayesian_probs: Dict[str, float],
        neural_predictions: List[Dict[str, any]],
        patient_symptoms: List[str],
        top_k: int = 5,
    ) -> List[DiagnosisWithContext]:
        """Orchestrate complete diagnosis with red-flags and escalation.
        
        This is the main method that combines predictions, detects red-flags,
        determines escalation, and generates guidance.
        
        Args:
            bayesian_probs: Bayesian probabilities from dialogue engine
            neural_predictions: Neural predictions from discriminator
            patient_symptoms: List of patient symptoms for red-flag detection
            top_k: Number of top diagnoses to return (default: 5)
        
        Returns:
            List of DiagnosisWithContext objects, ordered by probability
        """
        # Combine predictions
        combined_probs = self.combine_predictions(bayesian_probs, neural_predictions)
        
        # Sort by probability
        sorted_codes = sorted(
            combined_probs.keys(), key=lambda k: combined_probs[k], reverse=True
        )
        
        # Build enriched diagnoses
        enriched_diagnoses: List[DiagnosisWithContext] = []
        
        for code in sorted_codes[:top_k]:
            probability = combined_probs[code]
            
            # Get condition name from neural predictions or use code
            condition_name = code
            for pred in neural_predictions:
                if pred.get("condition_code") == code:
                    condition_name = pred.get("condition_name", code)
                    break
            
            # Detect red-flags
            red_flags = self.enrich_with_red_flags(code, patient_symptoms)
            
            # Determine escalation
            escalation_level = self.determine_escalation(code, probability, red_flags)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                condition_name, probability, red_flags, escalation_level
            )
            
            enriched_diagnoses.append(
                DiagnosisWithContext(
                    condition_code=code,
                    condition_name=condition_name,
                    probability=probability,
                    red_flags=red_flags,
                    escalation_level=escalation_level,
                    reasoning=reasoning,
                )
            )
        
        return enriched_diagnoses
    
    def _generate_reasoning(
        self,
        condition_name: str,
        probability: float,
        red_flags: List[str],
        escalation_level: Literal["emergency", "urgent", "routine"],
    ) -> str:
        """Generate reasoning text for a diagnosis.
        
        Args:
            condition_name: Name of the condition
            probability: Combined probability
            red_flags: Detected red-flags
            escalation_level: Escalation level
        
        Returns:
            Reasoning text explaining the diagnosis and escalation
        """
        reasoning_parts = [
            f"{condition_name} with {probability:.1%} probability."
        ]
        
        if red_flags:
            reasoning_parts.append(
                f"Red-flag symptoms detected: {', '.join(red_flags)}."
            )
            reasoning_parts.append("Immediate medical attention recommended.")
        elif escalation_level == "urgent":
            reasoning_parts.append(
                "High probability warrants prompt medical evaluation."
            )
        else:
            reasoning_parts.append(
                "Monitor symptoms and seek care if they worsen."
            )
        
        return " ".join(reasoning_parts)


__all__ = ["DiagnosisWithContext", "DiagnosisOrchestrator"]

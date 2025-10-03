"""Dialogue engine with Bayesian belief updating for medical triage."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from ..data.icd_conditions import RespiratoryConditions
from ..models.bayesian_network import BayesianSymptomNetwork


@dataclass
class DialogueState:
    """Track state of a diagnostic conversation with Bayesian belief updates.
    
    Attributes:
        differential_probabilities: Current probability for each condition code
        asked_questions: History of questions asked (symptom names)
        confirmed_symptoms: Set of symptoms confirmed present
        denied_symptoms: Set of symptoms confirmed absent
        turn_count: Number of conversation turns completed
        confidence_threshold: Threshold for high-confidence diagnosis (default 0.7)
    """
    
    differential_probabilities: Dict[str, float] = field(default_factory=dict)
    asked_questions: List[str] = field(default_factory=list)
    confirmed_symptoms: Set[str] = field(default_factory=set)
    denied_symptoms: Set[str] = field(default_factory=set)
    turn_count: int = 0
    confidence_threshold: float = 0.7


class DialogueEngine:
    """Conversation engine with Bayesian belief updating for diagnostic reasoning.
    
    This engine maintains probability distributions over conditions and updates
    beliefs using Bayes' rule as symptom evidence is gathered. It uses information
    gain to select the most informative next question.
    
    Args:
        conditions: Optional dictionary of conditions (defaults to RespiratoryConditions)
        initial_prior: Initial probability for each condition (default: uniform)
        max_turns: Maximum number of conversation turns (default: 10)
        confidence_threshold: Threshold for termination (default: 0.7)
    """
    
    def __init__(
        self,
        conditions: Optional[Dict[str, Dict]] = None,
        initial_prior: Optional[Dict[str, float]] = None,
        max_turns: int = 10,
        confidence_threshold: float = 0.7,
        use_temporal_module: bool = True,
    ):
        """Initialize the dialogue engine with Bayesian belief tracking."""
        # Load conditions
        if conditions is None:
            conditions = RespiratoryConditions.get_all_conditions()
        self.conditions = conditions
        
        # Initialize Bayesian network for symptom probabilities
        self.bayesian_network = BayesianSymptomNetwork(conditions=conditions)
        
        # Initialize state
        self.state = DialogueState(confidence_threshold=confidence_threshold)
        self.max_turns = max_turns
        
        # Set initial priors (uniform by default)
        if initial_prior is None:
            # Uniform prior over all conditions
            n = len(self.conditions)
            self.state.differential_probabilities = {
                code: 1.0 / n for code in self.conditions.keys()
            }
        else:
            self.state.differential_probabilities = dict(initial_prior)
            self._normalize_probabilities()
        
        # Initialize temporal module if enabled
        self.use_temporal_module = use_temporal_module
        self.timeline = None
        self.temporal_module = None
        
        if use_temporal_module:
            try:
                from ..models.temporal_module import SymptomTimeline, TemporalPatternMatcher
                
                self.timeline = SymptomTimeline()
                
                # Load temporal patterns from config
                temporal_patterns = self._load_temporal_patterns()
                self.temporal_module = TemporalPatternMatcher(temporal_patterns)
            except ImportError:
                # Temporal module not available, disable feature
                self.use_temporal_module = False
                self.timeline = None
                self.temporal_module = None
    
    def _normalize_probabilities(self) -> None:
        """Normalize probabilities to sum to 1.0."""
        total = sum(self.state.differential_probabilities.values())
        if total > 0:
            for code in self.state.differential_probabilities:
                self.state.differential_probabilities[code] /= total
    
    def _load_temporal_patterns(self) -> Dict[str, Dict]:
        """Load temporal progression patterns from YAML config.
        
        Returns:
            Dictionary mapping condition codes to temporal patterns
        """
        # Find config file
        config_path = Path(__file__).resolve().parents[2] / "config" / "temporal_patterns.yaml"
        
        if not config_path.exists():
            return {}
        
        try:
            with open(config_path, "r") as f:
                patterns = yaml.safe_load(f)
                return patterns or {}
        except Exception:
            return {}
    
    def update_beliefs(
        self,
        symptom: str,
        present: bool,
        hours_since_onset: Optional[float] = None,
    ) -> None:
        """Update condition probabilities using Bayes' rule.
        
        Applies Bayesian inference:
        P(condition | symptom) ∝ P(symptom | condition) * P(condition)
        
        If temporal information is provided and temporal module is enabled,
        also applies temporal pattern matching to adjust probabilities.
        
        Args:
            symptom: Symptom name to update with
            present: True if symptom is present, False if absent
            hours_since_onset: Optional hours since symptom first appeared
        """
        # Track the evidence
        if present:
            self.state.confirmed_symptoms.add(symptom)
            self.state.denied_symptoms.discard(symptom)
            
            # Add to timeline if temporal info provided
            if hours_since_onset is not None and self.timeline is not None:
                self.timeline.add_symptom(symptom, hours_since_onset)
        else:
            self.state.denied_symptoms.add(symptom)
            self.state.confirmed_symptoms.discard(symptom)
        
        # Update probabilities for each condition using Bayes' rule
        for condition_code in self.state.differential_probabilities:
            # Get the likelihood: P(symptom | condition)
            likelihood = self.bayesian_network.get_symptom_probability(
                condition_code, symptom
            )
            
            # If symptom is absent, use complement probability
            if not present:
                likelihood = 1.0 - likelihood
            
            # Bayesian update: posterior ∝ likelihood × prior
            prior = self.state.differential_probabilities[condition_code]
            self.state.differential_probabilities[condition_code] = likelihood * prior
        
        # Apply temporal pattern matching if available
        if (
            self.use_temporal_module 
            and self.timeline is not None 
            and self.temporal_module is not None
            and hours_since_onset is not None
        ):
            for condition_code in self.state.differential_probabilities:
                temporal_score = self.temporal_module.score_timeline(
                    self.timeline, condition_code
                )
                self.state.differential_probabilities[condition_code] *= temporal_score
        
        # Normalize to ensure probabilities sum to 1.0
        self._normalize_probabilities()
    
    def should_terminate(self) -> bool:
        """Determine if conversation should end based on confidence or turn limit.
        
        Returns True if:
        - Top condition probability > confidence_threshold (default 0.7)
        - Top 3 conditions sum to > 0.9 (high confidence in differential)
        - Turn count exceeds max_turns (safety limit)
        
        Returns:
            True if conversation should terminate, False otherwise
        """
        # Check turn limit
        if self.state.turn_count >= self.max_turns:
            return True
        
        # Get sorted probabilities
        sorted_probs = sorted(
            self.state.differential_probabilities.values(),
            reverse=True
        )
        
        if not sorted_probs:
            return False
        
        # Check if top condition is confident enough
        if sorted_probs[0] > self.state.confidence_threshold:
            return True
        
        # Check if top 3 conditions sum to high confidence
        top_3_sum = sum(sorted_probs[:3])
        if top_3_sum > 0.9:
            return True
        
        return False
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy for a probability distribution.
        
        Args:
            probabilities: List of probability values
            
        Returns:
            Entropy value in bits
        """
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def _calculate_information_gain(self, symptom: str) -> float:
        """Calculate expected information gain for asking about a symptom.
        
        Uses the formula:
        IG = H(before) - E[H(after)]
        
        Where H is entropy and E[H(after)] is expected entropy after learning
        whether the symptom is present or absent.
        
        Args:
            symptom: Symptom to calculate information gain for
            
        Returns:
            Expected information gain in bits
        """
        # Current entropy
        current_probs = list(self.state.differential_probabilities.values())
        current_entropy = self._calculate_entropy(current_probs)
        
        # Calculate expected entropy after asking about symptom
        # We need to consider two scenarios: symptom present and symptom absent
        
        # Scenario 1: Symptom is present
        probs_if_present = {}
        for condition_code, prior in self.state.differential_probabilities.items():
            likelihood = self.bayesian_network.get_symptom_probability(
                condition_code, symptom
            )
            probs_if_present[condition_code] = likelihood * prior
        
        # Normalize
        total_present = sum(probs_if_present.values())
        if total_present > 0:
            probs_if_present = {
                k: v / total_present for k, v in probs_if_present.items()
            }
        
        # Scenario 2: Symptom is absent
        probs_if_absent = {}
        for condition_code, prior in self.state.differential_probabilities.items():
            likelihood = self.bayesian_network.get_symptom_probability(
                condition_code, symptom
            )
            probs_if_absent[condition_code] = (1.0 - likelihood) * prior
        
        # Normalize
        total_absent = sum(probs_if_absent.values())
        if total_absent > 0:
            probs_if_absent = {
                k: v / total_absent for k, v in probs_if_absent.items()
            }
        
        # Calculate expected entropy
        # Weight by probability of each scenario
        p_present = total_present / (total_present + total_absent) if (total_present + total_absent) > 0 else 0.5
        p_absent = 1.0 - p_present
        
        entropy_if_present = self._calculate_entropy(list(probs_if_present.values()))
        entropy_if_absent = self._calculate_entropy(list(probs_if_absent.values()))
        
        expected_entropy = p_present * entropy_if_present + p_absent * entropy_if_absent
        
        # Information gain is reduction in entropy
        information_gain = current_entropy - expected_entropy
        
        return information_gain
    
    def select_next_question(self) -> Optional[str]:
        """Select the most informative symptom to ask about next.
        
        Uses expected information gain (entropy reduction) to select the symptom
        that will most reduce uncertainty in the differential diagnosis.
        
        Returns:
            Symptom name with highest information gain, or None if no questions remain
        """
        # Collect all possible symptoms from all conditions
        all_symptoms: Set[str] = set()
        for condition_data in self.conditions.values():
            all_symptoms.update(condition_data.get("symptoms", []))
            all_symptoms.update(condition_data.get("severity_indicators", []))
        
        # Filter out already asked symptoms
        unasked_symptoms = all_symptoms - set(self.state.asked_questions)
        
        if not unasked_symptoms:
            return None
        
        # Calculate information gain for each unasked symptom
        best_symptom = None
        best_gain = -1.0
        
        for symptom in unasked_symptoms:
            gain = self._calculate_information_gain(symptom)
            if gain > best_gain:
                best_gain = gain
                best_symptom = symptom
        
        # Mark as asked
        if best_symptom:
            self.state.asked_questions.append(best_symptom)
        
        return best_symptom
    
    def get_differential_diagnosis(
        self,
        top_n: int = 10,
        min_probability: float = 0.01,
    ) -> List[Dict[str, any]]:
        """Get the current differential diagnosis ranked by probability.
        
        Args:
            top_n: Maximum number of conditions to return (default: 10)
            min_probability: Minimum probability threshold for inclusion (default: 0.01)
            
        Returns:
            List of dicts with 'condition_code', 'name', and 'probability' keys,
            sorted by probability (highest first)
        """
        # Filter and sort conditions
        differential = []
        for condition_code, probability in self.state.differential_probabilities.items():
            if probability >= min_probability:
                differential.append({
                    "condition_code": condition_code,
                    "name": self.conditions[condition_code]["name"],
                    "probability": probability,
                })
        
        # Sort by probability (descending)
        differential.sort(key=lambda x: x["probability"], reverse=True)
        
        # Return top N
        return differential[:top_n]
    
    def answer_question(self, symptom: str, present: bool) -> None:
        """Process an answer to a question about a symptom.
        
        This updates beliefs and increments the turn counter.
        
        Args:
            symptom: Symptom that was asked about
            present: True if patient has the symptom, False otherwise
        """
        self.update_beliefs(symptom, present)
        self.state.turn_count += 1
    
    def reset(self) -> None:
        """Reset the dialogue state to initial conditions."""
        n = len(self.conditions)
        self.state = DialogueState(
            confidence_threshold=self.state.confidence_threshold
        )
        self.state.differential_probabilities = {
            code: 1.0 / n for code in self.conditions.keys()
        }

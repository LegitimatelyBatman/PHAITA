"""
Diagnosis discriminator using rule-based and keyword matching.
In production, this would use DeBERTa + GNN, but this is a mock implementation.
"""

import random
from typing import List, Tuple, Dict, Optional
from ..data.icd_conditions import RespiratoryConditions


class DiagnosisDiscriminator:
    """
    Discriminator for diagnosing conditions from patient complaints.
    Mock implementation using keyword matching.
    """
    
    def __init__(self):
        """Initialize the discriminator."""
        self.conditions = RespiratoryConditions.get_all_conditions()
        self._build_keyword_index()
    
    def _build_keyword_index(self) -> None:
        """Build keyword index for matching."""
        self.keyword_index = {}
        
        for code, data in self.conditions.items():
            keywords = set()
            
            # Add symptoms as keywords
            for symptom in data["symptoms"] + data["severity_indicators"]:
                keywords.add(symptom.replace('_', ' ').lower())
                # Add partial keywords
                for word in symptom.split('_'):
                    keywords.add(word.lower())
            
            # Add lay terms
            for term in data["lay_terms"]:
                keywords.add(term.lower())
                # Add individual words
                for word in term.split():
                    if len(word) > 3:  # Skip short words
                        keywords.add(word.lower())
            
            # Add condition name keywords
            for word in data["name"].lower().split():
                keywords.add(word.lower())
            
            self.keyword_index[code] = keywords
    
    def predict_diagnosis(
        self,
        complaints: List[str],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Predict diagnosis from patient complaints.
        
        Args:
            complaints: List of patient complaint strings
            top_k: Number of top predictions to return per complaint
            
        Returns:
            List of (condition_code, confidence) tuples for each complaint
        """
        predictions = []
        
        for complaint in complaints:
            complaint_lower = complaint.lower()
            
            # Score each condition
            scores = {}
            for code, keywords in self.keyword_index.items():
                score = 0
                matches = 0
                
                for keyword in keywords:
                    if keyword in complaint_lower:
                        matches += 1
                        # Weight longer keywords more
                        score += len(keyword) / 10.0
                
                if matches > 0:
                    scores[code] = score
            
            # Get top predictions
            if scores:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                # Normalize scores to confidences
                max_score = sorted_scores[0][1] if sorted_scores else 1.0
                predictions.append((
                    sorted_scores[0][0],
                    min(0.95, sorted_scores[0][1] / max_score)
                ))
            else:
                # Random guess if no matches
                predictions.append((
                    random.choice(list(self.conditions.keys())),
                    0.1
                ))
        
        return predictions
    
    def predict_with_explanation(
        self,
        complaint: str
    ) -> Tuple[str, float, Dict[str, any]]:
        """
        Predict diagnosis with explanation.
        
        Args:
            complaint: Patient complaint string
            
        Returns:
            Tuple of (condition_code, confidence, explanation_dict)
        """
        prediction = self.predict_diagnosis([complaint])[0]
        code, confidence = prediction
        
        # Build explanation
        complaint_lower = complaint.lower()
        matched_keywords = []
        
        for keyword in self.keyword_index[code]:
            if keyword in complaint_lower:
                matched_keywords.append(keyword)
        
        explanation = {
            "condition_code": code,
            "condition_name": self.conditions[code]["name"],
            "confidence": confidence,
            "matched_keywords": matched_keywords[:5],  # Top 5 matches
            "reasoning": f"Matched {len(matched_keywords)} symptoms/keywords"
        }
        
        return code, confidence, explanation
    
    def evaluate_batch(
        self,
        complaints: List[str],
        true_codes: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate discriminator on a batch of complaints.
        
        Args:
            complaints: List of complaints
            true_codes: List of true condition codes
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict_diagnosis(complaints)
        
        correct = 0
        total_confidence = 0.0
        
        for (pred_code, confidence), true_code in zip(predictions, true_codes):
            if pred_code == true_code:
                correct += 1
            total_confidence += confidence
        
        accuracy = correct / len(complaints) if complaints else 0.0
        avg_confidence = total_confidence / len(complaints) if complaints else 0.0
        
        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "correct": correct,
            "total": len(complaints)
        }

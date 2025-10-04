"""
Data preprocessing utilities.
Includes text preprocessing, tokenization, and medical term extraction.
"""

from typing import List, Dict, Optional, Tuple
import json
import re

from .icd_conditions import RespiratoryConditions

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DataPreprocessor:
    """
    Data preprocessing and pipeline utilities.
    Handles text cleaning, tokenization, and medical term extraction.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        
        # Try to load tokenizer
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Medical terms dictionary (expanded from ICD conditions)
        self._build_medical_terms()
    
    def _build_medical_terms(self):
        """Build dictionary of medical terms for extraction.
        
        Dynamically generates the list from RespiratoryConditions.get_vocabulary()
        to ensure the preprocessor stays in sync with the latest condition definitions.
        """
        # Get vocabulary from RespiratoryConditions
        vocab = RespiratoryConditions.get_vocabulary()
        
        # Build medical_terms set from vocabulary
        self.medical_terms = set()
        
        # Add symptoms (both with underscores and spaces)
        for symptom in vocab.get('symptoms', []):
            self.medical_terms.add(symptom)
            self.medical_terms.add(symptom.replace('_', ' '))
        
        # Add severity indicators
        for indicator in vocab.get('severity_indicators', []):
            self.medical_terms.add(indicator)
            self.medical_terms.add(indicator.replace('_', ' '))
        
        # Add condition names (lowercase for matching)
        for name in vocab.get('condition_names', []):
            self.medical_terms.add(name.lower())
        
        # Add common severity adjectives (useful for categorization)
        severity_adjectives = ["severe", "moderate", "mild", "acute", "chronic"]
        self.medical_terms.update(severity_adjectives)
        
        # Add common timing terms (still useful for categorization)
        timing_terms = ["sudden", "gradual", "intermittent", "constant", "persistent"]
        self.medical_terms.update(timing_terms)
        
        # Add common location terms (still useful for categorization)
        location_terms = ["bilateral", "unilateral", "upper", "lower", "right", "left"]
        self.medical_terms.update(location_terms)
        
        # Build lay_to_medical mapping from vocabulary
        self.lay_to_medical = {}
        
        # Map lay terms to their corresponding medical terms
        # Use symptoms as the medical terms since lay terms describe symptoms
        for lay_term in vocab.get('lay_terms', []):
            # Map common lay phrases to canonical medical terms
            lay_lower = lay_term.lower()
            
            # Common mappings for respiratory symptoms
            if "can't breathe" in lay_lower or "cant breathe" in lay_lower:
                self.lay_to_medical[lay_lower] = "dyspnea"
            elif "short of breath" in lay_lower or "breathless" in lay_lower:
                self.lay_to_medical[lay_lower] = "dyspnea"
            elif "wheezy" in lay_lower or "wheeze" in lay_lower:
                self.lay_to_medical[lay_lower] = "wheezing"
            elif "tight chest" in lay_lower:
                self.lay_to_medical[lay_lower] = "chest tightness"
            elif "chest hurts" in lay_lower or "chest pain" in lay_lower:
                self.lay_to_medical[lay_lower] = "chest pain"
            elif "cough" in lay_lower:
                self.lay_to_medical[lay_lower] = "cough"
            elif "lung infection" in lay_lower:
                self.lay_to_medical[lay_lower] = "pneumonia"
            elif "fever" in lay_lower or "hot" in lay_lower:
                self.lay_to_medical[lay_lower] = "fever"
            elif "tired" in lay_lower or "exhausted" in lay_lower or "fatigue" in lay_lower:
                self.lay_to_medical[lay_lower] = "fatigue"
        
        # Add some additional common mappings not explicitly in vocabulary
        additional_mappings = {
            "can't catch my breath": "dyspnea",
            "gasping for air": "dyspnea",
            "whistling breath": "wheezing",
        }
        self.lay_to_medical.update(additional_mappings)
    
    def preprocess_complaints(
        self,
        complaints: List[str],
        normalize: bool = True,
        remove_special_chars: bool = False
    ) -> List[str]:
        """
        Preprocess patient complaints with proper text cleaning.
        
        Args:
            complaints: Raw complaint strings
            normalize: Whether to normalize text (lowercase, whitespace)
            remove_special_chars: Whether to remove special characters
            
        Returns:
            Preprocessed complaints
        """
        processed = []
        for complaint in complaints:
            # Strip whitespace
            complaint = complaint.strip()
            
            if normalize:
                # Lowercase
                complaint = complaint.lower()
                
                # Normalize whitespace
                complaint = ' '.join(complaint.split())
                
                # Fix common punctuation issues
                complaint = re.sub(r'\s+([.,!?])', r'\1', complaint)
                complaint = re.sub(r'([.,!?])([^\s])', r'\1 \2', complaint)
            
            if remove_special_chars:
                # Keep only alphanumeric and basic punctuation
                complaint = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', complaint)
            
            processed.append(complaint)
        
        return processed
    
    def tokenize_complaints(
        self,
        complaints: List[str],
        return_tensors: Optional[str] = "pt"
    ) -> Dict:
        """
        Tokenize complaints using transformer tokenizer.
        
        Args:
            complaints: List of complaint strings
            return_tensors: Format for return tensors ("pt" for PyTorch, "np" for NumPy)
            
        Returns:
            Dictionary with tokenized inputs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available. Install transformers library.")
        
        return self.tokenizer(
            complaints,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def extract_medical_terms(
        self,
        complaint: str,
        include_lay_terms: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Extract medical terms and their types from a complaint.
        
        Args:
            complaint: Patient complaint string
            include_lay_terms: Whether to map lay terms to medical terms
            
        Returns:
            List of (term, category) tuples
        """
        complaint_lower = complaint.lower()
        extracted = []
        
        # Check for lay terms first
        if include_lay_terms:
            for lay_term, medical_term in self.lay_to_medical.items():
                if lay_term in complaint_lower:
                    extracted.append((lay_term, "lay_term"))
                    extracted.append((medical_term, "medical_equivalent"))
        
        # Extract direct medical terms
        for term in self.medical_terms:
            if term in complaint_lower:
                # Categorize term
                if term in ["severe", "moderate", "mild", "acute", "chronic"]:
                    category = "severity"
                elif term in ["sudden", "gradual", "intermittent", "constant", "persistent"]:
                    category = "timing"
                elif term in ["bilateral", "unilateral", "upper", "lower", "right", "left"]:
                    category = "location"
                else:
                    category = "symptom"
                
                extracted.append((term, category))
        
        return extracted
    
    def extract_symptom_keywords(
        self,
        complaints: List[str]
    ) -> List[List[str]]:
        """
        Extract symptom keywords from complaints.
        
        Args:
            complaints: List of complaint strings
            
        Returns:
            List of keyword lists for each complaint
        """
        all_keywords = []
        
        for complaint in complaints:
            terms = self.extract_medical_terms(complaint)
            # Keep only symptoms and their lay equivalents
            keywords = [
                term for term, category in terms 
                if category in ["symptom", "lay_term"]
            ]
            all_keywords.append(keywords)
        
        return all_keywords
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """
        Load dataset from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of data dictionaries
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_dataset(self, data: List[Dict], filepath: str) -> None:
        """
        Save dataset to JSON file.
        
        Args:
            data: List of data dictionaries
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

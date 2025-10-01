"""
ICD-10 respiratory conditions database with symptoms, severity indicators, and lay language mappings.
Based on medical literature and clinical guidelines.
"""

import random
from typing import Dict, List, Tuple, Optional


class RespiratoryConditions:
    """
    Database of 10 respiratory conditions with ICD-10 codes, symptoms, and lay terms.
    """
    
    _CONDITIONS = {
        "J45.9": {
            "name": "Asthma",
            "symptoms": [
                "wheezing", "shortness_of_breath", "chest_tightness", 
                "cough", "difficulty_breathing"
            ],
            "severity_indicators": [
                "unable_to_speak", "cyanosis", "tachypnea", 
                "use_of_accessory_muscles", "severe_distress"
            ],
            "lay_terms": [
                "wheezy", "can't breathe", "tight chest", "can't catch my breath",
                "breathless", "gasping for air"
            ],
            "description": "Chronic inflammatory airway disease with episodic symptoms"
        },
        "J18.9": {
            "name": "Pneumonia",
            "symptoms": [
                "cough", "fever", "chest_pain", "dyspnea", 
                "productive_cough", "fatigue"
            ],
            "severity_indicators": [
                "high_fever", "severe_chest_pain", "confusion", 
                "rapid_breathing", "low_oxygen"
            ],
            "lay_terms": [
                "coughing up stuff", "chest hurts", "can't breathe properly",
                "really tired", "burning chest", "lung infection"
            ],
            "description": "Infection causing inflammation of lung tissue"
        },
        "J44.9": {
            "name": "COPD",
            "symptoms": [
                "chronic_cough", "dyspnea", "wheezing", "sputum_production",
                "chest_tightness"
            ],
            "severity_indicators": [
                "severe_breathlessness", "cyanosis", "weight_loss",
                "edema", "barrel_chest"
            ],
            "lay_terms": [
                "can't breathe", "constant cough", "bringing up phlegm",
                "out of breath", "chest feels tight"
            ],
            "description": "Chronic obstructive pulmonary disease with airflow limitation"
        },
        "J06.9": {
            "name": "Upper respiratory infection",
            "symptoms": [
                "sore_throat", "runny_nose", "cough", "sneezing",
                "nasal_congestion", "mild_fever"
            ],
            "severity_indicators": [
                "high_fever", "severe_throat_pain", "difficulty_swallowing",
                "persistent_cough"
            ],
            "lay_terms": [
                "stuffy nose", "scratchy throat", "head cold",
                "sniffles", "throat hurts"
            ],
            "description": "Common cold or upper respiratory tract infection"
        },
        "J20.9": {
            "name": "Acute bronchitis",
            "symptoms": [
                "cough", "sputum_production", "chest_discomfort",
                "mild_dyspnea", "wheezing"
            ],
            "severity_indicators": [
                "persistent_cough", "blood_in_sputum", "fever",
                "severe_chest_pain"
            ],
            "lay_terms": [
                "chest cold", "hacking cough", "coughing up mucus",
                "chest feels heavy", "can't stop coughing"
            ],
            "description": "Acute inflammation of bronchial tubes"
        },
        "J81.0": {
            "name": "Acute pulmonary edema",
            "symptoms": [
                "severe_dyspnea", "orthopnea", "cough", "wheezing",
                "anxiety", "pink_frothy_sputum"
            ],
            "severity_indicators": [
                "extreme_breathlessness", "cyanosis", "confusion",
                "diaphoresis", "tachycardia"
            ],
            "lay_terms": [
                "can't breathe lying down", "drowning feeling",
                "gasping for air", "fluid in lungs", "suffocating"
            ],
            "description": "Fluid accumulation in lung tissue"
        },
        "J93.0": {
            "name": "Spontaneous tension pneumothorax",
            "symptoms": [
                "sudden_chest_pain", "dyspnea", "tachycardia",
                "anxiety", "decreased_breath_sounds"
            ],
            "severity_indicators": [
                "severe_chest_pain", "extreme_breathlessness", "hypotension",
                "cyanosis", "respiratory_distress"
            ],
            "lay_terms": [
                "collapsed lung", "sharp chest pain", "sudden breathing trouble",
                "stabbing pain", "can't take deep breath"
            ],
            "description": "Air in pleural space causing lung collapse"
        },
        "J15.9": {
            "name": "Bacterial pneumonia",
            "symptoms": [
                "productive_cough", "fever", "chest_pain", "dyspnea",
                "chills", "fatigue"
            ],
            "severity_indicators": [
                "high_fever", "confusion", "severe_chest_pain",
                "rapid_breathing", "hypoxia"
            ],
            "lay_terms": [
                "lung infection", "coughing up colored mucus",
                "chest really hurts", "can't breathe right", "really sick"
            ],
            "description": "Bacterial infection of lung parenchyma"
        },
        "J12.9": {
            "name": "Viral pneumonia",
            "symptoms": [
                "dry_cough", "fever", "dyspnea", "myalgia",
                "headache", "fatigue"
            ],
            "severity_indicators": [
                "high_fever", "severe_dyspnea", "confusion",
                "rapid_breathing"
            ],
            "lay_terms": [
                "dry cough", "viral lung infection", "aching all over",
                "can't breathe well", "wiped out"
            ],
            "description": "Viral infection of lung tissue"
        },
        "J21.9": {
            "name": "Acute bronchiolitis",
            "symptoms": [
                "wheezing", "cough", "dyspnea", "tachypnea",
                "nasal_flaring", "retractions"
            ],
            "severity_indicators": [
                "severe_respiratory_distress", "cyanosis", "apnea",
                "poor_feeding", "lethargy"
            ],
            "lay_terms": [
                "baby can't breathe", "wheezing badly", "working hard to breathe",
                "breathing really fast", "chest pulling in"
            ],
            "description": "Inflammation of small airways, common in infants"
        }
    }
    
    @classmethod
    def get_all_conditions(cls) -> Dict[str, Dict]:
        """
        Get all respiratory conditions.
        
        Returns:
            Dictionary mapping ICD-10 codes to condition data
        """
        return cls._CONDITIONS.copy()
    
    @classmethod
    def get_condition_by_code(cls, code: str) -> Dict:
        """
        Get a specific condition by ICD-10 code.
        
        Args:
            code: ICD-10 code (e.g., 'J45.9')
            
        Returns:
            Condition data dictionary
            
        Raises:
            KeyError: If code not found
        """
        if code not in cls._CONDITIONS:
            raise KeyError(f"Condition code '{code}' not found")
        return cls._CONDITIONS[code].copy()
    
    @classmethod
    def get_random_condition(cls) -> Tuple[str, Dict]:
        """
        Get a random respiratory condition.
        
        Returns:
            Tuple of (code, condition_data)
        """
        code = random.choice(list(cls._CONDITIONS.keys()))
        return code, cls._CONDITIONS[code].copy()
    
    @classmethod
    def get_symptoms_for_condition(cls, code: str) -> List[str]:
        """
        Get all symptoms for a condition.
        
        Args:
            code: ICD-10 code
            
        Returns:
            List of symptoms
        """
        condition = cls.get_condition_by_code(code)
        return condition["symptoms"] + condition["severity_indicators"]
    
    @classmethod
    def get_lay_terms_for_condition(cls, code: str) -> List[str]:
        """
        Get lay language terms for a condition.
        
        Args:
            code: ICD-10 code
            
        Returns:
            List of lay terms
        """
        condition = cls.get_condition_by_code(code)
        return condition["lay_terms"]
    
    @classmethod
    def get_condition_names(cls) -> List[str]:
        """
        Get list of all condition names.
        
        Returns:
            List of condition names
        """
        return [data["name"] for data in cls._CONDITIONS.values()]
    
    @classmethod
    def search_by_symptom(cls, symptom: str) -> List[Tuple[str, Dict]]:
        """
        Find conditions that have a specific symptom.
        
        Args:
            symptom: Symptom to search for
            
        Returns:
            List of (code, condition_data) tuples
        """
        results = []
        for code, data in cls._CONDITIONS.items():
            all_symptoms = data["symptoms"] + data["severity_indicators"]
            if symptom in all_symptoms:
                results.append((code, data.copy()))
        return results

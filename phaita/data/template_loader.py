"""Template loader and manager for patient complaint generation."""

import random
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque


class TemplateManager:
    """Manages complaint templates with intelligent selection based on context."""
    
    def __init__(self, template_file: Optional[Path] = None):
        """Initialize template manager.
        
        Args:
            template_file: Path to YAML template file. If None, uses default location.
        """
        if template_file is None:
            # Try new config/templates.yaml location first
            project_root = Path(__file__).resolve().parents[2]
            config_template = project_root / "config" / "templates.yaml"
            legacy_template = Path(__file__).parent / "templates.yaml"
            
            if config_template.exists():
                template_file = config_template
            elif legacy_template.exists():
                template_file = legacy_template
            else:
                # Default to config location (new structure)
                template_file = config_template
        
        self.template_file = template_file
        self.templates: List[Dict] = []
        self.placeholders: Dict[str, List[str]] = {}
        self.recent_templates: deque = deque(maxlen=5)  # Track last 5 used templates
        
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load templates from YAML file."""
        try:
            with open(self.template_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            self.templates = data.get('templates', [])
            self.placeholders = data.get('placeholders', {})
            
            if not self.templates:
                raise ValueError("No templates found in template file")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Template file not found: {self.template_file}. "
                "Expected templates.yaml in config/ or phaita/data/ directory."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in template file: {e}")
    
    def select_template(
        self,
        age: int,
        severity: str,
        num_symptoms: int,
        avoid_recent: bool = True
    ) -> Dict:
        """Select appropriate template based on patient characteristics.
        
        Args:
            age: Patient age
            severity: Symptom severity (mild, moderate, severe)
            num_symptoms: Number of symptoms (for compound vs simple templates)
            avoid_recent: Whether to avoid recently used templates
        
        Returns:
            Selected template dictionary
        """
        # Filter templates by age appropriateness
        age_appropriate = [
            t for t in self.templates
            if t['age_appropriateness'][0] <= age <= t['age_appropriateness'][1]
        ]
        
        if not age_appropriate:
            age_appropriate = self.templates
        
        # Filter by severity match
        severity_matched = [
            t for t in age_appropriate
            if severity in t['severity_match']
        ]
        
        if not severity_matched:
            severity_matched = age_appropriate
        
        # Filter out recently used templates if requested
        if avoid_recent and self.recent_templates:
            available = [
                t for t in severity_matched
                if t['id'] not in self.recent_templates
            ]
            if available:
                severity_matched = available
        
        # Prefer simpler templates for single symptoms, compound for multiple
        if num_symptoms >= 2:
            # Boost compound templates
            candidates = severity_matched
        else:
            # Prefer non-compound templates for single symptom
            non_compound = [
                t for t in severity_matched
                if 'compound' not in t['id']
            ]
            candidates = non_compound if non_compound else severity_matched
        
        # Weight-based selection
        weights = [t['weight'] for t in candidates]
        selected = random.choices(candidates, weights=weights, k=1)[0]
        
        # Track usage
        self.recent_templates.append(selected['id'])
        
        return selected
    
    def fill_template(
        self,
        template: Dict,
        symptoms: List[str],
        demographics_summary: Optional[str] = None,
        trigger: Optional[str] = None
    ) -> str:
        """Fill template with actual values.
        
        Args:
            template: Template dictionary with pattern and placeholders
            symptoms: List of symptom phrases
            demographics_summary: Optional patient demographics string
            trigger: Optional trigger/event string
        
        Returns:
            Filled complaint string
        """
        pattern = template['pattern']
        required = template['placeholders']
        
        # Build replacement dictionary
        replacements = {}
        
        # Handle symptoms
        if 'symptoms' in required:
            replacements['symptoms'] = self._format_symptoms_list(symptoms)
        
        if 'symptom' in required:
            replacements['symptom'] = symptoms[0] if symptoms else "discomfort"
        
        if 'symptom1' in required:
            replacements['symptom1'] = symptoms[0] if len(symptoms) > 0 else "discomfort"
        
        if 'symptom2' in required:
            replacements['symptom2'] = symptoms[1] if len(symptoms) > 1 else "fatigue"
        
        # Handle other placeholders
        if 'duration' in required:
            replacements['duration'] = random.choice(self.placeholders.get('duration', ['a while']))
        
        if 'severity' in required:
            replacements['severity'] = random.choice(self.placeholders.get('severity', ['bad']))
        
        if 'emotion' in required:
            replacements['emotion'] = random.choice(self.placeholders.get('emotion', ['worried']))
        
        if 'action' in required:
            replacements['action'] = random.choice(self.placeholders.get('action', ['move']))
        
        if 'activity' in required:
            replacements['activity'] = random.choice(self.placeholders.get('activity', ['activity']))
        
        if 'trigger' in required:
            if trigger:
                replacements['trigger'] = trigger
            else:
                replacements['trigger'] = random.choice([
                    'physical exertion', 'stress', 'exposure to cold',
                    'lying down', 'eating'
                ])
        
        if 'demographics' in required:
            replacements['demographics'] = demographics_summary or "Patient"
        
        # Perform replacements
        filled = pattern
        for key, value in replacements.items():
            filled = filled.replace(f'{{{key}}}', value)
        
        return filled
    
    def _format_symptoms_list(self, symptoms: List[str]) -> str:
        """Format a list of symptoms into natural language.
        
        Args:
            symptoms: List of symptom phrases
        
        Returns:
            Natural language formatted symptom list
        """
        if not symptoms:
            return "general discomfort"
        elif len(symptoms) == 1:
            return symptoms[0]
        elif len(symptoms) == 2:
            return f"{symptoms[0]} and {symptoms[1]}"
        else:
            # Oxford comma for 3+ items
            return ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"
    
    def generate_complaint(
        self,
        symptoms: List[str],
        age: int = 40,
        severity: str = 'moderate',
        demographics_summary: Optional[str] = None,
        trigger: Optional[str] = None
    ) -> str:
        """Generate a complete complaint using templates.
        
        Args:
            symptoms: List of symptom phrases
            age: Patient age
            severity: Symptom severity
            demographics_summary: Optional demographics string
            trigger: Optional trigger/event
        
        Returns:
            Generated complaint string
        """
        if not symptoms:
            symptoms = ["discomfort"]
        
        template = self.select_template(
            age=age,
            severity=severity,
            num_symptoms=len(symptoms)
        )
        
        complaint = self.fill_template(
            template=template,
            symptoms=symptoms,
            demographics_summary=demographics_summary,
            trigger=trigger
        )
        
        return complaint
    
    def get_template_statistics(self) -> Dict:
        """Get statistics about available templates.
        
        Returns:
            Dictionary with template statistics
        """
        return {
            'total_templates': len(self.templates),
            'formality_levels': list(set(t['formality_level'] for t in self.templates)),
            'severity_coverage': list(set(
                s for t in self.templates for s in t['severity_match']
            )),
            'compound_templates': len([t for t in self.templates if 'compound' in t['id']]),
            'simple_templates': len([t for t in self.templates if 'compound' not in t['id']]),
        }


__all__ = ['TemplateManager']

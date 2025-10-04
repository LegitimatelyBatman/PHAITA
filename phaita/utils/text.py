"""Text processing utilities for symptom normalization and formatting.

This module provides centralized functions for consistent symptom normalization
across the PHAITA application. All symptom processing should use these functions
to ensure consistent matching and comparison.
"""


def normalize_symptom(symptom: str) -> str:
    """Normalize symptom string to lowercase with spaces.
    
    This is the primary normalization function used for symptom matching,
    comparison, and probability lookups throughout the application.
    
    Converts: 'Severe_Respiratory-Distress' -> 'severe respiratory distress'
    
    Args:
        symptom: Raw symptom name with any formatting (underscores, hyphens, mixed case)
        
    Returns:
        Normalized symptom name (lowercase, spaces only, stripped)
        
    Examples:
        >>> normalize_symptom('Shortness_of_Breath')
        'shortness of breath'
        >>> normalize_symptom('chest-pain')
        'chest pain'
        >>> normalize_symptom('  FEVER  ')
        'fever'
    """
    # Replace separators with spaces, then normalize whitespace
    normalized = symptom.lower().replace('_', ' ').replace('-', ' ').strip()
    # Collapse multiple spaces into single space
    return ' '.join(normalized.split())


def normalize_symptom_to_underscores(symptom: str) -> str:
    """Normalize symptom string to lowercase with underscores.
    
    This format is used in specific contexts like info sheets and CLI output
    where underscores are preferred over spaces for structured data.
    
    Converts: 'Severe Respiratory Distress' -> 'severe_respiratory_distress'
    
    Args:
        symptom: Raw symptom name with any formatting
        
    Returns:
        Normalized symptom name (lowercase, underscores, stripped)
        
    Examples:
        >>> normalize_symptom_to_underscores('Shortness of Breath')
        'shortness_of_breath'
        >>> normalize_symptom_to_underscores('chest-pain')
        'chest_pain'
        >>> normalize_symptom_to_underscores('  FEVER  ')
        'fever'
    """
    # Replace separators with underscores, then normalize
    normalized = symptom.strip().lower().replace(' ', '_').replace('-', '_')
    # Collapse multiple underscores into single underscore
    return '_'.join(filter(None, normalized.split('_')))

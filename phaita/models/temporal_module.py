"""Temporal symptom modeling for tracking symptom progression over time."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Import torch only where needed for LSTM encoder
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class SymptomTimeline:
    """Tracks symptom onset times and maintains ordered progression.
    
    This class stores symptom events with their onset times (hours since first symptom)
    and provides methods to retrieve progression patterns for matching against
    canonical disease progressions.
    
    Attributes:
        events: List of (symptom, hours_since_onset) tuples
    """
    
    def __init__(self):
        """Initialize an empty symptom timeline."""
        self.events: List[Tuple[str, float]] = []
    
    def add_symptom(self, symptom: str, hours_ago: float) -> None:
        """Add a symptom to the timeline with its onset time.
        
        Args:
            symptom: Name of the symptom
            hours_ago: Hours since this symptom first appeared
        """
        self.events.append((symptom, hours_ago))
        # Sort by hours_ago in descending order (most recent first)
        self.events.sort(key=lambda x: x[1], reverse=True)
    
    def get_progression_pattern(self) -> List[Tuple[str, float]]:
        """Return the symptom progression pattern as an ordered sequence.
        
        Returns:
            List of (symptom, hours_since_onset) tuples, ordered chronologically
            (earliest symptom first)
        """
        # Return in chronological order (reverse of internal storage)
        return list(reversed(self.events))
    
    def get_symptom_order(self) -> List[str]:
        """Return just the symptom names in chronological order.
        
        Returns:
            List of symptom names, ordered from earliest to most recent
        """
        return [symptom for symptom, _ in reversed(self.events)]
    
    def clear(self) -> None:
        """Clear all events from the timeline."""
        self.events.clear()


class TemporalSymptomEncoder(nn.Module if TORCH_AVAILABLE else object):
    """LSTM-based encoder for symptom sequences with temporal information.
    
    This module encodes sequences of (symptom, timestamp) tuples into a fixed-size
    temporal embedding vector that captures symptom progression patterns.
    
    Args:
        symptom_vocab_size: Size of symptom vocabulary
        symptom_embedding_dim: Dimension of symptom embeddings (default: 64)
        hidden_dim: LSTM hidden dimension (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.1)
    
    Raises:
        ImportError: If torch is not available
    """
    
    def __init__(
        self,
        symptom_vocab_size: int,
        symptom_embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TemporalSymptomEncoder. "
                "Install with: pip install torch"
            )
        
        super().__init__()
        
        self.symptom_vocab_size = symptom_vocab_size
        self.symptom_embedding_dim = symptom_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Symptom embeddings
        self.symptom_embedding = nn.Embedding(
            symptom_vocab_size, symptom_embedding_dim
        )
        
        # Time encoding (1D - just the timestamp)
        self.time_projection = nn.Linear(1, symptom_embedding_dim)
        
        # LSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=symptom_embedding_dim * 2,  # symptom + time
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        symptom_indices: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a symptom sequence with temporal information.
        
        Args:
            symptom_indices: Tensor of symptom indices [batch_size, seq_len]
            timestamps: Tensor of timestamps in hours [batch_size, seq_len]
            
        Returns:
            Temporal embedding vector [batch_size, hidden_dim]
        """
        batch_size, seq_len = symptom_indices.shape
        
        # Embed symptoms
        symptom_embeds = self.symptom_embedding(symptom_indices)  # [B, L, E]
        
        # Project timestamps
        time_embeds = self.time_projection(timestamps.unsqueeze(-1))  # [B, L, E]
        
        # Concatenate symptom and time embeddings
        combined_embeds = torch.cat([symptom_embeds, time_embeds], dim=-1)  # [B, L, 2E]
        
        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(combined_embeds)  # hidden: [num_layers, B, H]
        
        # Use final hidden state from last layer
        final_hidden = hidden[-1]  # [B, H]
        
        # Project to output space
        output = self.output_projection(final_hidden)  # [B, H]
        output = self.dropout(output)
        
        return output


class TemporalPatternMatcher:
    """Matches patient symptom timelines against canonical condition progressions.
    
    This class loads typical symptom progression patterns for each condition and
    scores how well a patient's timeline matches each pattern. Higher scores
    indicate better temporal alignment with the condition's typical progression.
    
    Args:
        temporal_patterns: Dictionary mapping condition codes to progression patterns
    """
    
    def __init__(self, temporal_patterns: Optional[Dict[str, Dict]] = None):
        """Initialize the temporal pattern matcher.
        
        Args:
            temporal_patterns: Dict with condition codes as keys, each containing
                              a 'typical_progression' list of symptom/onset_hour dicts
        """
        self.temporal_patterns = temporal_patterns or {}
    
    def score_timeline(
        self,
        timeline: SymptomTimeline,
        condition_code: str,
    ) -> float:
        """Score how well a patient timeline matches a condition's typical progression.
        
        Args:
            timeline: Patient's symptom timeline
            condition_code: ICD-10 condition code to match against
            
        Returns:
            Score between 0.5 and 1.5 (1.0 = neutral, >1.0 = good match, <1.0 = poor match)
        """
        if condition_code not in self.temporal_patterns:
            # No pattern defined, return neutral score
            return 1.0
        
        pattern = self.temporal_patterns[condition_code].get("typical_progression", [])
        if not pattern:
            return 1.0
        
        patient_progression = timeline.get_progression_pattern()
        if not patient_progression:
            return 1.0
        
        # Build lookup for patient symptoms and their onset times
        patient_symptoms = {symptom: hours for symptom, hours in patient_progression}
        
        # Calculate alignment scores
        alignment_scores = []
        
        for i, pattern_event in enumerate(pattern):
            pattern_symptom = pattern_event["symptom"]
            expected_onset = pattern_event["onset_hour"]
            
            if pattern_symptom in patient_symptoms:
                actual_onset = patient_symptoms[pattern_symptom]
                
                # Calculate temporal difference (in hours)
                time_diff = abs(actual_onset - expected_onset)
                
                # Score based on time difference (exponential decay)
                # Perfect match (0 hours diff) = 1.0
                # 24 hours diff = ~0.6
                # 48 hours diff = ~0.4
                temporal_score = 2.0 ** (-time_diff / 24.0)
                alignment_scores.append(temporal_score)
        
        if not alignment_scores:
            # No matching symptoms, slight penalty
            return 0.8
        
        # Average alignment score
        avg_alignment = sum(alignment_scores) / len(alignment_scores)
        
        # Bonus for having symptoms in correct order
        order_bonus = self._calculate_order_bonus(timeline, pattern)
        
        # Combine scores: base score + order bonus
        # Scale to range [0.5, 1.5] with 1.0 as neutral
        base_score = 0.7 + (avg_alignment * 0.5)  # Range [0.7, 1.2]
        final_score = base_score + (order_bonus * 0.3)  # Add up to 0.3 for order
        
        # Clamp to reasonable range
        return max(0.5, min(1.5, final_score))
    
    def _calculate_order_bonus(
        self,
        timeline: SymptomTimeline,
        pattern: List[Dict],
    ) -> float:
        """Calculate bonus for symptoms appearing in correct chronological order.
        
        Args:
            timeline: Patient's symptom timeline
            pattern: Expected progression pattern
            
        Returns:
            Bonus score between 0.0 and 1.0
        """
        patient_order = timeline.get_symptom_order()
        pattern_order = [event["symptom"] for event in pattern]
        
        # Find symptoms that appear in both
        common_symptoms = set(patient_order) & set(pattern_order)
        
        if len(common_symptoms) < 2:
            # Need at least 2 symptoms to assess order
            return 0.0
        
        # Check if common symptoms appear in same relative order
        patient_indices = {s: i for i, s in enumerate(patient_order) if s in common_symptoms}
        pattern_indices = {s: i for i, s in enumerate(pattern_order) if s in common_symptoms}
        
        # Count pairs that are in correct order
        correct_pairs = 0
        total_pairs = 0
        
        for s1 in common_symptoms:
            for s2 in common_symptoms:
                if s1 != s2:
                    total_pairs += 1
                    # Check if relative order matches
                    patient_s1_before_s2 = patient_indices[s1] < patient_indices[s2]
                    pattern_s1_before_s2 = pattern_indices[s1] < pattern_indices[s2]
                    
                    if patient_s1_before_s2 == pattern_s1_before_s2:
                        correct_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return correct_pairs / total_pairs

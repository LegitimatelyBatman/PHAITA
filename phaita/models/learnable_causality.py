"""
Learnable symptom causality module.

Makes symptom causality weights learnable through PyTorch parameters
instead of fixed weights from YAML configuration.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LearnableSymptomCausality(nn.Module):
    """
    Learnable symptom causality with PyTorch parameters.
    
    Initializes causal and temporal edge weights from YAML config but makes
    them learnable through gradient descent during training.
    """
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize learnable symptom causality.
        
        Args:
            config_path: Path to symptom_causality.yaml (optional)
            device: PyTorch device ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load initial values from YAML
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "symptom_causality.yaml"
        
        self.config_data = self._load_config(config_path)
        
        # Parse causal and temporal edges
        self.causal_edges_config = self.config_data.get('causal_edges', [])
        self.temporal_edges_config = self.config_data.get('temporal_edges', [])
        
        # Build edge lists and initial weights
        self.causal_edge_pairs = []  # List of (source, target) tuples
        self.temporal_edge_pairs = []  # List of (earlier, later) tuples
        
        causal_initial_weights = []
        for edge in self.causal_edges_config:
            source = edge.get('source')
            target = edge.get('target')
            strength = edge.get('strength', 0.8)
            if source and target:
                self.causal_edge_pairs.append((source, target))
                # Store as logit for unconstrained optimization
                causal_initial_weights.append(self._to_logit(strength))
        
        temporal_initial_weights = []
        temporal_delays = []
        for edge in self.temporal_edges_config:
            earlier = edge.get('earlier')
            later = edge.get('later')
            strength = edge.get('strength', 0.7)
            delay_hours = edge.get('typical_delay_hours', 24)
            if earlier and later:
                self.temporal_edge_pairs.append((earlier, later))
                temporal_initial_weights.append(self._to_logit(strength))
                # Normalize delay to [0, 1] range (max 168 hours = 1 week)
                temporal_delays.append(min(delay_hours / 168.0, 1.0))
        
        # Create learnable parameters
        if causal_initial_weights:
            self.causal_weights = nn.Parameter(
                torch.tensor(causal_initial_weights, dtype=torch.float32, device=self.device)
            )
        else:
            # Empty tensor for no causal edges
            self.causal_weights = nn.Parameter(torch.zeros(0, dtype=torch.float32, device=self.device))
        
        if temporal_initial_weights:
            self.temporal_weights = nn.Parameter(
                torch.tensor(temporal_initial_weights, dtype=torch.float32, device=self.device)
            )
            # Temporal delays are fixed (not learned) but stored as buffer
            self.register_buffer('temporal_delays', 
                               torch.tensor(temporal_delays, dtype=torch.float32, device=self.device))
        else:
            self.temporal_weights = nn.Parameter(torch.zeros(0, dtype=torch.float32, device=self.device))
            self.register_buffer('temporal_delays', torch.zeros(0, dtype=torch.float32, device=self.device))
        
        # Store edge type configuration
        self.edge_types = self.config_data.get('edge_types', {
            'co_occurrence': 0,
            'causal': 1,
            'temporal': 2
        })
        
        # Edge weight configuration (these could also be made learnable)
        self.edge_weights_config = self.config_data.get('edge_weights', {
            'co_occurrence_weight': 0.4,
            'causal_weight': 1.0,
            'temporal_weight': 0.6,
            'reverse_causal_factor': 0.3
        })
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load symptom causality configuration from YAML."""
        default_config = {
            'causal_edges': [],
            'temporal_edges': [],
            'edge_types': {
                'co_occurrence': 0,
                'causal': 1,
                'temporal': 2
            },
            'edge_weights': {
                'co_occurrence_weight': 0.4,
                'causal_weight': 1.0,
                'temporal_weight': 0.6,
                'reverse_causal_factor': 0.3
            }
        }
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception:
            pass
        
        return default_config
    
    def _to_logit(self, prob: float) -> float:
        """Convert probability to logit for unconstrained optimization."""
        prob = max(min(prob, 0.99), 0.01)  # Clamp to avoid inf
        return torch.logit(torch.tensor(prob)).item()
    
    def get_causal_edges(self) -> List[Tuple[str, str, float]]:
        """
        Get causal edges with learned strengths.
        
        Returns:
            List of (source, target, strength) tuples
        """
        edges = []
        strengths = torch.sigmoid(self.causal_weights).cpu().detach().numpy()
        
        for i, (source, target) in enumerate(self.causal_edge_pairs):
            edges.append((source, target, float(strengths[i])))
        
        return edges
    
    def get_temporal_edges(self) -> List[Tuple[str, str, float, float]]:
        """
        Get temporal edges with learned strengths and fixed delays.
        
        Returns:
            List of (earlier, later, strength, normalized_delay) tuples
        """
        edges = []
        strengths = torch.sigmoid(self.temporal_weights).cpu().detach().numpy()
        delays = self.temporal_delays.cpu().numpy()
        
        for i, (earlier, later) in enumerate(self.temporal_edge_pairs):
            edges.append((earlier, later, float(strengths[i]), float(delays[i])))
        
        return edges
    
    def get_config_for_gnn(self) -> Dict:
        """
        Get configuration dictionary compatible with SymptomGraphBuilder.
        
        Returns:
            Dictionary with causal_edges, temporal_edges, edge_types, edge_weights
        """
        # Convert learned weights back to config format
        causal_edges = []
        for source, target, strength in self.get_causal_edges():
            causal_edges.append({
                'source': source,
                'target': target,
                'strength': strength,
                'evidence': 'Learned from data'
            })
        
        temporal_edges = []
        for earlier, later, strength, delay in self.get_temporal_edges():
            temporal_edges.append({
                'earlier': earlier,
                'later': later,
                'strength': strength,
                'typical_delay_hours': delay * 168.0,  # Denormalize
                'evidence': 'Learned from data'
            })
        
        return {
            'causal_edges': causal_edges,
            'temporal_edges': temporal_edges,
            'edge_types': self.edge_types,
            'edge_weights': self.edge_weights_config
        }
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns learned weights for gradient computation.
        
        Returns:
            Tuple of (causal_strengths, temporal_strengths) with sigmoid applied
        """
        causal_strengths = torch.sigmoid(self.causal_weights) if len(self.causal_weights) > 0 else self.causal_weights
        temporal_strengths = torch.sigmoid(self.temporal_weights) if len(self.temporal_weights) > 0 else self.temporal_weights
        return causal_strengths, temporal_strengths


def create_learnable_causality(config_path: Optional[str] = None, device: Optional[str] = None) -> LearnableSymptomCausality:
    """
    Factory function to create a learnable symptom causality module.
    
    Args:
        config_path: Optional path to symptom_causality.yaml
        device: Optional device ('cpu' or 'cuda')
        
    Returns:
        Initialized LearnableSymptomCausality module
    """
    return LearnableSymptomCausality(config_path=config_path, device=device)

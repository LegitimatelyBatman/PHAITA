"""
Graph Neural Network module for symptom relationship modeling.
Uses Graph Attention Networks (GAT) to model relationships between symptoms.
Requires torch-geometric to be properly installed.

Optimizations:
- torch.compile() for PyTorch 2.0+ JIT compilation speedup
- Cached edge_index/edge_weight as registered buffers
- Vectorized operations to minimize Python overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import networkx as nx
import sys
import yaml
from pathlib import Path

# Enforce required dependencies
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
except ImportError as e:
    raise ImportError(
        "torch-geometric is required for GNN modules. "
        "Install with: pip install torch-geometric==2.6.1\n"
        "Note: torch-geometric requires torch==2.5.1"
    ) from e

# Try to import learnable causality module (optional)
try:
    from .learnable_causality import LearnableSymptomCausality
    LEARNABLE_CAUSALITY_AVAILABLE = True
except ImportError:
    LEARNABLE_CAUSALITY_AVAILABLE = False
    LearnableSymptomCausality = None


class SymptomGraphBuilder:
    """
    Builds symptom relationship graphs from ICD-10 conditions.
    Supports causal and temporal edges in addition to co-occurrence.
    """
    
    def __init__(self, conditions: Dict, causality_config_path: Optional[str] = None, 
                 learnable_causality: Optional['LearnableSymptomCausality'] = None):
        """
        Initialize graph builder.
        
        Args:
            conditions: Dictionary of ICD-10 conditions with symptoms
            causality_config_path: Optional path to causality config YAML
            learnable_causality: Optional learnable causality module (overrides config_path)
        """
        self.conditions = conditions
        self.symptom_to_idx = {}
        self.idx_to_symptom = {}
        self._build_symptom_vocabulary()
        
        # Setup causality (learnable or fixed)
        self.learnable_causality = learnable_causality
        self.causality_config = None
        
        if learnable_causality is not None:
            # Use learnable causality - get config from module
            self.causality_config = learnable_causality.get_config_for_gnn()
        elif causality_config_path:
            self._load_causality_config(causality_config_path)
        else:
            # Try to load from default location - check new structure first
            project_root = Path(__file__).resolve().parents[2]
            medical_knowledge_path = project_root / "config" / "medical_knowledge.yaml"
            legacy_path = project_root / "config" / "symptom_causality.yaml"
            
            if medical_knowledge_path.exists():
                self._load_causality_config(str(medical_knowledge_path))
            elif legacy_path.exists():
                self._load_causality_config(str(legacy_path))
    
    def _load_causality_config(self, config_path: str):
        """Load causality configuration from YAML file.
        
        Supports both medical_knowledge.yaml and legacy symptom_causality.yaml.
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Handle new medical_knowledge.yaml structure
            if 'symptom_causality' in config_data:
                config_data = config_data['symptom_causality']
            
            self.causality_config = config_data
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load causality config from {config_path}: {e}", RuntimeWarning)
            self.causality_config = None
        
    def _build_symptom_vocabulary(self):
        """Build a vocabulary of all unique symptoms."""
        all_symptoms = set()
        for code, data in self.conditions.items():
            all_symptoms.update(data["symptoms"])
            all_symptoms.update(data["severity_indicators"])
        
        for idx, symptom in enumerate(sorted(all_symptoms)):
            self.symptom_to_idx[symptom] = idx
            self.idx_to_symptom[idx] = symptom
    
    def build_cooccurrence_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph based on symptom co-occurrence in conditions.
        
        Returns:
            edge_index: [2, num_edges] tensor of edge connections
            edge_weight: [num_edges] tensor of edge weights
        """
        num_symptoms = len(self.symptom_to_idx)
        cooccurrence_matrix = torch.zeros(num_symptoms, num_symptoms)
        
        # Count co-occurrences
        for code, data in self.conditions.items():
            symptoms = data["symptoms"] + data["severity_indicators"]
            symptom_indices = [self.symptom_to_idx[s] for s in symptoms if s in self.symptom_to_idx]
            
            # Create edges between co-occurring symptoms
            for i in symptom_indices:
                for j in symptom_indices:
                    if i != j:
                        cooccurrence_matrix[i, j] += 1
        
        # Convert to edge_index format
        edge_index = []
        edge_weight = []
        
        for i in range(num_symptoms):
            for j in range(num_symptoms):
                if cooccurrence_matrix[i, j] > 0:
                    edge_index.append([i, j])
                    # Normalize weights
                    edge_weight.append(cooccurrence_matrix[i, j])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            # Normalize weights
            edge_weight = edge_weight / edge_weight.max()
        else:
            # Empty graph fallback
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        
        return edge_index, edge_weight
    
    def get_num_nodes(self) -> int:
        """Get number of nodes in the graph."""
        return len(self.symptom_to_idx)
    
    def build_causal_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build graph with causal and temporal edges from causality config.
        
        Returns:
            edge_index: [2, num_edges] tensor of edge connections
            edge_weight: [num_edges] tensor of edge weights
            edge_attr: [num_edges, 3] tensor of edge features [type, strength, delay]
        """
        if not self.causality_config:
            # Fall back to co-occurrence graph with default edge attributes
            edge_index, edge_weight = self.build_cooccurrence_graph()
            # Create edge attributes: [edge_type=0 (co-occurrence), strength=weight, delay=0]
            num_edges = edge_index.shape[1]
            edge_attr = torch.zeros(num_edges, 3)
            edge_attr[:, 0] = 0  # co-occurrence type
            edge_attr[:, 1] = edge_weight  # strength from weight
            edge_attr[:, 2] = 0  # no delay
            return edge_index, edge_weight, edge_attr
        
        # Get edge type IDs from config
        edge_types = self.causality_config.get('edge_types', {
            'co_occurrence': 0,
            'causal': 1,
            'temporal': 2
        })
        
        # Get edge weights config
        edge_weights_config = self.causality_config.get('edge_weights', {
            'co_occurrence_weight': 0.4,
            'causal_weight': 1.0,
            'temporal_weight': 0.6,
            'reverse_causal_factor': 0.3
        })
        
        # Start with co-occurrence edges
        edge_list = []
        edge_weight_list = []
        edge_attr_list = []
        
        # Build co-occurrence edges
        cooccur_edge_index, cooccur_edge_weight = self.build_cooccurrence_graph()
        if cooccur_edge_index.shape[1] > 0:
            for i in range(cooccur_edge_index.shape[1]):
                src, tgt = cooccur_edge_index[0, i].item(), cooccur_edge_index[1, i].item()
                weight = cooccur_edge_weight[i].item() * edge_weights_config['co_occurrence_weight']
                edge_list.append([src, tgt])
                edge_weight_list.append(weight)
                edge_attr_list.append([edge_types['co_occurrence'], weight, 0.0])
        
        # Add causal edges
        causal_edges = self.causality_config.get('causal_edges', [])
        for edge in causal_edges:
            source = edge['source']
            target = edge['target']
            strength = edge['strength']
            
            # Check if symptoms exist in vocabulary
            if source not in self.symptom_to_idx or target not in self.symptom_to_idx:
                continue
            
            src_idx = self.symptom_to_idx[source]
            tgt_idx = self.symptom_to_idx[target]
            
            # Forward edge (source -> target)
            forward_weight = strength * edge_weights_config['causal_weight']
            edge_list.append([src_idx, tgt_idx])
            edge_weight_list.append(forward_weight)
            edge_attr_list.append([edge_types['causal'], strength, 0.0])
            
            # Reverse edge (target -> source) with reduced weight
            reverse_weight = strength * edge_weights_config['causal_weight'] * edge_weights_config['reverse_causal_factor']
            edge_list.append([tgt_idx, src_idx])
            edge_weight_list.append(reverse_weight)
            edge_attr_list.append([edge_types['causal'], strength * edge_weights_config['reverse_causal_factor'], 0.0])
        
        # Add temporal edges
        temporal_edges = self.causality_config.get('temporal_edges', [])
        for edge in temporal_edges:
            earlier = edge['earlier']
            later = edge['later']
            delay = edge.get('typical_delay_hours', 0)
            strength = edge.get('strength', 0.5)
            
            # Check if symptoms exist in vocabulary
            if earlier not in self.symptom_to_idx or later not in self.symptom_to_idx:
                continue
            
            earlier_idx = self.symptom_to_idx[earlier]
            later_idx = self.symptom_to_idx[later]
            
            # Temporal edge (earlier -> later)
            temporal_weight = strength * edge_weights_config['temporal_weight']
            edge_list.append([earlier_idx, later_idx])
            edge_weight_list.append(temporal_weight)
            # Normalize delay to [0, 1] range (assuming max 168 hours = 1 week)
            normalized_delay = min(delay / 168.0, 1.0)
            edge_attr_list.append([edge_types['temporal'], strength, normalized_delay])
        
        # Convert to tensors
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            # Empty graph fallback
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
        
        return edge_index, edge_weight, edge_attr


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network for symptom relationship modeling.
    Falls back to MLP if torch_geometric is not available.
    
    Optimizations:
    - torch.compile() for faster execution in PyTorch 2.0+
    - Cached computations where possible
    - Efficient batch processing
    """
    
    def __init__(
        self, 
        num_nodes: int,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None
    ):
        """
        Initialize GAT.
        
        Args:
            num_nodes: Number of nodes in graph
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden dimension for GAT layers
            output_dim: Output dimension
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
            edge_dim: Dimension of edge features (None to disable edge features)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        
        # Node embeddings (learnable features for each symptom)
        self.node_embeddings = nn.Embedding(num_nodes, node_feature_dim)
        
        # Use torch_geometric GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(node_feature_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim)
            )
        
        # Last layer (single head)
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout, edge_dim=edge_dim)
            )
        else:
            # Single layer case
            self.gat_layers.append(
                GATConv(node_feature_dim, output_dim, heads=1, dropout=dropout, edge_dim=edge_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Cache node IDs for reuse
        self.register_buffer('_node_ids_cache', torch.arange(num_nodes))
        
    def _forward_impl(
        self, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Internal forward pass implementation (can be compiled).
        
        Args:
            edge_index: [2, num_edges] edge connections
            edge_weight: [num_edges] optional edge weights
            batch_size: Batch size for output
            edge_attr: [num_edges, edge_dim] optional edge features
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Get node features using cached node IDs
        x = self.node_embeddings(self._node_ids_cache)
        
        # Apply GAT layers with vectorized operations
        for i, layer in enumerate(self.gat_layers):
            # Pass edge_attr if edge_dim is enabled
            if self.edge_dim is not None and edge_attr is not None:
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        # Global pooling to get graph-level embedding (mean across all nodes)
        # Use keepdim=True for efficient repeat operation
        graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Efficient batch expansion using expand (no memory copy)
        # Only use repeat if batch_size > 1 to avoid unnecessary operations
        if batch_size > 1:
            graph_embedding = graph_embedding.expand(batch_size, -1).contiguous()
        
        return graph_embedding
    
    def forward(
        self, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GAT.
        
        Args:
            edge_index: [2, num_edges] edge connections
            edge_weight: [num_edges] optional edge weights
            batch_size: Batch size for output
            edge_attr: [num_edges, edge_dim] optional edge features
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        return self._forward_impl(edge_index, edge_weight, batch_size, edge_attr)


class SymptomGraphModule(nn.Module):
    """
    Complete symptom graph module combining graph construction and GAT.
    
    Optimizations:
    - Edge index and weights cached as buffers (already on correct device)
    - torch.compile() for PyTorch 2.0+ performance boost
    - Efficient batch processing
    """
    
    def __init__(
        self,
        conditions: Dict,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_compile: bool = True,
        use_causal_edges: bool = True,
        causality_config_path: Optional[str] = None,
        learnable_causality: Optional['LearnableSymptomCausality'] = None
    ):
        """
        Initialize symptom graph module.
        
        Args:
            conditions: Dictionary of ICD-10 conditions
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
            use_compile: Whether to use torch.compile for optimization (PyTorch 2.0+)
            use_causal_edges: Whether to use causal edges (default True)
            causality_config_path: Optional path to causality config YAML
            learnable_causality: Optional learnable causality module (overrides config_path)
        """
        super().__init__()
        
        # Build graph structure
        self.graph_builder = SymptomGraphBuilder(
            conditions, 
            causality_config_path,
            learnable_causality=learnable_causality
        )
        self.use_causal_edges = use_causal_edges
        
        # Build graph with or without causal edges
        if use_causal_edges and self.graph_builder.causality_config:
            edge_index, edge_weight, edge_attr = self.graph_builder.build_causal_graph()
            edge_dim = 3  # [edge_type, strength, temporal_delay]
        else:
            edge_index, edge_weight = self.graph_builder.build_cooccurrence_graph()
            edge_attr = None
            edge_dim = None
        
        # Register as buffers (not parameters, but part of state)
        # This ensures they move with the model to GPU/CPU
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        if edge_attr is not None:
            self.register_buffer('edge_attr', edge_attr)
        else:
            self.register_buffer('edge_attr', torch.zeros((0, 3)))
        
        # Build GAT
        self.gat = GraphAttentionNetwork(
            num_nodes=self.graph_builder.get_num_nodes(),
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # Apply torch.compile if available and requested (PyTorch 2.0+)
        self._use_compile = use_compile
        self._compiled_gat = None
        
        if use_compile and hasattr(torch, 'compile'):
            try:
                # Compile the internal forward implementation for best performance
                # Use 'reduce-overhead' mode for faster execution
                self._compiled_gat = torch.compile(
                    self.gat._forward_impl,
                    mode='reduce-overhead',
                    fullgraph=False  # Allow graph breaks for flexibility
                )
            except Exception as e:
                # Compilation may fail on some platforms, fall back gracefully
                import warnings
                warnings.warn(
                    f"torch.compile failed, falling back to eager mode: {e}",
                    RuntimeWarning
                )
                self._compiled_gat = None
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get graph embeddings for batch.
        
        Args:
            batch_size: Number of samples in batch
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Use compiled version if available, otherwise use regular forward
        edge_attr = self.edge_attr if self.edge_attr.shape[0] > 0 else None
        if self._compiled_gat is not None:
            return self._compiled_gat(self.edge_index, self.edge_weight, batch_size, edge_attr)
        else:
            return self.gat(self.edge_index, self.edge_weight, batch_size, edge_attr)
    
    def get_symptom_indices(self, symptoms: List[str]) -> List[int]:
        """
        Get indices for a list of symptoms.
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            List of symptom indices
        """
        return [
            self.graph_builder.symptom_to_idx.get(s, -1) 
            for s in symptoms
        ]

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


class SymptomGraphBuilder:
    """
    Builds symptom relationship graphs from ICD-10 conditions.
    """
    
    def __init__(self, conditions: Dict):
        """
        Initialize graph builder.
        
        Args:
            conditions: Dictionary of ICD-10 conditions with symptoms
        """
        self.conditions = conditions
        self.symptom_to_idx = {}
        self.idx_to_symptom = {}
        self._build_symptom_vocabulary()
        
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
        dropout: float = 0.1
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
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node embeddings (learnable features for each symptom)
        self.node_embeddings = nn.Embedding(num_nodes, node_feature_dim)
        
        # Use torch_geometric GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(node_feature_dim, hidden_dim, heads=num_heads, dropout=dropout)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
        
        # Last layer (single head)
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout)
            )
        else:
            # Single layer case
            self.gat_layers.append(
                GATConv(node_feature_dim, output_dim, heads=1, dropout=dropout)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Cache node IDs for reuse
        self.register_buffer('_node_ids_cache', torch.arange(num_nodes))
        
    def _forward_impl(
        self, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor] = None,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Internal forward pass implementation (can be compiled).
        
        Args:
            edge_index: [2, num_edges] edge connections
            edge_weight: [num_edges] optional edge weights
            batch_size: Batch size for output
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Get node features using cached node IDs
        x = self.node_embeddings(self._node_ids_cache)
        
        # Apply GAT layers with vectorized operations
        for i, layer in enumerate(self.gat_layers):
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
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Forward pass through GAT.
        
        Args:
            edge_index: [2, num_edges] edge connections
            edge_weight: [num_edges] optional edge weights
            batch_size: Batch size for output
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        return self._forward_impl(edge_index, edge_weight, batch_size)


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
        use_compile: bool = True
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
        """
        super().__init__()
        
        # Build graph structure
        self.graph_builder = SymptomGraphBuilder(conditions)
        edge_index, edge_weight = self.graph_builder.build_cooccurrence_graph()
        
        # Register as buffers (not parameters, but part of state)
        # This ensures they move with the model to GPU/CPU
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        
        # Build GAT
        self.gat = GraphAttentionNetwork(
            num_nodes=self.graph_builder.get_num_nodes(),
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
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
        if self._compiled_gat is not None:
            return self._compiled_gat(self.edge_index, self.edge_weight, batch_size)
        else:
            return self.gat(self.edge_index, self.edge_weight, batch_size)
    
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

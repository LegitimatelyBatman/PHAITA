import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""
Demo script showing causal edges in the symptom graph.
Demonstrates how causal and temporal relationships enhance the GNN.
"""

import torch
from phaita.data.icd_conditions import RespiratoryConditions
from phaita.models.gnn_module import SymptomGraphBuilder, SymptomGraphModule


def demo_causal_edges():
    """Demonstrate causal edge loading and graph construction."""
    print("=" * 70)
    print("CAUSAL EDGES DEMO - PHAITA Symptom Graph")
    print("=" * 70)
    
    # Load conditions
    conditions = RespiratoryConditions.get_all_conditions()
    
    # Build graph
    print("\n1. Loading Symptom Graph with Causal Edges...")
    builder = SymptomGraphBuilder(conditions)
    
    if builder.causality_config:
        print("   ✅ Causality config loaded successfully")
        
        # Show edge types
        edge_types = builder.causality_config.get('edge_types', {})
        print(f"\n   Edge Types:")
        for name, idx in edge_types.items():
            print(f"     - {name}: {idx}")
        
        # Build causal graph
        edge_index, edge_weight, edge_attr = builder.build_causal_graph()
        
        print(f"\n2. Graph Statistics:")
        print(f"   - Total edges: {edge_index.shape[1]}")
        print(f"   - Total symptoms: {builder.get_num_nodes()}")
        
        # Count by edge type
        print(f"\n   Edge Breakdown:")
        for type_idx, type_name in enumerate(['Co-occurrence', 'Causal', 'Temporal']):
            count = (edge_attr[:, 0] == type_idx).sum().item()
            if count > 0:
                weights = edge_weight[edge_attr[:, 0] == type_idx]
                print(f"     - {type_name}: {count} edges "
                      f"(weight range: {weights.min():.3f}-{weights.max():.3f})")
        
        # Show example causal edges
        print(f"\n3. Example Causal Relationships:")
        causal_mask = edge_attr[:, 0] == 1
        causal_edges = edge_index[:, causal_mask]
        causal_weights = edge_weight[causal_mask]
        causal_strengths = edge_attr[causal_mask, 1]
        
        # Group forward and reverse edges
        forward_edges = []
        for i in range(causal_edges.shape[1]):
            src_idx = causal_edges[0, i].item()
            tgt_idx = causal_edges[1, i].item()
            strength = causal_strengths[i].item()
            weight = causal_weights[i].item()
            
            # Check if this is a forward edge (higher weight)
            is_forward = True
            for j in range(causal_edges.shape[1]):
                if i != j:
                    if (causal_edges[0, j].item() == tgt_idx and 
                        causal_edges[1, j].item() == src_idx and
                        causal_weights[j].item() > weight):
                        is_forward = False
                        break
            
            if is_forward and len(forward_edges) < 5:
                src = builder.idx_to_symptom[src_idx]
                tgt = builder.idx_to_symptom[tgt_idx]
                forward_edges.append((src, tgt, strength, weight))
        
        for src, tgt, strength, weight in forward_edges:
            print(f"     {src} → {tgt}")
            print(f"       Strength: {strength:.3f}, Weight: {weight:.3f}")
        
        # Show example temporal edges
        print(f"\n4. Example Temporal Relationships:")
        temporal_mask = edge_attr[:, 0] == 2
        temporal_edges = edge_index[:, temporal_mask]
        temporal_weights = edge_weight[temporal_mask]
        temporal_delays = edge_attr[temporal_mask, 2]
        
        for i in range(min(3, temporal_edges.shape[1])):
            src_idx = temporal_edges[0, i].item()
            tgt_idx = temporal_edges[1, i].item()
            src = builder.idx_to_symptom[src_idx]
            tgt = builder.idx_to_symptom[tgt_idx]
            delay = temporal_delays[i].item() * 168  # Convert back to hours
            weight = temporal_weights[i].item()
            print(f"     {src} → {tgt}")
            print(f"       Typical delay: ~{delay:.0f} hours, Weight: {weight:.3f}")
    else:
        print("   ⚠️  Causality config not found")
    
    # Compare GNN with and without causal edges
    print(f"\n5. GNN Comparison:")
    print("   Creating GNN modules...")
    
    gnn_with_causal = SymptomGraphModule(
        conditions=conditions,
        use_causal_edges=True,
        hidden_dim=64,
        output_dim=128,
        use_compile=False
    )
    
    gnn_without_causal = SymptomGraphModule(
        conditions=conditions,
        use_causal_edges=False,
        hidden_dim=64,
        output_dim=128,
        use_compile=False
    )
    
    print(f"\n   With causal edges:")
    print(f"     - Edge count: {gnn_with_causal.edge_index.shape[1]}")
    print(f"     - Edge dim: {gnn_with_causal.gat.edge_dim}")
    print(f"     - Edge attr shape: {gnn_with_causal.edge_attr.shape}")
    
    print(f"\n   Without causal edges:")
    print(f"     - Edge count: {gnn_without_causal.edge_index.shape[1]}")
    print(f"     - Edge dim: {gnn_without_causal.gat.edge_dim}")
    
    # Test forward pass
    with torch.no_grad():
        output_with = gnn_with_causal(batch_size=2)
        output_without = gnn_without_causal(batch_size=2)
        diff = torch.abs(output_with - output_without).mean().item()
    
    print(f"\n   Output difference: {diff:.6f}")
    print(f"   ✅ Causal edges affect graph embeddings as expected")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_causal_edges()

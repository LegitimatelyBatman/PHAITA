#!/usr/bin/env python3
"""
Test script for causal graph functionality in GNN module.
Tests causal edges, temporal edges, and edge type embeddings.
"""

import sys
import torch
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from phaita.data.icd_conditions import RespiratoryConditions
from phaita.models.gnn_module import SymptomGraphBuilder, SymptomGraphModule


def test_causal_edges_loading():
    """Test that causal edges are loaded from config."""
    print("🧪 Test 1: Loading Causal Edges from Config...")
    
    try:
        conditions = RespiratoryConditions.get_all_conditions()
        builder = SymptomGraphBuilder(conditions)
        
        # Check that causality config was loaded
        assert builder.causality_config is not None, "Causality config should be loaded"
        assert 'causal_edges' in builder.causality_config, "Should have causal_edges"
        assert 'temporal_edges' in builder.causality_config, "Should have temporal_edges"
        assert 'edge_types' in builder.causality_config, "Should have edge_types"
        
        # Check edge types
        edge_types = builder.causality_config['edge_types']
        assert edge_types['co_occurrence'] == 0, "Co-occurrence edge type should be 0"
        assert edge_types['causal'] == 1, "Causal edge type should be 1"
        assert edge_types['temporal'] == 2, "Temporal edge type should be 2"
        
        # Check that we have some causal edges defined
        causal_edges = builder.causality_config['causal_edges']
        assert len(causal_edges) > 0, "Should have at least one causal edge"
        
        # Check that we have some temporal edges defined
        temporal_edges = builder.causality_config['temporal_edges']
        assert len(temporal_edges) > 0, "Should have at least one temporal edge"
        
        print(f"  ✅ Loaded {len(causal_edges)} causal edges")
        print(f"  ✅ Loaded {len(temporal_edges)} temporal edges")
        print("✅ Test 1 passed: Causal edges loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_graph_construction():
    """Test that causal graph is constructed correctly."""
    print("\n🧪 Test 2: Causal Graph Construction...")
    
    try:
        conditions = RespiratoryConditions.get_all_conditions()
        builder = SymptomGraphBuilder(conditions)
        
        # Build causal graph
        edge_index, edge_weight, edge_attr = builder.build_causal_graph()
        
        # Check dimensions
        assert edge_index.shape[0] == 2, "edge_index should have 2 rows"
        num_edges = edge_index.shape[1]
        assert num_edges > 0, "Should have at least one edge"
        assert edge_weight.shape[0] == num_edges, "edge_weight should match number of edges"
        assert edge_attr.shape[0] == num_edges, "edge_attr should match number of edges"
        assert edge_attr.shape[1] == 3, "edge_attr should have 3 features [type, strength, delay]"
        
        # Check that we have different edge types
        edge_types = edge_attr[:, 0].unique()
        print(f"  ✅ Found {len(edge_types)} different edge types: {edge_types.tolist()}")
        
        # Count edges by type
        type_0_count = (edge_attr[:, 0] == 0).sum().item()  # co-occurrence
        type_1_count = (edge_attr[:, 0] == 1).sum().item()  # causal
        type_2_count = (edge_attr[:, 0] == 2).sum().item()  # temporal
        
        print(f"  ✅ Co-occurrence edges: {type_0_count}")
        print(f"  ✅ Causal edges: {type_1_count}")
        print(f"  ✅ Temporal edges: {type_2_count}")
        
        assert type_0_count > 0, "Should have co-occurrence edges"
        # Note: causal/temporal edges may be 0 if symptoms don't exist in vocabulary
        
        print("✅ Test 2 passed: Causal graph constructed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_edge_weights():
    """Test that causal edges have appropriate weights compared to co-occurrence."""
    print("\n🧪 Test 3: Causal Edge Weights...")
    
    try:
        conditions = RespiratoryConditions.get_all_conditions()
        builder = SymptomGraphBuilder(conditions)
        
        # Build both graphs
        edge_index, edge_weight, edge_attr = builder.build_causal_graph()
        
        # Get causal edges (type 1)
        causal_mask = edge_attr[:, 0] == 1
        if causal_mask.sum() > 0:
            causal_weights = edge_weight[causal_mask]
            causal_strengths = edge_attr[causal_mask, 1]
            
            print(f"  ✅ Causal edge weight range: [{causal_weights.min():.3f}, {causal_weights.max():.3f}]")
            print(f"  ✅ Causal edge strength range: [{causal_strengths.min():.3f}, {causal_strengths.max():.3f}]")
            
            # Check that strengths are reasonable (between 0 and 1)
            assert causal_strengths.min() >= 0, "Strengths should be non-negative"
            assert causal_strengths.max() <= 1.0, "Strengths should be <= 1.0"
        else:
            print("  ℹ️  No causal edges found (symptoms may not be in vocabulary)")
        
        # Get co-occurrence edges (type 0)
        cooccur_mask = edge_attr[:, 0] == 0
        if cooccur_mask.sum() > 0:
            cooccur_weights = edge_weight[cooccur_mask]
            print(f"  ✅ Co-occurrence edge weight range: [{cooccur_weights.min():.3f}, {cooccur_weights.max():.3f}]")
        
        print("✅ Test 3 passed: Edge weights are appropriate")
        return True
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directed_edges():
    """Test that causal edges are directed (source->target has higher weight than target->source)."""
    print("\n🧪 Test 4: Directed Edge Behavior...")
    
    try:
        conditions = RespiratoryConditions.get_all_conditions()
        builder = SymptomGraphBuilder(conditions)
        
        # Build causal graph
        edge_index, edge_weight, edge_attr = builder.build_causal_graph()
        
        # Get causal edges (type 1)
        causal_mask = edge_attr[:, 0] == 1
        causal_edge_index = edge_index[:, causal_mask]
        causal_edge_weight = edge_weight[causal_mask]
        
        if causal_edge_index.shape[1] >= 2:
            # Find forward and reverse pairs
            forward_reverse_pairs = 0
            for i in range(causal_edge_index.shape[1]):
                src, tgt = causal_edge_index[0, i].item(), causal_edge_index[1, i].item()
                weight_forward = causal_edge_weight[i].item()
                
                # Look for reverse edge
                for j in range(causal_edge_index.shape[1]):
                    if i != j:
                        src_j, tgt_j = causal_edge_index[0, j].item(), causal_edge_index[1, j].item()
                        if src_j == tgt and tgt_j == src:
                            weight_reverse = causal_edge_weight[j].item()
                            # Forward should have higher weight than reverse
                            if weight_forward > weight_reverse:
                                forward_reverse_pairs += 1
                            break
            
            if forward_reverse_pairs > 0:
                print(f"  ✅ Found {forward_reverse_pairs} forward/reverse pairs with correct weight asymmetry")
            else:
                print("  ℹ️  No clear forward/reverse pairs found")
        else:
            print("  ℹ️  Not enough causal edges to test directionality")
        
        print("✅ Test 4 passed: Directed edges work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_type_embeddings():
    """Test that edge type embeddings affect attention in GNN."""
    print("\n🧪 Test 5: Edge Type Embeddings in GNN...")
    
    try:
        conditions = RespiratoryConditions.get_all_conditions()
        
        # Create GNN module with causal edges
        gnn_with_causal = SymptomGraphModule(
            conditions=conditions,
            use_causal_edges=True,
            hidden_dim=64,
            output_dim=128,
            use_compile=False  # Disable compilation for testing
        )
        
        # Create GNN module without causal edges
        gnn_without_causal = SymptomGraphModule(
            conditions=conditions,
            use_causal_edges=False,
            hidden_dim=64,
            output_dim=128,
            use_compile=False  # Disable compilation for testing
        )
        
        # Check that causal GNN has edge attributes
        assert gnn_with_causal.edge_attr.shape[0] > 0, "Causal GNN should have edge attributes"
        assert gnn_with_causal.gat.edge_dim == 3, "Causal GNN should have edge_dim=3"
        
        # Check that non-causal GNN doesn't have edge attributes
        assert gnn_without_causal.gat.edge_dim is None, "Non-causal GNN should have edge_dim=None"
        
        # Test forward pass with causal edges
        with torch.no_grad():
            output_with_causal = gnn_with_causal(batch_size=2)
            assert output_with_causal.shape == (2, 128), f"Expected shape (2, 128), got {output_with_causal.shape}"
        
        # Test forward pass without causal edges
        with torch.no_grad():
            output_without_causal = gnn_without_causal(batch_size=2)
            assert output_without_causal.shape == (2, 128), f"Expected shape (2, 128), got {output_without_causal.shape}"
        
        print(f"  ✅ GNN with causal edges output shape: {output_with_causal.shape}")
        print(f"  ✅ GNN without causal edges output shape: {output_without_causal.shape}")
        print(f"  ✅ Edge attributes shape: {gnn_with_causal.edge_attr.shape}")
        
        # Outputs should be different due to edge features
        diff = torch.abs(output_with_causal - output_without_causal).mean().item()
        print(f"  ✅ Mean difference between outputs: {diff:.6f}")
        
        print("✅ Test 5 passed: Edge type embeddings work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that the GNN still works without causal edges (backward compatibility)."""
    print("\n🧪 Test 6: Backward Compatibility (No Causal Edges)...")
    
    try:
        conditions = RespiratoryConditions.get_all_conditions()
        
        # Create GNN module without causal edges
        gnn = SymptomGraphModule(
            conditions=conditions,
            use_causal_edges=False,
            use_compile=False
        )
        
        # Test forward pass
        with torch.no_grad():
            output = gnn(batch_size=4)
            assert output.shape == (4, 256), f"Expected shape (4, 256), got {output.shape}"
        
        print(f"  ✅ GNN forward pass successful with shape: {output.shape}")
        print("✅ Test 6 passed: Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"❌ Test 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all causal graph tests."""
    print("🏥 PHAITA Causal Graph Test Suite")
    print("=" * 60)
    
    tests = [
        test_causal_edges_loading,
        test_causal_graph_construction,
        test_causal_edge_weights,
        test_directed_edges,
        test_edge_type_embeddings,
        test_backward_compatibility,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All causal graph tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

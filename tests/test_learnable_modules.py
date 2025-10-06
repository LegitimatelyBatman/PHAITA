#!/usr/bin/env python3
"""
Test learnable comorbidity effects and symptom causality modules.
"""

import sys


def test_learnable_comorbidity():
    """Test learnable comorbidity effects module."""
    print("🧪 Testing Learnable Comorbidity Effects...")
    
    try:
        from phaita.models.learnable_comorbidity import LearnableComorbidityEffects
        import torch
        
        # Create learnable comorbidity module
        learnable_comorbidity = LearnableComorbidityEffects()
        
        # Check that it's a PyTorch module with parameters
        assert isinstance(learnable_comorbidity, torch.nn.Module), "Should be a PyTorch Module"
        assert learnable_comorbidity.comorbidity_weights.requires_grad, "Weights should be learnable"
        
        print(f"✅ Created learnable comorbidity module")
        print(f"   Comorbidities: {len(learnable_comorbidity.comorbidity_vocab)}")
        print(f"   Symptoms: {len(learnable_comorbidity.symptom_vocab)}")
        print(f"   Parameters shape: {learnable_comorbidity.comorbidity_weights.shape}")
        print(f"   Total parameters: {learnable_comorbidity.comorbidity_weights.numel()}")
        
        # Test getting modifiers
        diabetes_modifiers = learnable_comorbidity.get_symptom_modifiers("diabetes")
        assert isinstance(diabetes_modifiers, dict), "Should return dict"
        assert len(diabetes_modifiers) > 0, "Should have some modifiers"
        
        print(f"✅ Diabetes modifiers: {list(diabetes_modifiers.keys())[:3]}...")
        
        # Test getting full comorbidity data
        diabetes_data = learnable_comorbidity.get_comorbidity_data("diabetes")
        assert 'symptom_modifiers' in diabetes_data, "Should have symptom_modifiers"
        assert 'specific_symptoms' in diabetes_data, "Should have specific_symptoms"
        assert 'probability' in diabetes_data, "Should have probability"
        
        print(f"✅ Full comorbidity data structure correct")
        
        # Test gradient computation
        weights = learnable_comorbidity.forward()
        loss = weights.sum()
        loss.backward()
        assert learnable_comorbidity.comorbidity_weights.grad is not None, "Gradients should be computed"
        
        print(f"✅ Gradients computed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Learnable comorbidity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_learnable_causality():
    """Test learnable symptom causality module."""
    print("\n🧪 Testing Learnable Symptom Causality...")
    
    try:
        from phaita.models.learnable_causality import LearnableSymptomCausality
        import torch
        
        # Create learnable causality module
        learnable_causality = LearnableSymptomCausality()
        
        # Check that it's a PyTorch module with parameters
        assert isinstance(learnable_causality, torch.nn.Module), "Should be a PyTorch Module"
        
        print(f"✅ Created learnable causality module")
        print(f"   Causal edges: {len(learnable_causality.causal_edge_pairs)}")
        print(f"   Temporal edges: {len(learnable_causality.temporal_edge_pairs)}")
        
        if len(learnable_causality.causal_edge_pairs) > 0:
            assert learnable_causality.causal_weights.requires_grad, "Causal weights should be learnable"
            print(f"   Causal parameters: {learnable_causality.causal_weights.numel()}")
        
        if len(learnable_causality.temporal_edge_pairs) > 0:
            assert learnable_causality.temporal_weights.requires_grad, "Temporal weights should be learnable"
            print(f"   Temporal parameters: {learnable_causality.temporal_weights.numel()}")
        
        # Test getting edges
        causal_edges = learnable_causality.get_causal_edges()
        temporal_edges = learnable_causality.get_temporal_edges()
        
        print(f"✅ Retrieved {len(causal_edges)} causal edges")
        print(f"✅ Retrieved {len(temporal_edges)} temporal edges")
        
        if causal_edges:
            source, target, strength = causal_edges[0]
            print(f"   Example causal edge: {source} → {target} (strength: {strength:.3f})")
        
        if temporal_edges:
            earlier, later, strength, delay = temporal_edges[0]
            print(f"   Example temporal edge: {earlier} → {later} (strength: {strength:.3f}, delay: {delay:.3f})")
        
        # Test getting config for GNN
        config = learnable_causality.get_config_for_gnn()
        assert 'causal_edges' in config, "Config should have causal_edges"
        assert 'temporal_edges' in config, "Config should have temporal_edges"
        assert 'edge_types' in config, "Config should have edge_types"
        
        print(f"✅ GNN-compatible config generated")
        
        # Test gradient computation
        if len(learnable_causality.causal_weights) > 0:
            causal_strengths, temporal_strengths = learnable_causality.forward()
            loss = causal_strengths.sum()
            if len(temporal_strengths) > 0:
                loss = loss + temporal_strengths.sum()
            loss.backward()
            assert learnable_causality.causal_weights.grad is not None, "Gradients should be computed"
            print(f"✅ Gradients computed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Learnable causality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_bayesian_with_learnable():
    """Test EnhancedBayesianNetwork with learnable comorbidity."""
    print("\n🧪 Testing EnhancedBayesianNetwork with Learnable Comorbidity...")
    
    try:
        from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network
        
        # Create with learnable comorbidity
        network = create_enhanced_bayesian_network(use_learnable_comorbidity=True)
        
        assert network.use_learnable_comorbidity, "Should be using learnable comorbidity"
        assert network.learnable_comorbidity is not None, "Should have learnable module"
        
        print(f"✅ Created enhanced network with learnable comorbidity")
        
        # Test symptom sampling
        symptoms, metadata = network.sample_symptoms("J45.9", comorbidities=["diabetes"])
        assert len(symptoms) > 0, "Should generate symptoms"
        
        print(f"✅ Symptom sampling works with learnable comorbidity")
        print(f"   Generated {len(symptoms)} symptoms")
        
        # Test without learnable comorbidity (backward compatibility)
        network_fixed = create_enhanced_bayesian_network(use_learnable_comorbidity=False)
        assert not network_fixed.use_learnable_comorbidity, "Should not be using learnable comorbidity"
        
        symptoms_fixed, metadata_fixed = network_fixed.sample_symptoms("J45.9", comorbidities=["diabetes"])
        assert len(symptoms_fixed) > 0, "Should generate symptoms with fixed weights"
        
        print(f"✅ Fixed-weight mode still works (backward compatible)")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Bayesian with learnable test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_with_learnable():
    """Test GNN module with learnable causality."""
    print("\n🧪 Testing GNN with Learnable Causality...")
    
    try:
        from phaita.models.gnn_module import SymptomGraphModule
        from phaita.models.learnable_causality import LearnableSymptomCausality
        from phaita.data.icd_conditions import RespiratoryConditions
        
        conditions = RespiratoryConditions.get_all_conditions()
        
        # Create learnable causality module
        learnable_causality = LearnableSymptomCausality()
        
        # Create GNN with learnable causality
        gnn = SymptomGraphModule(
            conditions=conditions,
            use_causal_edges=True,
            learnable_causality=learnable_causality
        )
        
        print(f"✅ Created GNN with learnable causality")
        
        # Test forward pass
        output = gnn(batch_size=2)
        assert output.shape == (2, 256), f"Expected shape (2, 256), got {output.shape}"
        
        print(f"✅ Forward pass successful: {output.shape}")
        
        # Test backward compatibility (without learnable)
        gnn_fixed = SymptomGraphModule(
            conditions=conditions,
            use_causal_edges=True
        )
        
        output_fixed = gnn_fixed(batch_size=2)
        assert output_fixed.shape == (2, 256), f"Expected shape (2, 256), got {output_fixed.shape}"
        
        print(f"✅ Fixed-weight mode still works (backward compatible)")
        
        return True
        
    except Exception as e:
        print(f"❌ GNN with learnable test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all learnable module tests."""
    print("🧪 Learnable Modules Test Suite")
    print("=" * 60)
    
    results = []
    results.append(("Learnable Comorbidity", test_learnable_comorbidity()))
    results.append(("Learnable Causality", test_learnable_causality()))
    results.append(("Enhanced Bayesian + Learnable", test_enhanced_bayesian_with_learnable()))
    results.append(("GNN + Learnable", test_gnn_with_learnable()))
    
    print("=" * 60)
    print("📊 Test Results:")
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All learnable module tests passed!")
        return 0
    else:
        print("\n❌ Some learnable module tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

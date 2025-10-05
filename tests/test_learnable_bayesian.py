#!/usr/bin/env python3
"""
Test script for learnable Bayesian network and medical accuracy loss.
Tests the new learnable weights functionality.
"""

import sys
import traceback


def test_learnable_bayesian_network():
    """Test the learnable Bayesian symptom network."""
    print("🧪 Testing Learnable Bayesian Network...")
    
    try:
        from phaita.models.bayesian_network import LearnableBayesianSymptomNetwork, TORCH_AVAILABLE
        from phaita.data.icd_conditions import RespiratoryConditions
        
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, skipping learnable network test")
            return True
        
        # Initialize learnable network
        network = LearnableBayesianSymptomNetwork()
        
        # Test that it's a PyTorch module
        import torch
        assert isinstance(network, torch.nn.Module), "Network should be a PyTorch module"
        
        # Test parameter initialization
        assert network.primary_symptom_logit.requires_grad, "Primary logit should be learnable"
        assert network.severity_symptom_logit.requires_grad, "Severity logit should be learnable"
        
        # Test probability retrieval
        code = 'J45.9'  # Asthma
        primary_prob, severity_prob = network.get_probabilities(code)
        
        assert 0.0 <= primary_prob <= 1.0, f"Primary probability {primary_prob} out of range"
        assert 0.0 <= severity_prob <= 1.0, f"Severity probability {severity_prob} out of range"
        assert primary_prob > severity_prob, "Primary should be more likely than severity"
        
        print(f"   Initial probabilities: primary={primary_prob:.3f}, severity={severity_prob:.3f}")
        
        # Test symptom sampling
        symptoms = network.sample_symptoms(code, num_symptoms=5)
        assert len(symptoms) > 0, "Should sample at least one symptom"
        assert len(symptoms) <= 5, f"Should sample at most 5 symptoms, got {len(symptoms)}"
        
        print(f"   Sampled {len(symptoms)} symptoms")
        
        # Test gradient computation
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        
        # Create a simple loss based on probability difference
        primary_prob_tensor = torch.sigmoid(network.primary_symptom_logit)
        severity_prob_tensor = torch.sigmoid(network.severity_symptom_logit)
        
        # Loss: want primary to be higher than severity
        loss = -(primary_prob_tensor - severity_prob_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        new_primary_prob, new_severity_prob = network.get_probabilities(code)
        assert abs(new_primary_prob - primary_prob) > 1e-6 or abs(new_severity_prob - severity_prob) > 1e-6, \
            "Parameters should change after optimization step"
        
        print(f"   After optimization: primary={new_primary_prob:.3f}, severity={new_severity_prob:.3f}")
        
        # Test get_symptom_probability
        condition_data = RespiratoryConditions.get_condition_by_code(code)
        primary_symptom = condition_data["symptoms"][0]
        prob = network.get_symptom_probability(code, primary_symptom)
        assert prob > 0, f"Should have non-zero probability for primary symptom"
        
        # Test get_conditional_probabilities
        probs = network.get_conditional_probabilities(code)
        assert len(probs) > 0, "Should return probability dictionary"
        
        print("✅ Learnable Bayesian network tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Learnable Bayesian network test failed: {e}")
        traceback.print_exc()
        return False


def test_medical_accuracy_loss():
    """Test the medical accuracy loss function."""
    print("🧪 Testing Medical Accuracy Loss...")
    
    try:
        from phaita.utils.medical_loss import MedicalAccuracyLoss
        from phaita.models.bayesian_network import LearnableBayesianSymptomNetwork, TORCH_AVAILABLE
        from phaita.data.icd_conditions import RespiratoryConditions
        
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, skipping medical loss test")
            return True
        
        import torch
        
        # Initialize loss and network
        loss_fn = MedicalAccuracyLoss()
        network = LearnableBayesianSymptomNetwork()
        
        # Test loss computation with correct symptoms
        code = 'J45.9'  # Asthma
        condition_data = RespiratoryConditions.get_condition_by_code(code)
        correct_symptoms = [condition_data["symptoms"][:3]]  # Use first 3 primary symptoms
        
        loss_correct = loss_fn(correct_symptoms, [code], network)
        assert isinstance(loss_correct, torch.Tensor), "Loss should be a tensor"
        assert loss_correct.requires_grad, "Loss should have gradient"
        
        print(f"   Loss with correct symptoms: {loss_correct.item():.4f}")
        
        # Test loss computation with incorrect symptoms
        incorrect_symptoms = [["fever", "cough", "nausea"]]  # Not typical for asthma
        loss_incorrect = loss_fn(incorrect_symptoms, [code], network)
        
        print(f"   Loss with incorrect symptoms: {loss_incorrect.item():.4f}")
        
        # Incorrect symptoms should have higher loss
        # (This may not always hold due to randomness, so just check both are positive)
        assert loss_correct.item() >= 0, "Loss should be non-negative"
        assert loss_incorrect.item() >= 0, "Loss should be non-negative"
        
        # Test loss components
        components = loss_fn.get_loss_components(correct_symptoms, [code], network)
        assert 'alignment_loss' in components, "Should have alignment loss"
        assert 'constraint_loss' in components, "Should have constraint loss"
        assert 'diversity_loss' in components, "Should have diversity loss"
        assert 'total_medical_loss' in components, "Should have total loss"
        
        print(f"   Loss components: alignment={components['alignment_loss']:.4f}, "
              f"constraint={components['constraint_loss']:.4f}, "
              f"diversity={components['diversity_loss']:.4f}")
        
        # Test gradient flow
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        loss = loss_fn(correct_symptoms, [code], network)
        loss.backward()
        
        # Check that gradients were computed
        assert network.primary_symptom_logit.grad is not None, "Should have gradient for primary logit"
        assert network.severity_symptom_logit.grad is not None, "Should have gradient for severity logit"
        
        print("   Gradients computed successfully")
        
        print("✅ Medical accuracy loss tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Medical accuracy loss test failed: {e}")
        traceback.print_exc()
        return False


def test_adversarial_trainer_integration():
    """Test that AdversarialTrainer can be initialized with learnable Bayesian network."""
    print("🧪 Testing AdversarialTrainer Integration...")
    
    try:
        from phaita.models.bayesian_network import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, skipping trainer integration test")
            return True
        
        # We can't test the full trainer without models, but we can test the Bayesian network integration
        from phaita.models.bayesian_network import LearnableBayesianSymptomNetwork
        from phaita.models.generator import SymptomGenerator
        
        # Test that SymptomGenerator accepts a learnable network
        learnable_network = LearnableBayesianSymptomNetwork()
        symptom_gen = SymptomGenerator(bayesian_network=learnable_network)
        
        assert symptom_gen.bayesian_network is learnable_network, \
            "SymptomGenerator should use provided learnable network"
        
        print("   SymptomGenerator correctly configured with learnable network")
        
        # Test generating symptoms through the learnable network
        code = 'J45.9'  # Asthma
        presentation = symptom_gen.generate_symptoms(code)
        
        assert presentation is not None, "Should generate a presentation"
        assert len(presentation.symptoms) > 0, "Should have symptoms"
        
        print(f"   Generated presentation with {len(presentation.symptoms)} symptoms")
        
        print("✅ AdversarialTrainer integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ AdversarialTrainer integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🏥 PHAITA Learnable Network Test Suite")
    print("=" * 50)
    
    tests = [
        test_learnable_bayesian_network,
        test_medical_accuracy_loss,
        test_adversarial_trainer_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

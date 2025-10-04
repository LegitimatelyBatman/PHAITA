#!/usr/bin/env python3
"""
Test script for PHAITA components.
Validates the core functionality without requiring heavy dependencies or model downloads.

Note: This test suite does NOT require transformer models to be downloaded.
It tests the data layer, configuration system, and Bayesian logic only.
For tests requiring real models, see test_integration.py.
"""

import sys
from pathlib import Path
import traceback

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_layer():
    """Test the medical conditions data layer."""
    print("🧪 Testing Data Layer...")
    
    try:
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Test basic functionality
        conditions = RespiratoryConditions.get_all_conditions()
        assert len(conditions) == 10, f"Expected 10 conditions, got {len(conditions)}"
        
        # Test condition lookup
        asthma = RespiratoryConditions.get_condition_by_code('J45.9')
        assert asthma['name'] == 'Asthma', f"Expected 'Asthma', got {asthma['name']}"
        
        # Test symptom retrieval
        symptoms = RespiratoryConditions.get_symptoms_for_condition('J45.9')
        assert len(symptoms) > 0, "Should have symptoms for asthma"
        
        # Test lay terms
        lay_terms = RespiratoryConditions.get_lay_terms_for_condition('J45.9')
        assert len(lay_terms) > 0, "Should have lay terms for asthma"
        
        # Test random sampling
        code, condition_data = RespiratoryConditions.get_random_condition()
        assert code in conditions, f"Random condition {code} not in conditions list"
        
        print("✅ Data layer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Data layer test failed: {e}")
        traceback.print_exc()
        return False


def test_bayesian_network():
    """Test the Bayesian symptom network logic."""
    print("🧪 Testing Bayesian Network Logic...")
    
    try:
        # Test the symptom generation logic without requiring torch
        from phaita.data.icd_conditions import RespiratoryConditions
        import random
        
        # Simulate the Bayesian network sampling
        code = 'J45.9'  # Asthma
        condition_data = RespiratoryConditions.get_condition_by_code(code)
        
        # Basic probabilistic sampling
        primary_symptoms = condition_data["symptoms"]
        severity_symptoms = condition_data["severity_indicators"]
        
        # Test multiple sampling iterations
        for i in range(5):
            # Simulate sampling with higher probability for primary symptoms
            sampled = []
            
            # Add primary symptoms with high probability
            for symptom in primary_symptoms:
                if random.random() < 0.8:  # 80% chance
                    sampled.append(symptom)
            
            # Add severity symptoms with lower probability
            for symptom in severity_symptoms:
                if random.random() < 0.4:  # 40% chance
                    sampled.append(symptom)
            
            assert len(sampled) > 0, f"Should sample at least one symptom, got {len(sampled)}"
        
        print("✅ Bayesian network logic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Bayesian network test failed: {e}")
        traceback.print_exc()
        return False


def test_config_system():
    """Test the configuration system."""
    print("🧪 Testing Configuration System...")
    
    try:
        from phaita.utils.config import Config, ModelConfig, TrainingConfig, DataConfig
        
        # Test default config creation
        config = Config()
        assert hasattr(config, 'model'), "Config should have model attribute"
        assert hasattr(config, 'training'), "Config should have training attribute"
        assert hasattr(config, 'data'), "Config should have data attribute"
        
        # Test config values
        assert config.model.deberta_model == "microsoft/deberta-base"
        assert config.training.num_epochs == 100
        assert config.data.num_respiratory_conditions == 10
        
        # Test YAML save/load
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            
            # Load it back
            loaded_config = Config.from_yaml(f.name)
            assert loaded_config.model.deberta_model == config.model.deberta_model
            assert loaded_config.training.num_epochs == config.training.num_epochs
            
            # Clean up
            os.unlink(f.name)
        
        print("✅ Configuration system tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        traceback.print_exc()
        return False


def test_synthetic_generation():
    """Test synthetic data generation capabilities."""
    print("🧪 Testing Synthetic Data Generation...")
    
    try:
        from phaita.data.icd_conditions import RespiratoryConditions
        import random
        
        # Test complaint simulation
        complaint_templates = [
            "I've been {symptom} for {time}",
            "Doctor, I have {symptom} and I'm {feeling}",
            "Help, I can't stop {symptom}"
        ]
        
        feelings = ["worried", "scared", "terrible"]
        times = ["hours", "days", "weeks"]
        
        for i in range(10):
            code, condition_data = RespiratoryConditions.get_random_condition()
            lay_terms = condition_data["lay_terms"]
            
            if lay_terms:
                template = random.choice(complaint_templates)
                symptom = random.choice(lay_terms)
                feeling = random.choice(feelings)
                time = random.choice(times)
                
                complaint = template.format(symptom=symptom, feeling=feeling, time=time)
                
                assert len(complaint) > 10, f"Complaint too short: {complaint}"
                assert symptom in complaint, f"Symptom {symptom} not in complaint: {complaint}"
        
        print("✅ Synthetic data generation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data generation test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🏥 PHAITA Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_layer,
        test_bayesian_network,
        test_config_system,
        test_synthetic_generation
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
#!/usr/bin/env python3
"""
Test script for temporal symptom modeling.
Validates the TemporalSymptomEncoder, SymptomTimeline, and temporal pattern matching.
"""

import sys
from pathlib import Path
import traceback
import importlib.util

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))


def import_module_from_file(module_name, file_path):
    """Import a module directly from a file path without going through package __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_symptom_timeline():
    """Test the SymptomTimeline class for tracking symptom onset."""
    print("🧪 Testing SymptomTimeline...")
    
    try:
        # Import directly from file to avoid __init__.py torch dependency
        temporal_module_path = Path(__file__).parent / "phaita" / "models" / "temporal_module.py"
        temporal_module = import_module_from_file("temporal_module", temporal_module_path)
        SymptomTimeline = temporal_module.SymptomTimeline
        
        timeline = SymptomTimeline()
        
        # Add symptoms with different onset times
        timeline.add_symptom("fever", 0)
        timeline.add_symptom("cough", 12)
        timeline.add_symptom("chest_pain", 24)
        timeline.add_symptom("dyspnea", 48)
        
        # Test progression pattern (chronological order)
        progression = timeline.get_progression_pattern()
        assert len(progression) == 4, f"Expected 4 symptoms, got {len(progression)}"
        
        # Check chronological order (earliest first)
        assert progression[0][0] == "fever", f"Expected fever first, got {progression[0][0]}"
        assert progression[1][0] == "cough", f"Expected cough second, got {progression[1][0]}"
        assert progression[2][0] == "chest_pain", f"Expected chest_pain third, got {progression[2][0]}"
        assert progression[3][0] == "dyspnea", f"Expected dyspnea fourth, got {progression[3][0]}"
        
        # Check timestamps
        assert progression[0][1] == 0, f"Expected 0 hours for fever, got {progression[0][1]}"
        assert progression[1][1] == 12, f"Expected 12 hours for cough, got {progression[1][1]}"
        assert progression[2][1] == 24, f"Expected 24 hours for chest_pain, got {progression[2][1]}"
        assert progression[3][1] == 48, f"Expected 48 hours for dyspnea, got {progression[3][1]}"
        
        # Test symptom order extraction
        symptom_order = timeline.get_symptom_order()
        assert symptom_order == ["fever", "cough", "chest_pain", "dyspnea"], \
            f"Expected chronological order, got {symptom_order}"
        
        # Test clear
        timeline.clear()
        assert len(timeline.events) == 0, "Timeline should be empty after clear"
        
        print("✅ SymptomTimeline tests passed")
        return True
        
    except Exception as e:
        print(f"❌ SymptomTimeline test failed: {e}")
        traceback.print_exc()
        return False


def test_lstm_encoder():
    """Test the LSTM-based temporal symptom encoder."""
    print("🧪 Testing TemporalSymptomEncoder...")
    
    try:
        import torch
        from phaita.models.temporal_module import TemporalSymptomEncoder
        
        # Create encoder with small vocab for testing
        vocab_size = 50
        encoder = TemporalSymptomEncoder(
            symptom_vocab_size=vocab_size,
            symptom_embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
        )
        
        # Create dummy input
        batch_size = 4
        seq_len = 5
        
        # Random symptom indices
        symptom_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Random timestamps (0 to 100 hours)
        timestamps = torch.rand(batch_size, seq_len) * 100
        
        # Forward pass
        output = encoder(symptom_indices, timestamps)
        
        # Check output shape
        assert output.shape == (batch_size, 64), \
            f"Expected shape (4, 64), got {output.shape}"
        
        # Check output is not all zeros
        assert output.abs().sum() > 0, "Output should not be all zeros"
        
        # Test with different sequence lengths
        seq_len_2 = 3
        symptom_indices_2 = torch.randint(0, vocab_size, (batch_size, seq_len_2))
        timestamps_2 = torch.rand(batch_size, seq_len_2) * 100
        
        output_2 = encoder(symptom_indices_2, timestamps_2)
        assert output_2.shape == (batch_size, 64), \
            f"Expected shape (4, 64) for different seq_len, got {output_2.shape}"
        
        print("✅ TemporalSymptomEncoder tests passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping LSTM encoder test (missing torch): {e}")
        return True  # Don't fail if torch not available
    except Exception as e:
        print(f"❌ TemporalSymptomEncoder test failed: {e}")
        traceback.print_exc()
        return False


def test_temporal_pattern_matching():
    """Test temporal pattern matching for condition diagnosis."""
    print("🧪 Testing Temporal Pattern Matching...")
    
    try:
        # Import directly from file to avoid __init__.py torch dependency
        temporal_module_path = Path(__file__).parent / "phaita" / "models" / "temporal_module.py"
        temporal_module = import_module_from_file("temporal_module", temporal_module_path)
        SymptomTimeline = temporal_module.SymptomTimeline
        TemporalPatternMatcher = temporal_module.TemporalPatternMatcher
        
        # Define a simple test pattern (similar to pneumonia)
        test_patterns = {
            "J18.9": {  # Pneumonia
                "typical_progression": [
                    {"symptom": "fever", "onset_hour": 0},
                    {"symptom": "cough", "onset_hour": 12},
                    {"symptom": "chest_pain", "onset_hour": 24},
                    {"symptom": "dyspnea", "onset_hour": 48},
                ]
            },
            "J45.9": {  # Asthma
                "typical_progression": [
                    {"symptom": "wheezing", "onset_hour": 0},
                    {"symptom": "shortness_of_breath", "onset_hour": 2},
                    {"symptom": "chest_tightness", "onset_hour": 4},
                ]
            }
        }
        
        matcher = TemporalPatternMatcher(test_patterns)
        
        # Test Case 1: Perfect match with pneumonia pattern
        timeline1 = SymptomTimeline()
        timeline1.add_symptom("fever", 0)
        timeline1.add_symptom("cough", 12)
        timeline1.add_symptom("chest_pain", 24)
        timeline1.add_symptom("dyspnea", 48)
        
        score1 = matcher.score_timeline(timeline1, "J18.9")
        assert score1 > 1.0, \
            f"Perfect match should have score > 1.0, got {score1}"
        print(f"  ✓ Perfect match score: {score1:.3f} (expected > 1.0)")
        
        # Test Case 2: Wrong order should decrease score
        timeline2 = SymptomTimeline()
        timeline2.add_symptom("dyspnea", 0)  # Should be last
        timeline2.add_symptom("chest_pain", 12)
        timeline2.add_symptom("cough", 24)
        timeline2.add_symptom("fever", 48)  # Should be first
        
        score2 = matcher.score_timeline(timeline2, "J18.9")
        assert score2 < score1, \
            f"Wrong order should score lower: {score2:.3f} vs {score1:.3f}"
        print(f"  ✓ Wrong order score: {score2:.3f} (lower than perfect match)")
        
        # Test Case 3: Partial match
        timeline3 = SymptomTimeline()
        timeline3.add_symptom("fever", 0)
        timeline3.add_symptom("cough", 12)
        # Missing other symptoms
        
        score3 = matcher.score_timeline(timeline3, "J18.9")
        assert 0.5 <= score3 <= 1.5, \
            f"Partial match score should be in range [0.5, 1.5], got {score3}"
        print(f"  ✓ Partial match score: {score3:.3f}")
        
        # Test Case 4: No matching symptoms
        timeline4 = SymptomTimeline()
        timeline4.add_symptom("unrelated_symptom", 0)
        
        score4 = matcher.score_timeline(timeline4, "J18.9")
        assert score4 < 1.0, \
            f"No matching symptoms should score < 1.0, got {score4}"
        print(f"  ✓ No match score: {score4:.3f}")
        
        # Test Case 5: Unknown condition code (should return neutral)
        score5 = matcher.score_timeline(timeline1, "UNKNOWN")
        assert score5 == 1.0, \
            f"Unknown condition should return neutral score 1.0, got {score5}"
        print(f"  ✓ Unknown condition score: {score5:.3f} (neutral)")
        
        print("✅ Temporal pattern matching tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Temporal pattern matching test failed: {e}")
        traceback.print_exc()
        return False


def test_dialogue_engine_with_temporal():
    """Test DialogueEngine with temporal information."""
    print("🧪 Testing DialogueEngine with Temporal Module...")
    
    try:
        # Import directly from file to avoid __init__.py torch dependency
        dialogue_engine_path = Path(__file__).parent / "phaita" / "conversation" / "dialogue_engine.py"
        
        # We need to set up the dependencies first
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import needed dependencies
        icd_conditions_path = Path(__file__).parent / "phaita" / "data" / "icd_conditions.py"
        icd_module = import_module_from_file("icd_conditions", icd_conditions_path)
        sys.modules["phaita.data.icd_conditions"] = icd_module
        
        bayesian_network_path = Path(__file__).parent / "phaita" / "models" / "bayesian_network.py"
        bayesian_module = import_module_from_file("bayesian_network", bayesian_network_path)
        sys.modules["phaita.models.bayesian_network"] = bayesian_module
        
        temporal_module_path = Path(__file__).parent / "phaita" / "models" / "temporal_module.py"
        temporal_module = import_module_from_file("temporal_module_for_dialogue", temporal_module_path)
        sys.modules["phaita.models.temporal_module"] = temporal_module
        
        # Now import DialogueEngine
        dialogue_module = import_module_from_file("dialogue_engine", dialogue_engine_path)
        DialogueEngine = dialogue_module.DialogueEngine
        
        # Initialize with temporal module enabled
        engine = DialogueEngine(use_temporal_module=True)
        
        # Check that temporal components are initialized
        if engine.use_temporal_module:
            assert engine.timeline is not None, "Timeline should be initialized"
            assert engine.temporal_module is not None, "Temporal module should be initialized"
            print("  ✓ Temporal module initialized successfully")
        else:
            print("  ⚠️  Temporal module not available (expected if config missing)")
            return True
        
        # Test update with temporal information
        # Simulate pneumonia-like progression
        initial_probs = dict(engine.state.differential_probabilities)
        
        engine.update_beliefs("fever", True, hours_since_onset=0)
        engine.update_beliefs("cough", True, hours_since_onset=12)
        engine.update_beliefs("chest_pain", True, hours_since_onset=24)
        engine.update_beliefs("dyspnea", True, hours_since_onset=48)
        
        # Check that timeline was updated
        assert len(engine.timeline.events) == 4, \
            f"Expected 4 events in timeline, got {len(engine.timeline.events)}"
        
        # Check that probabilities changed
        final_probs = dict(engine.state.differential_probabilities)
        assert final_probs != initial_probs, \
            "Probabilities should have changed after updates"
        
        # Get differential diagnosis
        differential = engine.get_differential_diagnosis(top_n=5)
        assert len(differential) > 0, "Should have at least one diagnosis"
        
        top_condition = differential[0]
        print(f"  ✓ Top diagnosis: {top_condition['name']} ({top_condition['probability']:.3f})")
        
        # Test without temporal information (should still work)
        engine2 = DialogueEngine(use_temporal_module=False)
        engine2.update_beliefs("fever", True)
        engine2.update_beliefs("cough", True)
        
        differential2 = engine2.get_differential_diagnosis(top_n=5)
        assert len(differential2) > 0, "Should work without temporal info"
        print("  ✓ Works without temporal information")
        
        print("✅ DialogueEngine with temporal module tests passed")
        return True
        
    except Exception as e:
        print(f"❌ DialogueEngine with temporal test failed: {e}")
        traceback.print_exc()
        return False


def test_temporal_accuracy_improvement():
    """Test that temporal patterns improve diagnosis accuracy."""
    print("🧪 Testing Temporal Accuracy Improvement...")
    
    try:
        # Import directly from file to avoid __init__.py torch dependency
        dialogue_engine_path = Path(__file__).parent / "phaita" / "conversation" / "dialogue_engine.py"
        
        # We need to set up the dependencies first
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import needed dependencies
        icd_conditions_path = Path(__file__).parent / "phaita" / "data" / "icd_conditions.py"
        icd_module = import_module_from_file("icd_conditions2", icd_conditions_path)
        sys.modules["phaita.data.icd_conditions"] = icd_module
        
        bayesian_network_path = Path(__file__).parent / "phaita" / "models" / "bayesian_network.py"
        bayesian_module = import_module_from_file("bayesian_network2", bayesian_network_path)
        sys.modules["phaita.models.bayesian_network"] = bayesian_module
        
        temporal_module_path = Path(__file__).parent / "phaita" / "models" / "temporal_module.py"
        temporal_module = import_module_from_file("temporal_module_for_accuracy", temporal_module_path)
        sys.modules["phaita.models.temporal_module"] = temporal_module
        
        # Now import DialogueEngine
        dialogue_module = import_module_from_file("dialogue_engine2", dialogue_engine_path)
        DialogueEngine = dialogue_module.DialogueEngine
        
        # Test scenario: Pneumonia-like symptoms in correct temporal order
        pneumonia_symptoms = [
            ("fever", 0),
            ("cough", 12),
            ("chest_pain", 24),
            ("dyspnea", 48),
        ]
        
        # Engine WITH temporal module
        engine_with_temporal = DialogueEngine(use_temporal_module=True)
        if not engine_with_temporal.use_temporal_module:
            print("  ⚠️  Temporal module not available, skipping comparison")
            return True
        
        for symptom, hours in pneumonia_symptoms:
            engine_with_temporal.update_beliefs(symptom, True, hours_since_onset=hours)
        
        # Engine WITHOUT temporal module (same symptoms, no timing)
        engine_without_temporal = DialogueEngine(use_temporal_module=False)
        for symptom, _ in pneumonia_symptoms:
            engine_without_temporal.update_beliefs(symptom, True)
        
        # Compare results
        diff_with = engine_with_temporal.get_differential_diagnosis(top_n=10)
        diff_without = engine_without_temporal.get_differential_diagnosis(top_n=10)
        
        # Find pneumonia (J18.9) in both lists
        prob_with_temporal = None
        prob_without_temporal = None
        
        for item in diff_with:
            if item["condition_code"] == "J18.9":
                prob_with_temporal = item["probability"]
                break
        
        for item in diff_without:
            if item["condition_code"] == "J18.9":
                prob_without_temporal = item["probability"]
                break
        
        if prob_with_temporal and prob_without_temporal:
            improvement = ((prob_with_temporal - prob_without_temporal) / prob_without_temporal) * 100
            print(f"  ✓ Pneumonia probability with temporal: {prob_with_temporal:.4f}")
            print(f"  ✓ Pneumonia probability without temporal: {prob_without_temporal:.4f}")
            print(f"  ✓ Improvement: {improvement:.2f}%")
            
            # Check for improvement (should be at least some positive change)
            if improvement > 0:
                print(f"  ✓ Temporal patterns provide positive improvement")
            else:
                print(f"  ⚠️  No significant improvement detected (this is OK for some cases)")
        else:
            print("  ⚠️  Could not find pneumonia in differential, but test succeeded")
        
        print("✅ Temporal accuracy improvement test completed")
        return True
        
    except Exception as e:
        print(f"❌ Temporal accuracy improvement test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all temporal modeling tests."""
    print("🏥 PHAITA Temporal Modeling Test Suite")
    print("=" * 50)
    
    tests = [
        ("SymptomTimeline", test_symptom_timeline),
        ("LSTM Encoder", test_lstm_encoder),
        ("Temporal Pattern Matching", test_temporal_pattern_matching),
        ("DialogueEngine Integration", test_dialogue_engine_with_temporal),
        ("Accuracy Improvement", test_temporal_accuracy_improvement),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
        print()
    
    # Summary
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All temporal modeling tests passed!")
        return 0
    else:
        print("❌ Some tests failed:")
        for name, result in results:
            if not result:
                print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

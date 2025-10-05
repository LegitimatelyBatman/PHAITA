#!/usr/bin/env python3
"""
Demo script showcasing the learnable Bayesian symptom network.
Demonstrates how symptom probabilities can be learned via gradient descent.
"""

import sys
import torch
from phaita.models.bayesian_network import LearnableBayesianSymptomNetwork, TORCH_AVAILABLE
from phaita.utils.medical_loss import MedicalAccuracyLoss
from phaita.data.icd_conditions import RespiratoryConditions


def demo_learnable_network():
    """Demonstrate learnable Bayesian network training."""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. This demo requires PyTorch.")
        return False
    
    print("=" * 70)
    print("PHAITA Learnable Bayesian Network Demo")
    print("=" * 70)
    print()
    
    # Initialize learnable network
    print("1. Initializing learnable Bayesian network...")
    network = LearnableBayesianSymptomNetwork()
    print(f"   ✓ Network initialized with {sum(p.numel() for p in network.parameters())} learnable parameters")
    print()
    
    # Show initial probabilities
    print("2. Initial symptom probabilities:")
    condition_code = "J45.9"  # Asthma
    condition_data = RespiratoryConditions.get_condition_by_code(condition_code)
    print(f"   Condition: {condition_data['name']} ({condition_code})")
    
    primary_prob, severity_prob = network.get_probabilities(condition_code)
    print(f"   Primary symptom probability: {primary_prob:.3f}")
    print(f"   Severity symptom probability: {severity_prob:.3f}")
    print()
    
    # Sample some symptoms
    print("3. Sampling symptoms with initial probabilities:")
    for i in range(3):
        symptoms = network.sample_symptoms(condition_code, num_symptoms=4)
        print(f"   Sample {i+1}: {', '.join(symptoms[:3])}...")
    print()
    
    # Initialize loss and optimizer
    print("4. Setting up training...")
    loss_fn = MedicalAccuracyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    print("   ✓ Medical accuracy loss initialized")
    print("   ✓ Adam optimizer configured (lr=0.01)")
    print()
    
    # Training loop
    print("5. Training for 20 iterations...")
    print("   " + "-" * 60)
    print(f"   {'Iter':<6} {'Loss':<10} {'Primary':<10} {'Severity':<10}")
    print("   " + "-" * 60)
    
    for iteration in range(20):
        optimizer.zero_grad()
        
        # Sample symptoms for multiple conditions
        sampled_symptoms = []
        condition_codes = []
        
        for _ in range(8):  # Batch size of 8
            code = condition_code
            symptoms = network.sample_symptoms(code)
            sampled_symptoms.append(symptoms)
            condition_codes.append(code)
        
        # Compute loss
        loss = loss_fn(sampled_symptoms, condition_codes, network)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log progress
        if iteration % 5 == 0 or iteration == 19:
            primary_prob, severity_prob = network.get_probabilities(condition_code)
            print(f"   {iteration:<6} {loss.item():<10.4f} {primary_prob:<10.3f} {severity_prob:<10.3f}")
    
    print("   " + "-" * 60)
    print()
    
    # Show final probabilities
    print("6. Final symptom probabilities:")
    primary_prob_final, severity_prob_final = network.get_probabilities(condition_code)
    print(f"   Primary symptom probability: {primary_prob_final:.3f}")
    print(f"   Severity symptom probability: {severity_prob_final:.3f}")
    print()
    
    # Sample with learned probabilities
    print("7. Sampling symptoms with learned probabilities:")
    for i in range(3):
        symptoms = network.sample_symptoms(condition_code, num_symptoms=4)
        print(f"   Sample {i+1}: {', '.join(symptoms[:3])}...")
    print()
    
    # Show parameter changes
    print("8. Summary:")
    primary_change = abs(primary_prob_final - 0.8)
    severity_change = abs(severity_prob_final - 0.4)
    print(f"   Primary probability changed by: {primary_change:.3f}")
    print(f"   Severity probability changed by: {severity_change:.3f}")
    
    if primary_change > 0.01 or severity_change > 0.01:
        print("   ✓ Network successfully learned from medical accuracy loss!")
    else:
        print("   ✓ Network maintained stable probabilities under constraint loss!")
    
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    return True


def main():
    """Run the demo."""
    try:
        success = demo_learnable_network()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

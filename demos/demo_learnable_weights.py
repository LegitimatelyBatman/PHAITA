#!/usr/bin/env python3
"""
Demo of learnable comorbidity effects and symptom causality.

Shows how to use the new learnable modules and compare them to fixed weights.
"""

import torch
from phaita.models.learnable_comorbidity import LearnableComorbidityEffects
from phaita.models.learnable_causality import LearnableSymptomCausality
from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network
from phaita.data.icd_conditions import RespiratoryConditions


def demo_learnable_comorbidity():
    """Demonstrate learnable comorbidity effects."""
    print("=" * 70)
    print("LEARNABLE COMORBIDITY EFFECTS DEMO")
    print("=" * 70)
    
    # Create learnable comorbidity module
    learnable = LearnableComorbidityEffects()
    
    print(f"\n📊 Module Statistics:")
    print(f"   Comorbidities: {len(learnable.comorbidity_vocab)}")
    print(f"   Symptoms affected: {len(learnable.symptom_vocab)}")
    print(f"   Total parameters: {learnable.comorbidity_weights.numel()}")
    print(f"   Device: {learnable.device}")
    
    # Show learned modifiers for diabetes
    print(f"\n💊 Diabetes Effect Modifiers (learned):")
    diabetes_modifiers = learnable.get_symptom_modifiers("diabetes")
    for symptom, multiplier in sorted(diabetes_modifiers.items())[:5]:
        print(f"   {symptom}: {multiplier:.3f}x")
    
    # Demonstrate gradient computation
    print(f"\n🎓 Gradient Computation Demo:")
    weights = learnable.forward()
    loss = weights.sum()
    loss.backward()
    
    grad_norm = learnable.comorbidity_weights.grad.norm().item()
    print(f"   Loss: {loss.item():.3f}")
    print(f"   Gradient norm: {grad_norm:.3f}")
    print(f"   ✅ Gradients computed successfully!")
    
    # Use in enhanced Bayesian network
    print(f"\n🏥 Integration with Enhanced Bayesian Network:")
    network_learnable = create_enhanced_bayesian_network(use_learnable_comorbidity=True)
    network_fixed = create_enhanced_bayesian_network(use_learnable_comorbidity=False)
    
    # Sample with diabetes comorbidity
    symptoms_learnable, _ = network_learnable.sample_symptoms(
        "J18.9",  # Pneumonia
        comorbidities=["diabetes"]
    )
    symptoms_fixed, _ = network_fixed.sample_symptoms(
        "J18.9",
        comorbidities=["diabetes"]
    )
    
    print(f"   Learnable mode: {len(symptoms_learnable)} symptoms")
    print(f"   Fixed mode: {len(symptoms_fixed)} symptoms")
    print(f"   ✅ Both modes work correctly!")


def demo_learnable_causality():
    """Demonstrate learnable symptom causality."""
    print("\n" + "=" * 70)
    print("LEARNABLE SYMPTOM CAUSALITY DEMO")
    print("=" * 70)
    
    # Create learnable causality module
    learnable = LearnableSymptomCausality()
    
    print(f"\n📊 Module Statistics:")
    print(f"   Causal edges: {len(learnable.causal_edge_pairs)}")
    print(f"   Temporal edges: {len(learnable.temporal_edge_pairs)}")
    print(f"   Causal parameters: {learnable.causal_weights.numel()}")
    print(f"   Temporal parameters: {learnable.temporal_weights.numel()}")
    print(f"   Device: {learnable.device}")
    
    # Show learned causal edges
    print(f"\n🔗 Causal Edges (learned strengths):")
    causal_edges = learnable.get_causal_edges()
    for source, target, strength in causal_edges[:5]:
        print(f"   {source} → {target}: {strength:.3f}")
    
    # Show learned temporal edges
    print(f"\n⏰ Temporal Edges (learned strengths, fixed delays):")
    temporal_edges = learnable.get_temporal_edges()
    for earlier, later, strength, delay in temporal_edges[:3]:
        delay_hours = delay * 168.0  # Denormalize
        print(f"   {earlier} → {later}: strength={strength:.3f}, delay={delay_hours:.1f}h")
    
    # Demonstrate gradient computation
    print(f"\n🎓 Gradient Computation Demo:")
    causal_strengths, temporal_strengths = learnable.forward()
    loss = causal_strengths.sum() + temporal_strengths.sum()
    loss.backward()
    
    causal_grad_norm = learnable.causal_weights.grad.norm().item() if len(learnable.causal_weights) > 0 else 0
    temporal_grad_norm = learnable.temporal_weights.grad.norm().item() if len(learnable.temporal_weights) > 0 else 0
    print(f"   Loss: {loss.item():.3f}")
    print(f"   Causal gradient norm: {causal_grad_norm:.3f}")
    print(f"   Temporal gradient norm: {temporal_grad_norm:.3f}")
    print(f"   ✅ Gradients computed successfully!")
    
    # Show GNN-compatible config
    print(f"\n🧠 GNN Configuration:")
    config = learnable.get_config_for_gnn()
    print(f"   Edge types: {list(config['edge_types'].keys())}")
    print(f"   Causal edges: {len(config['causal_edges'])}")
    print(f"   Temporal edges: {len(config['temporal_edges'])}")
    print(f"   ✅ GNN integration ready!")


def demo_comparison():
    """Compare learned vs fixed weights."""
    print("\n" + "=" * 70)
    print("LEARNED VS FIXED WEIGHTS COMPARISON")
    print("=" * 70)
    
    # Create both versions
    learnable_comorbidity = LearnableComorbidityEffects()
    
    # Get diabetes modifiers
    diabetes_learned = learnable_comorbidity.get_symptom_modifiers("diabetes")
    
    # Load fixed values from config (these are the initial values)
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config" / "comorbidity_effects.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            diabetes_fixed = config.get('diabetes', {}).get('symptom_modifiers', {})
    else:
        diabetes_fixed = {"fatigue": 1.3, "infection_risk": 1.5}
    
    print(f"\n💊 Diabetes Symptom Modifiers:")
    print(f"\n{'Symptom':<20} {'Fixed Weight':<15} {'Learned Weight':<15} {'Difference':<10}")
    print("-" * 65)
    
    for symptom in sorted(set(list(diabetes_fixed.keys()) + list(diabetes_learned.keys()))):
        fixed_val = diabetes_fixed.get(symptom, 1.0)
        learned_val = diabetes_learned.get(symptom, 1.0)
        diff = learned_val - fixed_val
        print(f"{symptom:<20} {fixed_val:<15.3f} {learned_val:<15.3f} {diff:+.3f}")
    
    print(f"\n📝 Note: Learned weights start from fixed (clinical) values.")
    print(f"   During training, they would diverge to optimize for data.")


def demo_training_workflow():
    """Show how to use learnable modules in training."""
    print("\n" + "=" * 70)
    print("TRAINING WORKFLOW EXAMPLE")
    print("=" * 70)
    
    print(f"\n📚 Example code for using learnable modules in training:")
    print("""
from phaita.training.adversarial_trainer import AdversarialTrainer

# Create trainer with all learnable modules enabled
trainer = AdversarialTrainer(
    use_pretrained_generator=True,
    use_pretrained_discriminator=True,
    use_learnable_bayesian=True,       # Learn Bayesian priors
    use_learnable_comorbidity=True,    # Learn comorbidity effects  
    use_learnable_causality=True,      # Learn causal strengths
    bayesian_lr=1e-3,
    comorbidity_lr=1e-3,
    causality_lr=1e-3,
    device="cuda"
)

# Train (will update all learnable weights)
trainer.train(num_epochs=10, batch_size=8)

# Inspect learned weights
if trainer.learnable_comorbidity:
    diabetes_effects = trainer.learnable_comorbidity.get_symptom_modifiers("diabetes")
    print("Learned diabetes effects:", diabetes_effects)

# Save learned weights
import torch
if trainer.learnable_comorbidity:
    torch.save(trainer.learnable_comorbidity.state_dict(), "learned_comorbidity.pt")
if trainer.learnable_causality:
    torch.save(trainer.learnable_causality.state_dict(), "learned_causality.pt")
""")
    
    print(f"\n✅ See docs/features/LEARNABLE_WEIGHTS_GUIDE.md for full documentation")


def main():
    """Run all demos."""
    print("\n🎯 PHAITA Learnable Weights Demo")
    print("   Making comorbidity effects and symptom causality learnable!\n")
    
    try:
        demo_learnable_comorbidity()
        demo_learnable_causality()
        demo_comparison()
        demo_training_workflow()
        
        print("\n" + "=" * 70)
        print("🎉 All demos completed successfully!")
        print("=" * 70)
        print(f"\n📖 For more information:")
        print(f"   - Documentation: docs/features/LEARNABLE_WEIGHTS_GUIDE.md")
        print(f"   - Tests: tests/test_learnable_modules.py")
        print(f"   - Modules: phaita/models/learnable_*.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
Demonstration of the uncertainty quantification implementation.
This demo shows the API and expected behavior without requiring model downloads.
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_uncertainty_api():
    """Demonstrate the uncertainty quantification API."""
    print("=" * 70)
    print("Uncertainty Quantification Demo")
    print("=" * 70)
    print()
    
    print("This demo shows the new uncertainty quantification features:")
    print()
    print("1. predict_with_uncertainty() - Monte Carlo Dropout")
    print("   - Enables dropout during inference")
    print("   - Runs multiple forward passes (default: 20)")
    print("   - Returns mean and standard deviation of predictions")
    print()
    print("2. predict_diagnosis() enhancements")
    print("   - New parameter: use_mc_dropout (default: True)")
    print("   - New parameter: num_mc_samples (default: 20)")
    print("   - New output: 'uncertainty' field (0-1, lower is more certain)")
    print("   - New output: 'confidence_level' field ('high' or 'low')")
    print()
    
    print("-" * 70)
    print("API Usage Example:")
    print("-" * 70)
    print()
    
    code_example = '''
from phaita.models.discriminator import DiagnosisDiscriminator

# Initialize discriminator
model = DiagnosisDiscriminator(use_pretrained=True)

# Example 1: Get predictions with uncertainty (MC dropout enabled)
predictions = model.predict_diagnosis(
    complaints=["I have chest pain and shortness of breath"],
    top_k=3,
    use_mc_dropout=True,      # Enable Monte Carlo Dropout
    num_mc_samples=20         # Number of samples
)

# Access uncertainty information
for pred in predictions[0]:
    print(f"Condition: {pred['condition_name']}")
    print(f"Probability: {pred['probability']:.3f}")
    print(f"Uncertainty: {pred['uncertainty']:.3f}")  # New field
    print(f"Confidence: {pred['confidence_level']}")  # New field
    print()

# Example 2: Direct uncertainty estimation with structured output
results = model.predict_with_uncertainty(
    complaints=["I have chest pain"],
    num_samples=20,
    top_k=3
)
# Returns list of dictionaries with uncertainty info
for pred in results[0]:
    print(f"Condition: {pred['condition_name']}")
    print(f"Probability: {pred['probability']:.3f}")
    print(f"Uncertainty: {pred['uncertainty']:.3f}")
    print(f"Confidence: {pred['confidence_level']}")
    print()

# Example 3: Disable MC dropout for faster inference
predictions_fast = model.predict_diagnosis(
    complaints=["I have chest pain"],
    top_k=1,
    use_mc_dropout=False  # Single forward pass, faster
)
'''
    print(code_example)
    
    print("-" * 70)
    print("Output Structure:")
    print("-" * 70)
    print()
    
    output_example = '''{
    "condition_code": "J45.9",
    "condition_name": "Asthma",
    "probability": 0.7234,
    "confidence_interval": (0.6845, 0.7623),
    "evidence": {
        "key_symptoms": ["wheezing", "shortness_of_breath", ...],
        "severity_indicators": ["chest_tightness", ...],
        "description": "..."
    },
    "uncertainty": 0.2341,           # NEW: Uncertainty score [0-1]
    "confidence_level": "high"       # NEW: "high" or "low"
}'''
    print(output_example)
    print()
    
    print("-" * 70)
    print("Technical Details:")
    print("-" * 70)
    print()
    print("Uncertainty Calculation:")
    print("  - Epistemic uncertainty: Entropy of prediction distribution")
    print("  - Aleatoric uncertainty: MC dropout variance")
    print("  - Combined score: 70% entropy + 30% variance")
    print("  - Threshold: uncertainty < 0.3 = 'high' confidence")
    print()
    print("Monte Carlo Dropout:")
    print("  - Standard Bayesian deep learning technique")
    print("  - Measures model uncertainty (epistemic)")
    print("  - More samples = more accurate uncertainty estimate")
    print("  - Recommended: 20-50 samples for production")
    print()
    print("Entropy Normalization:")
    print("  - Formula: H = -Σ(p * log(p))")
    print("  - Max entropy = log(num_conditions) ≈ 2.303 for 10 conditions")
    print("  - Normalized to [0, 1] range")
    print()
    
    print("-" * 70)
    print("Use Cases:")
    print("-" * 70)
    print()
    print("High Uncertainty (uncertainty > 0.3):")
    print("  - Ambiguous symptoms")
    print("  - Multiple conditions possible")
    print("  - Recommend additional questions/tests")
    print("  - Flag for expert review")
    print()
    print("Low Uncertainty (uncertainty < 0.3):")
    print("  - Clear symptom presentation")
    print("  - Single condition likely")
    print("  - Higher confidence in diagnosis")
    print("  - Can proceed with treatment plan")
    print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_uncertainty_api()

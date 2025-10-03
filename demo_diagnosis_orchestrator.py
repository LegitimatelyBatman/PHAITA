"""
Demo for the Diagnosis Orchestrator with Red-Flag Integration.

This script demonstrates how to use the DiagnosisOrchestrator to combine
predictions from Bayesian reasoning and neural networks, detect red-flag
symptoms, and provide appropriate escalation guidance.
"""

from phaita.triage import DiagnosisOrchestrator


def demo_emergency_case():
    """Demonstrate orchestration for an emergency case with red-flags."""
    print("=" * 70)
    print("SCENARIO 1: Emergency Case - Severe Asthma Attack")
    print("=" * 70)
    
    orchestrator = DiagnosisOrchestrator()
    
    # Bayesian probabilities from dialogue engine
    bayesian_probs = {
        "J45.9": 0.6,   # Asthma
        "J18.9": 0.25,  # Pneumonia
        "J44.9": 0.15,  # COPD
    }
    
    # Neural network predictions from discriminator
    neural_predictions = [
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.7},
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.2},
        {"condition_code": "J44.9", "condition_name": "COPD", "probability": 0.1},
    ]
    
    # Patient symptoms including red-flags
    patient_symptoms = [
        "cough",
        "wheezing",
        "severe respiratory distress",  # RED FLAG
        "unable to speak full sentences",  # RED FLAG
        "chest tightness",
    ]
    
    print("\n📋 Patient Symptoms:")
    for symptom in patient_symptoms:
        marker = "🚨" if "severe" in symptom or "unable" in symptom else "•"
        print(f"   {marker} {symptom}")
    
    print("\n🔬 Orchestrating Diagnosis...\n")
    
    # Run orchestration
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs=bayesian_probs,
        neural_predictions=neural_predictions,
        patient_symptoms=patient_symptoms,
        top_k=3
    )
    
    # Display results
    print("📊 Differential Diagnosis (Top 3):\n")
    for i, diagnosis in enumerate(diagnoses, 1):
        print(f"{i}. {diagnosis.condition_name} ({diagnosis.condition_code})")
        print(f"   • Probability: {diagnosis.probability:.1%}")
        print(f"   • Escalation: {diagnosis.escalation_level.upper()}")
        if diagnosis.red_flags:
            print(f"   • 🚨 Red Flags Detected: {', '.join(diagnosis.red_flags)}")
        print(f"   • Reasoning: {diagnosis.reasoning}")
        print()
    
    # Show guidance for top diagnosis
    top_diagnosis = diagnoses[0]
    guidance = orchestrator.generate_guidance_text(top_diagnosis.escalation_level)
    print("📢 Recommended Action:")
    print(f"   {guidance}")
    print()


def demo_urgent_case():
    """Demonstrate orchestration for an urgent case without red-flags."""
    print("=" * 70)
    print("SCENARIO 2: Urgent Case - Probable Upper Respiratory Infection")
    print("=" * 70)
    
    orchestrator = DiagnosisOrchestrator()
    
    bayesian_probs = {
        "J06.9": 0.55,  # Upper respiratory infection
        "J20.9": 0.30,  # Bronchitis
        "J45.9": 0.15,  # Asthma
    }
    
    neural_predictions = [
        {
            "condition_code": "J06.9",
            "condition_name": "Upper Respiratory Infection",
            "probability": 0.6,
        },
        {
            "condition_code": "J20.9",
            "condition_name": "Acute Bronchitis",
            "probability": 0.3,
        },
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.1},
    ]
    
    patient_symptoms = [
        "mild cough",
        "runny nose",
        "sore throat",
        "mild fatigue",
        "low-grade fever",
    ]
    
    print("\n📋 Patient Symptoms:")
    for symptom in patient_symptoms:
        print(f"   • {symptom}")
    
    print("\n🔬 Orchestrating Diagnosis...\n")
    
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs=bayesian_probs,
        neural_predictions=neural_predictions,
        patient_symptoms=patient_symptoms,
        top_k=3
    )
    
    print("📊 Differential Diagnosis (Top 3):\n")
    for i, diagnosis in enumerate(diagnoses, 1):
        print(f"{i}. {diagnosis.condition_name} ({diagnosis.condition_code})")
        print(f"   • Probability: {diagnosis.probability:.1%}")
        print(f"   • Escalation: {diagnosis.escalation_level.upper()}")
        if diagnosis.red_flags:
            print(f"   • 🚨 Red Flags Detected: {', '.join(diagnosis.red_flags)}")
        print(f"   • Reasoning: {diagnosis.reasoning}")
        print()
    
    top_diagnosis = diagnoses[0]
    guidance = orchestrator.generate_guidance_text(top_diagnosis.escalation_level)
    print("📢 Recommended Action:")
    print(f"   {guidance}")
    print()


def demo_routine_case():
    """Demonstrate orchestration for a routine case with low probabilities."""
    print("=" * 70)
    print("SCENARIO 3: Routine Case - Mild Symptoms, Low Certainty")
    print("=" * 70)
    
    orchestrator = DiagnosisOrchestrator()
    
    bayesian_probs = {
        "J06.9": 0.35,  # Upper respiratory infection
        "J20.9": 0.30,  # Bronchitis
        "J45.9": 0.20,  # Asthma
        "J44.9": 0.15,  # COPD
    }
    
    neural_predictions = [
        {
            "condition_code": "J06.9",
            "condition_name": "Upper Respiratory Infection",
            "probability": 0.4,
        },
        {
            "condition_code": "J20.9",
            "condition_name": "Acute Bronchitis",
            "probability": 0.3,
        },
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.2},
        {"condition_code": "J44.9", "condition_name": "COPD", "probability": 0.1},
    ]
    
    patient_symptoms = [
        "occasional cough",
        "mild congestion",
    ]
    
    print("\n📋 Patient Symptoms:")
    for symptom in patient_symptoms:
        print(f"   • {symptom}")
    
    print("\n🔬 Orchestrating Diagnosis...\n")
    
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs=bayesian_probs,
        neural_predictions=neural_predictions,
        patient_symptoms=patient_symptoms,
        top_k=4
    )
    
    print("📊 Differential Diagnosis (Top 4):\n")
    for i, diagnosis in enumerate(diagnoses, 1):
        print(f"{i}. {diagnosis.condition_name} ({diagnosis.condition_code})")
        print(f"   • Probability: {diagnosis.probability:.1%}")
        print(f"   • Escalation: {diagnosis.escalation_level.upper()}")
        if diagnosis.red_flags:
            print(f"   • 🚨 Red Flags Detected: {', '.join(diagnosis.red_flags)}")
        print(f"   • Reasoning: {diagnosis.reasoning}")
        print()
    
    top_diagnosis = diagnoses[0]
    guidance = orchestrator.generate_guidance_text(top_diagnosis.escalation_level)
    print("📢 Recommended Action:")
    print(f"   {guidance}")
    print()


def demo_custom_weights():
    """Demonstrate using custom ensemble weights."""
    print("=" * 70)
    print("SCENARIO 4: Custom Ensemble Weights")
    print("=" * 70)
    
    orchestrator = DiagnosisOrchestrator()
    
    bayesian_probs = {"J45.9": 0.7, "J18.9": 0.3}
    
    neural_predictions = [
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.4},
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.6},
    ]
    
    patient_symptoms = ["cough", "wheezing"]
    
    print("\n🔬 Default weights (0.6 neural, 0.4 Bayesian):")
    combined_default = orchestrator.combine_predictions(
        bayesian_probs, neural_predictions
    )
    for code, prob in sorted(
        combined_default.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"   {code}: {prob:.1%}")
    
    print("\n🔬 Custom weights (0.3 neural, 0.7 Bayesian):")
    combined_custom = orchestrator.combine_predictions(
        bayesian_probs, neural_predictions, neural_weight=0.3, bayesian_weight=0.7
    )
    for code, prob in sorted(
        combined_custom.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"   {code}: {prob:.1%}")
    
    print("\n💡 Note: Adjusting weights allows you to prioritize either the")
    print("   Bayesian prior (from dialogue) or neural predictions (from text).")
    print()


if __name__ == "__main__":
    print("\n🏥 PHAITA Diagnosis Orchestrator Demo")
    print("=" * 70)
    print(
        "This demo shows how the orchestrator combines Bayesian and neural"
    )
    print("predictions, detects red-flags, and determines escalation levels.")
    print()
    
    demo_emergency_case()
    demo_urgent_case()
    demo_routine_case()
    demo_custom_weights()
    
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)

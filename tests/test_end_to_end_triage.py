"""End-to-end integration test for complete triage workflow.

Simulates: Patient complaint → Dialogue questions → Final diagnosis → Escalation guidance
"""

from phaita.conversation.dialogue_engine import DialogueEngine
from phaita.models.discriminator import DiagnosisDiscriminator
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator
from phaita.models.generator import SymptomGenerator, ComplaintGenerator


def test_complete_triage_workflow_asthma():
    """Test complete triage workflow for asthma case."""
    print("\n🏥 Testing Complete Asthma Triage Workflow")
    print("=" * 60)
    
    # Step 1: Generate synthetic patient complaint
    print("\n1. Generating patient complaint...")
    symptom_gen = SymptomGenerator()
    complaint_gen = ComplaintGenerator()  # ML-first, falls back to template mode if unavailable
    
    presentation = symptom_gen.generate_symptoms("J45.9")  # Asthma
    presentation = complaint_gen.generate_complaint(presentation=presentation)
    complaint = presentation.complaint_text
    
    print(f"   Patient: \"{complaint}\"")
    print(f"   True condition: Asthma (J45.9)")
    print(f"   Actual symptoms: {', '.join(presentation.symptoms[:3])}")
    
    # Step 2: Initialize dialogue engine
    print("\n2. Initializing dialogue engine...")
    dialogue = DialogueEngine(max_turns=5, confidence_threshold=0.7)
    print(f"   ✓ Initial priors: uniform over 10 conditions")
    
    # Step 3: Simulate multi-turn conversation
    print("\n3. Conducting diagnostic interview...")
    turn = 0
    while not dialogue.should_terminate() and turn < 5:
        turn += 1
        question = dialogue.select_next_question()
        
        if question is None:
            print(f"   Turn {turn}: No more questions")
            break
        
        # Simulate patient response based on actual symptoms
        is_present = question in presentation.symptoms
        response = "Yes" if is_present else "No"
        
        print(f"   Turn {turn}: Do you have {question}?")
        print(f"            Response: {response}")
        
        dialogue.answer_question(question, is_present)
        
        # Show top diagnosis after each turn
        current_diff = dialogue.get_differential_diagnosis(top_n=3)
        top = current_diff[0]
        print(f"            Current top: {top['name']} (P={top['probability']:.3f})")
    
    print(f"\n   ✓ Conversation completed after {turn} turns")
    
    # Step 4: Get neural network predictions
    print("\n4. Running neural network diagnosis...")
    # Note: In a real deployment, this would use DiagnosisDiscriminator(use_pretrained=True)
    # For offline testing, we use mock predictions that simulate the discriminator output
    print(f"   ⚠️  Using mock neural predictions (offline mode)")
    neural_predictions = [
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.6},
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.2},
        {"condition_code": "J44.9", "condition_name": "COPD", "probability": 0.1},
        {"condition_code": "J06.9", "condition_name": "Upper respiratory infection", "probability": 0.05},
        {"condition_code": "J20.9", "condition_name": "Acute bronchitis", "probability": 0.05},
    ]
    print(f"   Mock neural top prediction: {neural_predictions[0]['condition_name']}")
    print(f"   Mock neural confidence: {neural_predictions[0]['probability']:.3f}")
    
    # Step 5: Orchestrate final diagnosis
    print("\n5. Orchestrating final diagnosis with red-flag detection...")
    orchestrator = DiagnosisOrchestrator()
    
    bayesian_probs = dialogue.state.differential_probabilities
    patient_symptoms = list(dialogue.state.confirmed_symptoms)
    
    final_diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs=bayesian_probs,
        neural_predictions=neural_predictions,
        patient_symptoms=patient_symptoms,
        top_k=5
    )
    
    # Step 6: Generate guidance
    print("\n6. Final Differential Diagnosis:")
    print("   " + "-" * 56)
    for i, diagnosis in enumerate(final_diagnoses, 1):
        escalation_emoji = {
            'emergency': '🚑',
            'urgent': '⚠️',
            'routine': '📋'
        }[diagnosis.escalation_level]
        
        print(f"   {i}. {diagnosis.condition_name:25s} {diagnosis.probability:5.1%}  {escalation_emoji} {diagnosis.escalation_level}")
        if diagnosis.red_flags:
            print(f"      Red-flags: {', '.join(diagnosis.red_flags)}")
    
    guidance = orchestrator.generate_guidance_text(final_diagnoses[0].escalation_level)
    print(f"\n   Guidance: {guidance[:100]}...")
    
    # Verification
    print("\n7. Verification:")
    top_diagnosis = final_diagnoses[0]
    
    # Check asthma is in top 3
    top_3_codes = [d.condition_code for d in final_diagnoses[:3]]
    assert "J45.9" in top_3_codes, f"Asthma should be in top 3, got {top_3_codes}"
    print(f"   ✓ Asthma correctly identified in top 3")
    
    # Check escalation is appropriate (asthma without red-flags should be urgent or routine)
    assert top_diagnosis.escalation_level in ['urgent', 'routine'], (
        f"Expected urgent/routine for asthma without red-flags, got {top_diagnosis.escalation_level}"
    )
    print(f"   ✓ Appropriate escalation level: {top_diagnosis.escalation_level}")
    
    # Check guidance text exists
    assert len(guidance) > 50, "Guidance text should be descriptive"
    print(f"   ✓ Guidance text generated ({len(guidance)} characters)")
    
    print("\n✅ Complete triage workflow successful!")
    return True


def test_complete_triage_workflow_emergency():
    """Test complete triage workflow for emergency condition with red-flags."""
    print("\n🚑 Testing Emergency Triage Workflow (Pneumothorax)")
    print("=" * 60)
    
    # Simulate emergency presentation with red-flags
    complaint = "Sudden severe chest pain, can't breathe, very anxious"
    
    print(f"\n1. Emergency Complaint:")
    print(f"   Patient: \"{complaint}\"")
    print(f"   Simulated condition: Pneumothorax (J93.0)")
    
    # Initialize components
    dialogue = DialogueEngine(max_turns=3, confidence_threshold=0.8)
    orchestrator = DiagnosisOrchestrator()
    
    # Simulate quick dialogue (emergency should terminate quickly)
    print("\n2. Brief diagnostic interview (emergency triage)...")
    
    # Simulate confirming critical symptoms
    dialogue.answer_question("chest_pain", present=True)
    dialogue.answer_question("dyspnea", present=True)
    dialogue.answer_question("sudden_onset", present=True)
    
    print(f"   ✓ Confirmed critical symptoms in {dialogue.state.turn_count} turns")
    
    # Get predictions
    # Note: In a real deployment, this would use DiagnosisDiscriminator(use_pretrained=True)
    # For offline testing, we use mock predictions that simulate the discriminator output
    print(f"   ⚠️  Using mock neural predictions (offline mode)")
    neural_predictions = [
        {"condition_code": "J93.0", "condition_name": "Pneumothorax", "probability": 0.5},
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.2},
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.15},
        {"condition_code": "J81.0", "condition_name": "Acute pulmonary edema", "probability": 0.1},
        {"condition_code": "J44.9", "condition_name": "COPD", "probability": 0.05},
    ]
    
    bayesian_probs = dialogue.state.differential_probabilities
    
    # Emergency symptoms - use actual red-flag symptoms from red_flags.yaml
    patient_symptoms = [
        "sudden_stabbing_chest_pain",  # red-flag for J93.0
        "severe_breathing_difficulty",  # red-flag for J93.0
        "sudden_onset",
    ]
    
    # Orchestrate diagnosis
    final_diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs=bayesian_probs,
        neural_predictions=neural_predictions,
        patient_symptoms=patient_symptoms,
        top_k=3
    )
    
    print("\n3. Final Diagnosis:")
    top = final_diagnoses[0]
    print(f"   Top: {top.condition_name} ({top.probability:.1%})")
    print(f"   Escalation: 🚑 {top.escalation_level}")
    if top.red_flags:
        print(f"   Red-flags: {', '.join(top.red_flags)}")
    
    # Verification
    print("\n4. Verification:")
    assert top.escalation_level == 'emergency', (
        f"Expected emergency escalation, got {top.escalation_level}"
    )
    print(f"   ✓ Emergency escalation triggered")
    
    guidance = orchestrator.generate_guidance_text(top.escalation_level)
    assert any(keyword in guidance.lower() for keyword in ['911', 'emergency', 'immediately']), (
        "Emergency guidance should mention 911/emergency/immediately"
    )
    print(f"   ✓ Emergency guidance appropriate")
    
    print("\n✅ Emergency triage workflow successful!")
    return True


if __name__ == "__main__":
    print("🧪 End-to-End Triage Integration Tests")
    print("=" * 60)
    
    try:
        test_complete_triage_workflow_asthma()
        print("\n" + "=" * 60)
        test_complete_triage_workflow_emergency()
        
        print("\n" + "=" * 60)
        print("🎉 All end-to-end tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

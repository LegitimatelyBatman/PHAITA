import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""Demo script showing the DialogueEngine with Bayesian belief updating in action."""

from phaita import DialogueEngine


def format_symptom_name(symptom: str) -> str:
    """Format symptom name for display."""
    return symptom.replace("_", " ").title()


def main():
    print("🏥 PHAITA Dialogue Engine Demo")
    print("=" * 60)
    print("\nThis demo shows the Bayesian belief updating dialogue engine")
    print("for medical triage conversations.\n")
    
    # Create the engine
    engine = DialogueEngine(max_turns=10, confidence_threshold=0.7)
    print(f"✅ Initialized engine with {len(engine.conditions)} respiratory conditions")
    print(f"   Max turns: {engine.max_turns}")
    print(f"   Confidence threshold: {engine.state.confidence_threshold}\n")
    
    # Show initial state
    print("📊 Initial State (Uniform Prior):")
    initial_diff = engine.get_differential_diagnosis(top_n=3)
    for i, entry in enumerate(initial_diff, 1):
        print(f"   {i}. {entry['name']:20s} P={entry['probability']:.3f}")
    print()
    
    # Simulate a conversation about asthma
    print("💬 Simulating Diagnostic Conversation")
    print("-" * 60)
    
    # Turn 1: Ask about wheezing
    turn = 1
    question = engine.select_next_question()
    print(f"\nTurn {turn}:")
    print(f"   ❓ Question: Do you have {format_symptom_name(question)}?")
    print(f"   💭 Patient: Yes, I do.")
    
    engine.answer_question(question, present=True)
    print(f"   ✓ Updated beliefs with: {format_symptom_name(question)} = PRESENT")
    
    diff = engine.get_differential_diagnosis(top_n=3)
    print(f"   📊 Top diagnoses:")
    for i, entry in enumerate(diff, 1):
        print(f"      {i}. {entry['name']:20s} P={entry['probability']:.3f}")
    print(f"   🔍 Should terminate? {engine.should_terminate()}")
    
    # Turn 2: Ask about another symptom
    if not engine.should_terminate():
        turn = 2
        question = engine.select_next_question()
        print(f"\nTurn {turn}:")
        print(f"   ❓ Question: Do you have {format_symptom_name(question)}?")
        
        # Simulate a positive answer for breathing-related symptoms
        if "breath" in question.lower() or "dyspnea" in question.lower():
            print(f"   💭 Patient: Yes, it's been difficult to breathe.")
            engine.answer_question(question, present=True)
            print(f"   ✓ Updated beliefs with: {format_symptom_name(question)} = PRESENT")
        else:
            print(f"   💭 Patient: No, I don't have that.")
            engine.answer_question(question, present=False)
            print(f"   ✓ Updated beliefs with: {format_symptom_name(question)} = ABSENT")
        
        diff = engine.get_differential_diagnosis(top_n=3)
        print(f"   📊 Top diagnoses:")
        for i, entry in enumerate(diff, 1):
            print(f"      {i}. {entry['name']:20s} P={entry['probability']:.3f}")
        print(f"   🔍 Should terminate? {engine.should_terminate()}")
    
    # Turn 3: Continue if needed
    if not engine.should_terminate():
        turn = 3
        question = engine.select_next_question()
        print(f"\nTurn {turn}:")
        print(f"   ❓ Question: Do you have {format_symptom_name(question)}?")
        
        # Simulate answer based on symptom
        if "tight" in question.lower() or "wheez" in question.lower():
            print(f"   💭 Patient: Yes, my chest feels tight.")
            engine.answer_question(question, present=True)
            print(f"   ✓ Updated beliefs with: {format_symptom_name(question)} = PRESENT")
        else:
            print(f"   💭 Patient: No.")
            engine.answer_question(question, present=False)
            print(f"   ✓ Updated beliefs with: {format_symptom_name(question)} = ABSENT")
        
        diff = engine.get_differential_diagnosis(top_n=3)
        print(f"   📊 Top diagnoses:")
        for i, entry in enumerate(diff, 1):
            print(f"      {i}. {entry['name']:20s} P={entry['probability']:.3f}")
        print(f"   🔍 Should terminate? {engine.should_terminate()}")
    
    # Final diagnosis
    print("\n" + "=" * 60)
    print("📋 Final Differential Diagnosis")
    print("=" * 60)
    
    final_diff = engine.get_differential_diagnosis(top_n=5, min_probability=0.01)
    
    print(f"\n🔍 After {engine.state.turn_count} turns:")
    print(f"   Confirmed symptoms: {', '.join(format_symptom_name(s) for s in engine.state.confirmed_symptoms)}")
    print(f"   Denied symptoms: {', '.join(format_symptom_name(s) for s in engine.state.denied_symptoms)}")
    
    print(f"\n📊 Top {len(final_diff)} Possible Conditions:\n")
    for i, entry in enumerate(final_diff, 1):
        confidence = "High" if entry['probability'] > 0.5 else "Medium" if entry['probability'] > 0.2 else "Low"
        bar = "█" * int(entry['probability'] * 40)
        print(f"   {i}. {entry['name']:25s} {entry['probability']:5.1%}  {confidence:7s}  {bar}")
    
    print("\n✨ Demo complete!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Bayesian belief updating with each symptom")
    print("  ✓ Information gain-based question selection")
    print("  ✓ Automatic termination when confidence is high")
    print("  ✓ Differential diagnosis ranked by probability")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Demo script for PHAITA deep learning models.
Shows the new DeBERTa + GNN discriminator and LLM-based generator.
"""

import torch

from phaita.data.preprocessing import DataPreprocessor
from phaita.data.synthetic_generator import SyntheticDataGenerator
from phaita.models.discriminator import DiagnosisDiscriminator
from phaita.models.generator import ComplaintGenerator, SymptomGenerator
from phaita.models.question_generator import QuestionGenerator
from phaita.triage import enrich_differential_with_guidance, format_differential_report


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_discriminator():
    """Demonstrate the discriminator with DeBERTa + GNN."""
    print_section("1. DiagnosisDiscriminator (DeBERTa + GNN)")
    
    # Initialize (ML-first, will fall back to lightweight mode if models unavailable)
    print("\n📊 Initializing discriminator...")
    disc = DiagnosisDiscriminator()  # Attempts ML first, falls back gracefully
    
    param_count = sum(p.numel() for p in disc.parameters())
    print(f"✓ Loaded discriminator with {param_count:,} parameters")
    
    # Test predictions
    print("\n🔍 Testing predictions on sample complaints:")
    complaints = [
        "I have a bad cough and chest pain that won't go away",
        "I can't breathe and I'm wheezing constantly",
        "Running a high fever with fatigue and body aches"
    ]
    
    predictions = disc.predict_diagnosis(complaints, top_k=3)

    for complaint, differential in zip(complaints, predictions):
        enriched = enrich_differential_with_guidance(differential)
        primary = enriched[0]
        report = format_differential_report(differential)
        print(f"\n  Complaint: {complaint[:50]}...")
        print(f"  Primary: {primary['condition_name']} ({primary['condition_code']}) "
              f"prob={primary['probability']:.2%}")
        print("  Differential:")
        for line in report.splitlines():
            print(f"    {line}")
    
    # Show detailed explanation
    print("\n📝 Detailed explanation for first complaint:")
    code, confidence, explanation = disc.predict_with_explanation(complaints[0])
    print(f"  Condition: {explanation['condition_name']}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Reasoning: {explanation['reasoning']}")
    print(f"  Top 3 predictions:")
    for pred_code, pred_conf in explanation['top_3_predictions']:
        print(f"    - {pred_code}: {pred_conf:.2%}")


def demo_generator():
    """Demonstrate the complaint generator."""
    print_section("2. ComplaintGenerator (Template/LLM-based)")
    
    print("\n📝 Initializing generators...")
    symptom_gen = SymptomGenerator()
    complaint_gen = ComplaintGenerator()  # ML-first, falls back to template mode if needed
    
    param_count = sum(p.numel() for p in complaint_gen.parameters())
    mode_msg = "template mode" if complaint_gen.template_mode else "ML mode"
    print(f"✓ Loaded generator with {param_count:,} parameters ({mode_msg})")
    
    print("\n🎲 Generating complaints for different conditions:")
    conditions = [
        ("J45.9", "Asthma"),
        ("J18.9", "Pneumonia"),
        ("J44.9", "COPD")
    ]
    
    for code, name in conditions:
        presentation = symptom_gen.generate_symptoms(code)
        presentation = complaint_gen.generate_complaint(presentation=presentation)
        print(f"\n  {name} ({code}):")
        print(f"    Symptoms: {', '.join(presentation.symptoms[:3])}")
        print(f"    Complaint: {presentation.complaint_text}")
        probs_preview = list(presentation.symptom_probabilities.items())[:2]
        if probs_preview:
            print(
                "    Probabilities: "
                + ", ".join(f"{symptom}={prob:.2f}" for symptom, prob in probs_preview)
            )


def demo_question_generator():
    """Demonstrate the question generator."""
    print_section("3. QuestionGenerator (Dynamic Triage)")
    
    print("\n❓ Initializing question generator...")
    qgen = QuestionGenerator()  # ML-first, falls back to template mode if needed
    
    param_count = sum(p.numel() for p in qgen.parameters())
    mode_msg = "template mode" if qgen.template_mode else "ML mode"
    print(f"✓ Loaded question generator with {param_count:,} parameters ({mode_msg})")
    
    print("\n💬 Generating clarifying questions:")
    symptom_sets = [
        ["cough", "fever"],
        ["dyspnea", "chest_pain"],
        ["wheezing", "tight_chest"]
    ]
    
    for symptoms in symptom_sets:
        question = qgen.generate_clarifying_question(symptoms)
        print(f"\n  Symptoms: {', '.join(symptoms)}")
        print(f"  Question: {question}")


def demo_preprocessing():
    """Demonstrate the data preprocessor."""
    print_section("4. DataPreprocessor (Medical Term Extraction)")
    
    print("\n🔧 Initializing preprocessor...")
    preprocessor = DataPreprocessor()
    print("✓ Loaded preprocessor with medical term dictionary")
    
    print("\n🔍 Testing medical term extraction:")
    complaints = [
        "I have a terrible cough and can't breathe",
        "My chest hurts when I breathe deeply",
        "I've been wheezy and feeling breathless"
    ]
    
    for complaint in complaints:
        terms = preprocessor.extract_medical_terms(complaint)
        print(f"\n  Complaint: {complaint}")
        print(f"  Extracted terms:")
        for term, category in terms[:5]:  # Show top 5
            print(f"    - {term} ({category})")


def demo_integrated_pipeline():
    """Demonstrate the complete pipeline."""
    print_section("5. Integrated Pipeline (End-to-End)")
    
    print("\n🔄 Running complete diagnostic pipeline:")
    
    # Step 1: Generate synthetic complaint
    print("\n  Step 1: Generate patient complaint")
    symptom_gen = SymptomGenerator()
    complaint_gen = ComplaintGenerator()  # ML-first, falls back if needed
    
    presentation = symptom_gen.generate_symptoms("J45.9")
    presentation = complaint_gen.generate_complaint(presentation=presentation)
    print(f"    Generated: {presentation.complaint_text}")

    # Step 2: Preprocess
    print("\n  Step 2: Preprocess and extract terms")
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_complaints([presentation.complaint_text])[0]
    terms = preprocessor.extract_medical_terms(processed)
    print(f"    Processed: {processed}")
    print(f"    Terms found: {len(terms)}")

    # Step 3: Predict diagnosis
    print("\n  Step 3: Predict diagnosis")
    disc = DiagnosisDiscriminator()  # ML-first, falls back if needed
    code, confidence, explanation = disc.predict_with_explanation(
        presentation.complaint_text
    )
    print(f"    Predicted: {explanation['condition_name']} ({code})")
    print(f"    Confidence: {confidence:.2%}")

    # Step 4: Generate follow-up question
    print("\n  Step 4: Generate follow-up question")
    qgen = QuestionGenerator()  # ML-first, falls back if needed
    question = qgen.generate_clarifying_question(presentation.symptoms)
    print(f"    Question: {question}")
    
    print("\n✅ Pipeline complete!")


def demo_model_statistics():
    """Show model statistics and architecture details."""
    print_section("6. Model Architecture & Statistics")
    
    print("\n📊 Model Details:")
    
    # Discriminator
    disc = DiagnosisDiscriminator()  # ML-first, falls back if needed
    disc_params = sum(p.numel() for p in disc.parameters())
    disc_mode = "lightweight" if not disc.use_pretrained else "ML"
    print(f"\n  DiagnosisDiscriminator:")
    print(f"    Parameters: {disc_params:,}")
    print(f"    Mode: {disc_mode}")
    print(f"    Architecture: {'Keyword-based' if not disc.use_pretrained else 'DeBERTa (768d) + GNN (256d) + Fusion'}")
    print(f"    Outputs: 10 diagnosis classes + real/fake score")
    
    # Generator
    gen = ComplaintGenerator()  # ML-first, falls back if needed
    gen_params = sum(p.numel() for p in gen.parameters())
    gen_mode = "template" if gen.template_mode else "ML"
    print(f"\n  ComplaintGenerator:")
    print(f"    Parameters: {gen_params:,} ({gen_mode} mode)")
    if gen.template_mode:
        print(f"    Note: Can use Mistral-7B (~7B params) with internet connection")
        print(f"    Templates: 8 grammatically-correct patterns")
    else:
        print(f"    Model: Mistral-7B-Instruct")
    
    # Question Generator
    qgen = QuestionGenerator()  # ML-first, falls back if needed
    qgen_params = sum(p.numel() for p in qgen.parameters())
    qgen_mode = "template" if qgen.template_mode else "ML"
    print(f"\n  QuestionGenerator:")
    print(f"    Parameters: {qgen_params:,}")
    print(f"    Mode: {qgen_mode}")
    
    # Total
    total_params = disc_params + gen_params + qgen_params
    print(f"\n  Total: {total_params:,} parameters")
    if gen.template_mode or not disc.use_pretrained:
        print(f"  Note: Running in fallback mode. With ML models: ~7.004M parameters")
    
    print("\n💾 Memory Usage:")
    print(f"    CPU mode (fallback): ~100 MB")
    print(f"    GPU mode (pretrained DeBERTa): ~1.5 GB")
    print(f"    GPU mode (+ Mistral-7B 4-bit): ~5 GB")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  PHAITA Deep Learning Models Demo")
    print("  Real DeBERTa + GNN + Mistral-7B Integration")
    print("=" * 70)
    
    print("\nℹ️  Note: Models attempt ML-first, fall back gracefully if unavailable")
    print("   For full ML experience, ensure:")
    print("   - Internet connection for model downloads")
    print("   - GPU with ~5GB VRAM (or 16GB+ RAM for CPU)")
    
    try:
        demo_discriminator()
        demo_generator()
        demo_question_generator()
        demo_preprocessing()
        demo_integrated_pipeline()
        demo_model_statistics()
        
        print("\n" + "=" * 70)
        print("  ✅ All demos completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Try with pretrained models: discriminator = DiagnosisDiscriminator(use_pretrained=True)")
        print("  2. Train on your data: See DEEP_LEARNING_GUIDE.md")
        print("  3. Evaluate performance: Use metrics from phaita.utils.metrics")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

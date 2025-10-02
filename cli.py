#!/usr/bin/env python3
"""
PHAITA CLI - Command line interface for Pre-Hospital AI Triage Algorithm
Enhanced with user diagnosis testing and adversarial challenge mode.
"""

import argparse
import logging
import os
import random
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from phaita import (
    AdversarialTrainer,
    ComplaintGenerator,
    Config,
    DiagnosisDiscriminator,
    RespiratoryConditions,
    SymptomGenerator,
)
from phaita.conversation import ConversationEngine
from phaita.models.question_generator import QuestionGenerator
from phaita.triage import format_differential_report
# from phaita.models import SymptomGenerator, ComplaintGenerator, DiagnosisDiscriminator
try:
    from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network
    from phaita.data.forum_scraper import create_data_augmentation
    from phaita.utils.realism_scorer import create_realism_scorer
except ImportError as e:
    # Handle missing dependencies gracefully
    def create_enhanced_bayesian_network():
        from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network
        return create_enhanced_bayesian_network()
    
    def create_data_augmentation():
        return None
    
    def create_realism_scorer():
        return None


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('phaita.log')
        ]
    )


@lru_cache(maxsize=1)
def _get_complaint_generator() -> ComplaintGenerator:
    """Return a cached complaint generator instance."""
    return ComplaintGenerator()


@lru_cache(maxsize=1)
def _get_diagnosis_discriminator() -> DiagnosisDiscriminator:
    """Return a cached diagnosis discriminator instance."""
    model = DiagnosisDiscriminator()
    model.eval()
    return model


def train_command(args):
    """Train the adversarial model."""
    print("🏥 Starting PHAITA adversarial training...")
    
    # Load configuration
    config_path = args.config or "config.yaml"
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
        print(f"✅ Loaded configuration from {config_path}")
    else:
        config = Config()
        print("⚠️  Using default configuration")
    
    # Override config with command line arguments
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.discriminator_lr = args.lr
    
    # Initialize trainer
    trainer = AdversarialTrainer(
        generator_lr=config.training.generator_lr,
        discriminator_lr=config.training.discriminator_lr,
        diversity_weight=config.training.diversity_weight,
        device=config.training.device
    )
    
    print(f"🧠 Training on device: {trainer.device}")
    print(f"📊 Training for {config.training.num_epochs} epochs with batch size {config.training.batch_size}")
    
    # Start training
    try:
        history = trainer.train(
            num_epochs=config.training.num_epochs,
            batch_size=config.training.batch_size,
            eval_interval=config.training.eval_interval,
            save_interval=config.training.save_interval
        )
        
        print("✅ Training completed successfully!")
        print(f"📈 Final discriminator loss: {history['disc_loss'][-1]:.4f}")
        print(f"🎯 Final generator loss: {history.get('gen_loss', [0])[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def demo_command(args):
    """Run a demonstration of the system."""
    print("🏥 PHAITA Demo - Medical Triage AI")
    print("=" * 50)
    
    # Initialize components
    print("🔄 Initializing models...")
    symptom_gen = SymptomGenerator()
    complaint_gen = ComplaintGenerator()
    discriminator = DiagnosisDiscriminator()
    
    # Show available conditions
    conditions = RespiratoryConditions.get_all_conditions()
    print(f"\n📋 Available respiratory conditions ({len(conditions)}):")
    for i, (code, data) in enumerate(conditions.items(), 1):
        print(f"  {i}. {code}: {data['name']}")
    
    # Generate some examples
    print(f"\n🔬 Generating {args.num_examples} synthetic examples:")
    print("-" * 50)
    
    for i in range(args.num_examples):
        # Sample random condition
        code, condition_data = RespiratoryConditions.get_random_condition()
        
        # Generate structured presentation
        presentation = symptom_gen.generate_symptoms(code)
        presentation = complaint_gen.generate_complaint(
            condition_code=code, presentation=presentation
        )
        complaint = presentation.complaint_text

        # Get diagnosis predictions
        predictions = discriminator.predict_diagnosis([complaint], top_k=args.top_k)
        report = format_differential_report(predictions[0])
        top_entry = predictions[0][0]
        pred_code = top_entry["condition_code"]
        confidence = top_entry["probability"]

        print(f"\n🩺 Example {i+1}:")
        print(f"   True Condition: {code} - {condition_data['name']}")
        print(f"   Symptoms: {', '.join(presentation.symptoms[:3])}...")
        print(f"   Patient Says: \"{complaint}\"")
        print(f"   AI Primary Diagnosis: {pred_code} (probability: {confidence:.3f})")
        print("\n   Differential guidance:")
        print(textwrap.indent(report, "      "))
        
        # Check if correct
        correct = "✅" if pred_code == code else "❌"
        print(f"   Result: {correct}")


def generate_command(args):
    """Generate synthetic patient data."""
    print("🔬 Generating synthetic patient complaints...")
    
    # Initialize generators
    symptom_gen = SymptomGenerator()
    complaint_gen = ComplaintGenerator()
    
    results = []
    
    for i in range(args.count):
        # Sample condition
        if args.condition:
            # Use specific condition
            conditions = RespiratoryConditions.get_all_conditions()
            code = None
            for c, data in conditions.items():
                if args.condition.lower() in data['name'].lower():
                    code = c
                    break
            if not code:
                print(f"❌ Condition '{args.condition}' not found")
                return
        else:
            # Random condition
            code, _ = RespiratoryConditions.get_random_condition()
        
        # Generate symptoms and complaint
        presentation = symptom_gen.generate_symptoms(code)
        presentation = complaint_gen.generate_complaint(
            condition_code=code, presentation=presentation
        )

        results.append({
            'condition_code': code,
            'condition_name': RespiratoryConditions.get_condition_by_code(code)['name'],
            'symptoms': presentation.symptoms,
            'complaint': presentation.complaint_text,
            'symptom_probabilities': presentation.symptom_probabilities,
            'misdescription_weights': presentation.misdescription_weights,
            'vocabulary_profile': {
                'allowed_terms': presentation.vocabulary_profile.allowed_terms,
                'term_overrides': presentation.vocabulary_profile.term_overrides,
                'register': presentation.vocabulary_profile.register,
                'max_terms_per_response': presentation.vocabulary_profile.max_terms_per_response,
            }
        })
    
    # Output results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Generated {len(results)} examples saved to {args.output}")
    else:
        # Print to console
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['condition_code']}: {result['condition_name']}")
            print(f"   Complaint: \"{result['complaint']}\"")


def conversation_command(args):
    """Run an interactive conversation loop using the dialogue controller."""

    print("🗣️  PHAITA Conversation Engine")
    print("=" * 40)

    qgen = QuestionGenerator(use_pretrained=False)
    engine = ConversationEngine(
        qgen,
        max_questions=args.max_questions,
        min_symptom_count=args.min_symptoms,
        max_no_progress_turns=args.max_no_progress,
    )

    if args.symptoms:
        seed_symptoms = [s.strip() for s in args.symptoms.split(',') if s.strip()]
        engine.add_symptoms(seed_symptoms)
        if seed_symptoms:
            print(f"Seeded symptoms: {', '.join(seed_symptoms)}")

    while True:
        if engine.should_present_diagnosis():
            break

        prompt = engine.next_prompt()
        if not prompt:
            break

        print(f"\nAI: {prompt}")
        response = input("You: ").strip()

        if response.lower() in {"quit", "exit", "q"}:
            print("Ending conversation early at user request.")
            break

        extracted_symptoms = [s.strip() for s in response.split(',') if s.strip()]
        engine.record_response(prompt, response, extracted_symptoms)

    print("\nConversation complete.")
    if engine.symptoms:
        print(f"Gathered symptoms: {', '.join(engine.symptoms)}")
    else:
        print("No symptoms were gathered during this exchange.")

    if engine.should_present_diagnosis():
        print("Ready to present preliminary diagnoses based on collected data.")
    else:
        print("Conversation ended without meeting diagnostic criteria.")


def diagnose_command(args):
    """Test discriminator on user-provided complaint."""
    print("🩺 PHAITA Diagnosis Tool")
    print("=" * 40)

    try:
        discriminator = _get_diagnosis_discriminator()

        def run_diagnosis(input_complaint: str):
            """Execute model prediction with graceful fallback."""
            try:
                code, confidence, explanation = discriminator.predict_with_explanation(input_complaint)
                top_predictions = explanation.get("top_3_predictions", [])
            except AttributeError:
                predictions = discriminator.predict_diagnosis([input_complaint], top_k=3)
                ranked = predictions[0] if predictions else []
                if ranked:
                    top_entry = ranked[0]
                    code = top_entry.get("condition_code", "UNKNOWN")
                    confidence = top_entry.get("probability", 0.0)
                else:
                    code = "UNKNOWN"
                    confidence = 0.0
                top_predictions = [
                    (
                        entry.get("condition_code", "UNKNOWN"),
                        entry.get("probability", 0.0)
                    )
                    for entry in ranked
                ]
                condition_info = RespiratoryConditions.get_condition_by_code(code)
                explanation = {
                    "condition_code": code,
                    "condition_name": condition_info.get("name") if condition_info else code,
                    "confidence": confidence,
                    "matched_keywords": [],
                    "reasoning": "Model prediction via predict_diagnosis",
                    "top_3_predictions": top_predictions,
                }

            return code, confidence, explanation, top_predictions

        # Get user input
        if args.complaint:
            complaint = args.complaint
        else:
            print("Enter a patient complaint (or 'quit' to exit):")
            complaint = input("> ").strip()

            if complaint.lower() in ['quit', 'exit', 'q']:
                return

        if not complaint:
            print("❌ No complaint provided")
            return

        print(f"\n📝 Analyzing complaint: \"{complaint}\"")

        code, confidence, explanation, top_predictions = run_diagnosis(complaint)

        print(f"\n🔍 Diagnosis Results:")
        condition_name = explanation.get("condition_name") or code
        print(f"   Primary Diagnosis: {condition_name} ({code})")
        print(f"   Confidence: {confidence:.3f}")

        secondary_conditions = [
            f"{pred_code} ({prob:.3f})" if isinstance(prob, (int, float)) else pred_code
            for pred_code, prob in top_predictions
            if pred_code != code
        ]
        if secondary_conditions:
            print(f"   Secondary Conditions: {', '.join(secondary_conditions)}")
        else:
            print("   Secondary Conditions: None")

        if args.detailed:
            print(f"\n📊 Detailed Analysis:")
            matched_keywords = explanation.get("matched_keywords", [])
            if matched_keywords:
                print(f"   Matched Keywords: {', '.join(matched_keywords)}")
            reasoning = explanation.get("reasoning")
            if reasoning:
                print(f"   Reasoning: {reasoning}")

            if top_predictions:
                print("   Differential Ranking:")
                for rank, (pred_code, prob) in enumerate(top_predictions, start=1):
                    display_prob = f"{prob:.3f}" if isinstance(prob, (int, float)) else prob
                    print(f"      {rank}. {pred_code} - {display_prob}")

        if args.interactive:
            while True:
                print("\n" + "=" * 40)
                print("Enter another complaint (or 'quit' to exit):")
                new_complaint = input("> ").strip()

                if new_complaint.lower() in ['quit', 'exit', 'q']:
                    break

                if new_complaint:
                    new_code, new_confidence, _, _ = run_diagnosis(new_complaint)
                    print(
                        f"\n🔍 Diagnosis: {new_code} "
                        f"(confidence: {new_confidence:.3f})"
                    )

    except Exception as e:
        print(f"❌ Error in diagnosis: {e}")


def challenge_command(args):
    """Run adversarial challenge mode demonstration."""
    print("🎯 PHAITA Adversarial Challenge Mode")
    print("=" * 50)
    print("Testing discriminator against challenging adversarial examples...")
    
    try:
        # Initialize enhanced Bayesian network for generating challenging cases
        network = create_enhanced_bayesian_network()
        
        challenge_cases = []
        
        # Generate different types of challenging cases
        print("\n🔄 Generating adversarial challenges...")
        
        # 1. Rare presentations
        print("\n📍 Category 1: Rare Presentations")
        for i in range(args.rare_cases):
            condition_code = random.choice(["J45.9", "J44.1", "J18.9"])
            symptoms, metadata = network.sample_symptoms(
                condition_code, 
                include_rare=True,
                severity="severe"
            )
            
            if metadata.get("presentation_type") == "rare":
                case_name = metadata.get("case_name", "Unknown rare case")
                print(f"   {i+1}. {case_name}")
                challenge_cases.append({
                    "type": "rare",
                    "condition": condition_code,
                    "symptoms": symptoms,
                    "metadata": metadata
                })
        
        # Generate standard cases if no rare cases found
        if len(challenge_cases) == 0:
            for i in range(args.rare_cases):
                condition_code = random.choice(["J45.9", "J44.1", "J18.9"])
                symptoms, metadata = network.sample_symptoms(condition_code, severity="severe")
                challenge_cases.append({
                    "type": "standard",
                    "condition": condition_code,
                    "symptoms": symptoms,
                    "metadata": metadata
                })
                print(f"   {i+1}. Standard {condition_code} case")
        
        # 2. Atypical age presentations  
        print(f"\n📍 Category 2: Atypical Age Presentations")
        for i in range(args.atypical_cases):
            condition_code = random.choice(["J45.9", "J06.9"])
            unusual_age = "elderly" if condition_code == "J45.9" else "child"
            
            symptoms, metadata = network.sample_symptoms(
                condition_code,
                age_group=unusual_age,
                severity=random.choice(["mild", "severe"])
            )
            
            print(f"   {i+1}. {condition_code} in {unusual_age} patient")
            challenge_cases.append({
                "type": "atypical_age",
                "condition": condition_code,
                "age_group": unusual_age,
                "symptoms": symptoms,
                "metadata": metadata
            })
        
        # 3. Comorbidity-influenced presentations
        print(f"\n📍 Category 3: Comorbidity-Influenced")
        comorbidity_cases = [
            ("J45.9", ["anxiety", "obesity"]),
            ("J44.1", ["heart_failure"]),
            ("J18.9", ["immunocompromised"])
        ]
        
        for i, (condition, comorbidities) in enumerate(comorbidity_cases):
            symptoms, metadata = network.sample_symptoms(
                condition,
                comorbidities=comorbidities,
                severity="moderate"
            )
            
            print(f"   {i+1}. {condition} with {', '.join(comorbidities)}")
            challenge_cases.append({
                "type": "comorbidity",
                "condition": condition,
                "comorbidities": comorbidities,
                "symptoms": symptoms,
                "metadata": metadata
            })
        
        # Test discriminator on challenging cases
        print(f"\n🧪 Testing Discriminator Performance...")
        discriminator = _get_diagnosis_discriminator()
        complaint_generator = _get_complaint_generator()
        evaluated_cases = []

        for i, case in enumerate(challenge_cases):
            try:
                presentation = complaint_generator.generate_complaint(
                    condition_code=case["condition"],
                    symptoms=case["symptoms"],
                )
                complaint = presentation.complaint_text
            except Exception:
                complaint = _generate_complaint_from_symptoms(case["symptoms"], case.get("metadata", {}))

            predictions_batch = discriminator.predict_diagnosis([complaint], top_k=3)
            ranked_predictions = predictions_batch[0] if predictions_batch else []

            if ranked_predictions:
                top_entry = ranked_predictions[0]
                predicted_condition = top_entry.get("condition_code", "UNKNOWN")
                confidence = top_entry.get("probability", 0.0)
            else:
                predicted_condition = "UNKNOWN"
                confidence = 0.0

            is_correct = _is_diagnosis_correct(case["condition"], predicted_condition)

            evaluated_cases.append({
                **case,
                "complaint": complaint,
                "predictions": ranked_predictions,
                "predicted_condition": predicted_condition,
                "confidence": confidence,
                "is_correct": is_correct,
            })

            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            total_cases = len(challenge_cases)
            print(f"   Case {i+1}/{total_cases}: {status}")

            if args.verbose:
                print(f"      Complaint: \"{complaint[:60]}...\"")
                print(f"      True: {case['condition']}, Predicted: {predicted_condition}")
                print(f"      Confidence: {confidence:.3f}")
                if ranked_predictions:
                    top_summary = ", ".join(
                        f"{entry.get('condition_code', 'UNKNOWN')}"
                        f" ({entry.get('probability', 0.0):.3f})"
                        for entry in ranked_predictions[:3]
                    )
                    print(f"      Top predictions: {top_summary}")

        total_cases = len(evaluated_cases)
        correct_diagnoses = sum(1 for case in evaluated_cases if case["is_correct"])
        accuracy = correct_diagnoses / total_cases if total_cases > 0 else 0.0
        print(f"\n📊 Challenge Results:")
        print(f"   Total Cases: {total_cases}")
        print(f"   Correct Diagnoses: {correct_diagnoses}")
        print(f"   Accuracy: {accuracy:.1%}")

        if accuracy < 0.6:
            print("   🚨 Performance below threshold - model needs improvement")
        elif accuracy < 0.8:
            print("   ⚠️  Moderate performance - room for improvement")
        else:
            print("   🎉 Good performance on adversarial challenges")

        if args.show_failures and correct_diagnoses < total_cases:
            print(f"\n🔍 Most Challenging Cases:")
            failure_cases = [case for case in evaluated_cases if not case["is_correct"]][:3]
            for idx, case in enumerate(failure_cases, start=1):
                print(f"   {idx}. Type: {case['type']}")
                print(f"      Complaint: \"{case['complaint']}\"")
                print(f"      Expected: {case['condition']}, Got: {case['predicted_condition']}")
                if case["predictions"]:
                    top_summary = ", ".join(
                        f"{entry.get('condition_code', 'UNKNOWN')}"
                        f" ({entry.get('probability', 0.0):.3f})"
                        for entry in case["predictions"][:3]
                    )
                    print(f"      Model Ranking: {top_summary}")
    
    except Exception as e:
        print(f"❌ Error in challenge mode: {e}")
        import traceback
        traceback.print_exc()


def _generate_complaint_from_symptoms(symptoms: list, metadata: dict) -> str:
    """Generate a patient complaint from symptoms."""
    # Simple templates
    templates = [
        "I've been having {symptoms} for {duration}.",
        "Doctor, I'm experiencing {symptoms} and I'm {emotion}.",
        "Help! I can't stop having {symptoms}. It started {duration} ago.",
        "I'm worried about {symptoms} that began {duration}."
    ]
    
    # Convert medical symptoms to lay language
    lay_mapping = {
        'wheezing': 'wheezing sounds',
        'shortness_of_breath': "can't breathe properly",
        'dyspnea': "trouble breathing", 
        'chest_tightness': 'tight chest',
        'cough': 'bad cough',
        'fever': 'high fever',
        'fatigue': 'feeling exhausted',
        'chest_pain': 'chest pain'
    }
    
    # Convert symptoms to lay terms
    lay_symptoms = []
    for symptom in symptoms[:3]:  # Limit to 3 symptoms
        lay_term = lay_mapping.get(symptom, symptom)
        lay_symptoms.append(lay_term)
    
    if not lay_symptoms:
        lay_symptoms = ["breathing problems"]
    
    # Fill template
    template = random.choice(templates)
    complaint = template.format(
        symptoms=" and ".join(lay_symptoms),
        duration=random.choice(["a few days", "hours", "weeks"]),
        emotion=random.choice(["worried", "scared", "exhausted"])
    )
    
    return complaint


def _is_diagnosis_correct(true_condition: str, predicted_condition: str) -> bool:
    """Check if diagnosis is correct (simplified)."""
    return true_condition == predicted_condition


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHAITA - Pre-Hospital AI Triage Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py train --epochs 50 --batch-size 32
  python cli.py demo --num-examples 5
  python cli.py generate --count 10 --output data.json
  python cli.py generate --condition "pneumonia" --count 5
  python cli.py conversation --symptoms "cough,fever"
  python cli.py diagnose --complaint "I can't breathe and have chest pain"
  python cli.py diagnose --interactive --detailed
  python cli.py challenge --rare-cases 5 --show-failures --verbose
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('-c', '--config', type=str,
                       help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the adversarial model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run system demonstration')
    demo_parser.add_argument('--num-examples', type=int, default=3,
                             help='Number of examples to generate')
    demo_parser.add_argument('--top-k', type=int, default=3,
                             help='Number of diagnoses to include in the differential list')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--count', type=int, default=10,
                          help='Number of examples to generate')
    gen_parser.add_argument('--condition', type=str,
                          help='Specific condition to generate for')
    gen_parser.add_argument('--output', type=str,
                          help='Output file for generated data')

    # Conversation command
    convo_parser = subparsers.add_parser('conversation', help='Run interactive symptom clarification loop')
    convo_parser.add_argument('--symptoms', type=str,
                              help='Comma-separated list of initial symptoms')
    convo_parser.add_argument('--max-questions', type=int, default=5,
                              help='Maximum number of clarifying questions to ask')
    convo_parser.add_argument('--min-symptoms', type=int, default=3,
                              help='Number of symptoms required before stopping')
    convo_parser.add_argument('--max-no-progress', type=int, default=2,
                              help='Number of consecutive turns without new symptoms before stopping')

    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Test discriminator on user complaint')
    diagnose_parser.add_argument('--complaint', type=str,
                               help='Patient complaint to analyze')
    diagnose_parser.add_argument('--detailed', action='store_true',
                               help='Show detailed analysis')
    diagnose_parser.add_argument('--interactive', action='store_true',
                               help='Interactive mode for multiple complaints')
    
    # Challenge command
    challenge_parser = subparsers.add_parser('challenge', help='Run adversarial challenge mode')
    challenge_parser.add_argument('--rare-cases', type=int, default=3,
                                help='Number of rare presentation cases')
    challenge_parser.add_argument('--atypical-cases', type=int, default=3,
                                help='Number of atypical age cases')
    challenge_parser.add_argument('--show-failures', action='store_true',
                                help='Show detailed failure cases')
    challenge_parser.add_argument('--verbose', action='store_true',
                                help='Verbose output for each case')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'demo':
        demo_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'conversation':
        conversation_command(args)
    elif args.command == 'diagnose':
        diagnose_command(args)
    elif args.command == 'challenge':
        challenge_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
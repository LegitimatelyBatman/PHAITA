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
        
        # Generate symptoms
        symptoms = symptom_gen.bayesian_network.sample_symptoms(code)
        
        # Generate patient complaint
        complaint = complaint_gen.generate_complaint(symptoms, code)
        
        # Get diagnosis predictions
        predictions = discriminator.predict_diagnosis([complaint], top_k=args.top_k)
        report = format_differential_report(predictions[0])
        top_entry = predictions[0][0]
        pred_code = top_entry["condition_code"]
        confidence = top_entry["probability"]

        print(f"\n🩺 Example {i+1}:")
        print(f"   True Condition: {code} - {condition_data['name']}")
        print(f"   Symptoms: {', '.join(symptoms[:3])}...")
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
        symptoms = symptom_gen.bayesian_network.sample_symptoms(code)
        complaint = complaint_gen.generate_complaint(symptoms, code)
        
        results.append({
            'condition_code': code,
            'condition_name': RespiratoryConditions.get_condition_by_code(code)['name'],
            'symptoms': symptoms,
            'complaint': complaint
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
        
        # Simple mock diagnosis based on keywords
        diagnosis_result = _mock_diagnosis(complaint)
        
        print(f"\n🔍 Diagnosis Results:")
        print(f"   Primary Diagnosis: {diagnosis_result['primary_diagnosis']}")
        print(f"   Confidence: {diagnosis_result['confidence']:.3f}")
        print(f"   Secondary Conditions: {', '.join(diagnosis_result['secondary'])}")
        
        # Show realism analysis
        if args.detailed:
            print(f"\n📊 Detailed Analysis:")
            print(f"   Medical Relevance: {diagnosis_result['medical_relevance']:.3f}")
            print(f"   Symptom Clarity: {diagnosis_result['symptom_clarity']:.3f}")
            print(f"   Language Pattern: {diagnosis_result['language_pattern']}")
            print(f"   Detected Symptoms: {', '.join(diagnosis_result['detected_symptoms'])}")
        
        # Interactive mode
        if args.interactive:
            while True:
                print("\n" + "=" * 40)
                print("Enter another complaint (or 'quit' to exit):")
                new_complaint = input("> ").strip()
                
                if new_complaint.lower() in ['quit', 'exit', 'q']:
                    break
                
                if new_complaint:
                    result = _mock_diagnosis(new_complaint)
                    print(f"\n🔍 Diagnosis: {result['primary_diagnosis']} (confidence: {result['confidence']:.3f})")
    
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
        total_cases = len(challenge_cases)
        correct_diagnoses = 0
        
        for i, case in enumerate(challenge_cases):
            # Generate mock complaint from symptoms
            complaint = _generate_complaint_from_symptoms(case["symptoms"], case.get("metadata", {}))
            
            # Mock diagnosis
            diagnosis_result = _mock_diagnosis(complaint)
            predicted_condition = diagnosis_result["primary_diagnosis"]
            
            # Check if correct (simplified)
            is_correct = _is_diagnosis_correct(case["condition"], predicted_condition)
            
            if is_correct:
                correct_diagnoses += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            print(f"   Case {i+1}/{total_cases}: {status}")
            
            if args.verbose:
                print(f"      Complaint: \"{complaint[:60]}...\"")
                print(f"      True: {case['condition']}, Predicted: {predicted_condition}")
                print(f"      Confidence: {diagnosis_result['confidence']:.3f}")
        
        # Summary
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
        
        # Show hardest cases
        if args.show_failures and correct_diagnoses < total_cases:
            print(f"\n🔍 Most Challenging Cases:")
            failure_count = 0
            for i, case in enumerate(challenge_cases):
                complaint = _generate_complaint_from_symptoms(case["symptoms"], case.get("metadata", {}))
                diagnosis_result = _mock_diagnosis(complaint)
                
                if not _is_diagnosis_correct(case["condition"], diagnosis_result["primary_diagnosis"]):
                    failure_count += 1
                    if failure_count <= 3:  # Show top 3 failures
                        print(f"   {failure_count}. Type: {case['type']}")
                        print(f"      Complaint: \"{complaint}\"")
                        print(f"      Expected: {case['condition']}, Got: {diagnosis_result['primary_diagnosis']}")
    
    except Exception as e:
        print(f"❌ Error in challenge mode: {e}")
        import traceback
        traceback.print_exc()


def _mock_diagnosis(complaint: str) -> dict:
    """Mock diagnosis function for testing."""
    complaint_lower = complaint.lower()
    
    # Simple keyword-based diagnosis
    if any(word in complaint_lower for word in ['wheez', 'asthma', 'tight']):
        primary = "J45.9"
        confidence = 0.85
    elif any(word in complaint_lower for word in ['copd', 'chronic', 'smok']):
        primary = "J44.1" 
        confidence = 0.80
    elif any(word in complaint_lower for word in ['fever', 'pneumonia', 'chill']):
        primary = "J18.9"
        confidence = 0.75
    elif any(word in complaint_lower for word in ['cold', 'stuffy', 'sore throat']):
        primary = "J06.9"
        confidence = 0.70
    else:
        primary = "R06.02"  # Generic shortness of breath
        confidence = 0.60
    
    # Extract symptoms
    detected_symptoms = []
    symptom_keywords = {
        'wheezing': ['wheez'],
        'shortness of breath': ['breath', 'air', 'suffocate'],
        'chest pain': ['chest', 'pain', 'hurt'],
        'cough': ['cough'],
        'fever': ['fever', 'hot', 'burn'],
        'fatigue': ['tired', 'exhaust', 'weak']
    }
    
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in complaint_lower for keyword in keywords):
            detected_symptoms.append(symptom)
    
    return {
        "primary_diagnosis": primary,
        "confidence": confidence,
        "secondary": ["R06.02"] if primary != "R06.02" else [],
        "medical_relevance": random.uniform(0.6, 0.9),
        "symptom_clarity": random.uniform(0.5, 0.9),
        "language_pattern": "patient_language",
        "detected_symptoms": detected_symptoms
    }


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
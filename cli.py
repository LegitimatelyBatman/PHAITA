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
from phaita.triage.info_sheet import format_info_sheet
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


_QUESTION_GENERATOR_FALLBACK_WARNED = False


class LightweightQuestionGenerator:
    """Fallback question generator used when pretrained model is unavailable."""

    def __init__(self, *, use_pretrained: bool = True, **_: dict):
        if not use_pretrained:
            raise ValueError("LightweightQuestionGenerator requires use_pretrained=True.")
        self._index = 0
        self._templates = [
            "Could you describe when the symptoms first started?",
            "Have you noticed anything that makes the symptoms better or worse?",
            "Are there any other symptoms you haven't mentioned yet?",
        ]

    def generate_clarifying_question(
        self,
        symptoms,
        previous_answers=None,
        previous_questions=None,
        conversation_history=None,
        **_: dict,
    ):
        if self._index < len(self._templates):
            question = self._templates[self._index]
        elif symptoms:
            symptom = symptoms[min(self._index - len(self._templates), len(symptoms) - 1)]
            readable = symptom.replace("_", " ")
            question = f"Could you share more details about the {readable}?"
        else:
            question = None

        self._index += 1
        return question


def _build_question_generator() -> QuestionGenerator:
    """Construct a question generator, falling back to a lightweight stub if needed."""

    global _QUESTION_GENERATOR_FALLBACK_WARNED

    try:
        return QuestionGenerator(use_pretrained=True)
    except Exception as exc:  # pragma: no cover - exercised in minimal environments
        if not _QUESTION_GENERATOR_FALLBACK_WARNED:
            print(
                "⚠️ Unable to load the pretrained question generator. "
                "Install transformers, bitsandbytes, and ensure hardware compatibility."
            )
            print(f"   Details: {exc}")
            print("   Falling back to a lightweight template-based question generator.")
            _QUESTION_GENERATOR_FALLBACK_WARNED = True
        return LightweightQuestionGenerator(use_pretrained=True)


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


def _normalize_symptom(symptom: str) -> str:
    return symptom.strip().lower().replace(" ", "_")


def _extract_symptoms_from_text(text: str) -> list:
    candidates = [segment for segment in text.split(",") if segment.strip()]
    normalized = [_normalize_symptom(candidate) for candidate in candidates if candidate.strip()]
    return [symptom for symptom in normalized if symptom]


def _summarize_transcript(chief_complaint: str, turns: list) -> str:
    lines = [f"Chief complaint: {chief_complaint}"]
    for turn in turns[1:]:
        question = turn.get("question", "")
        answer = turn.get("answer", "")
        if question:
            lines.append(f"Q: {question}")
        if answer:
            lines.append(f"A: {answer}")
    return "\n".join(lines)


def _run_triage_dialogue(
    discriminator: DiagnosisDiscriminator,
    engine: ConversationEngine,
    *,
    chief_complaint: str,
    response_provider,
    printer=print,
    initial_extracted_symptoms=None,
    top_k: int = 3,
):
    """Execute information-gain dialogue and update differential in real time."""

    initial_symptoms = list(initial_extracted_symptoms or [])
    engine.add_symptoms(initial_symptoms)

    turns = [
        {
            "question": "Chief complaint",
            "answer": chief_complaint,
            "extracted_symptoms": initial_symptoms,
        }
    ]

    def predict_and_print(header: str):
        transcript = _summarize_transcript(chief_complaint, turns)
        predictions_batch = discriminator.predict_diagnosis([transcript], top_k=top_k)
        ranked_predictions = predictions_batch[0] if predictions_batch else []
        if not ranked_predictions:
            return []
        if header:
            printer(header)
            printer(format_differential_report(ranked_predictions))
        return ranked_predictions

    ranked = predict_and_print("\n🔍 Initial differential:")
    engine.update_differential(ranked)

    while not engine.should_present_diagnosis():
        prompt = engine.next_prompt()
        if not prompt:
            break

        response, extracted_symptoms = response_provider(prompt, len(turns))
        if response is None:
            break

        engine.record_response(prompt, response, extracted_symptoms)
        turns.append(
            {
                "question": prompt,
                "answer": response,
                "extracted_symptoms": list(extracted_symptoms),
            }
        )

        ranked = predict_and_print("\n🔄 Updated differential:") or ranked
        engine.update_differential(ranked)

    return turns, ranked


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

    qgen = _build_question_generator()
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
    """Run the diagnosis workflow with an information-gain conversation loop."""

    print("🩺 PHAITA Diagnosis Tool")
    print("=" * 40)

    try:
        discriminator = _get_diagnosis_discriminator()
        question_generator = _build_question_generator()

        def run_session(complaint: str):
            if not complaint:
                print("❌ No complaint provided")
                return

            print(f"\n📝 Analyzing complaint: \"{complaint}\"")

            engine = ConversationEngine(
                question_generator,
                max_questions=5,
                min_symptom_count=3,
                max_no_progress_turns=2,
            )

            def interactive_response(prompt: str, _turn_index: int):
                print(f"\nAI: {prompt}")
                response = input("You: ").strip()
                if response.lower() in {"quit", "exit", "q"}:
                    print("Ending conversation early at user request.")
                    return None, []
                extracted = _extract_symptoms_from_text(response)
                return response, extracted

            initial_symptoms = _extract_symptoms_from_text(complaint)
            turns, ranked = _run_triage_dialogue(
                discriminator,
                engine,
                chief_complaint=complaint,
                response_provider=interactive_response,
                initial_extracted_symptoms=initial_symptoms,
            )

            if not ranked:
                print("❌ Unable to generate differential diagnosis.")
                return

            top_entry = ranked[0]
            print("\n🔎 Primary diagnosis candidate:")
            print(
                f"   {top_entry.get('condition_name', top_entry.get('condition_code'))}"
                f" ({top_entry.get('condition_code', 'UNKNOWN')})"
            )
            print(f"   Probability: {top_entry.get('probability', 0.0):.3f}")

            if len(ranked) > 1:
                print("   Other differentials:")
                for entry in ranked[1:]:
                    print(
                        f"      - {entry.get('condition_name', entry.get('condition_code'))}"
                        f" ({entry.get('condition_code')}) p={entry.get('probability', 0.0):.2f}"
                    )

            print("\n🗒️  Patient info sheet template:")
            info_sheet = format_info_sheet(
                turns,
                ranked,
                chief_complaint=complaint,
            )
            print(info_sheet)

            if args.detailed:
                print("\n🗂️  Conversation transcript:")
                for turn in turns:
                    question = turn.get("question")
                    answer = turn.get("answer")
                    if question:
                        print(f"   Q: {question}")
                    if answer:
                        print(f"   A: {answer}")

        if args.complaint:
            run_session(args.complaint)
        else:
            print("Enter a patient complaint (or 'quit' to exit):")
            complaint = input("> ").strip()
            if complaint.lower() in {"quit", "exit", "q"}:
                return
            run_session(complaint)

        if args.interactive:
            while True:
                print("\n" + "=" * 40)
                print("Enter another complaint (or 'quit' to exit):")
                new_complaint = input("> ").strip()
                if new_complaint.lower() in {"quit", "exit", "q"}:
                    break
                run_session(new_complaint)

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
                severity="moderate"
            )

            comorbidity_hints = [
                f"history_of_{_normalize_symptom(comorbidity)}"
                for comorbidity in comorbidities
            ]
            augmented_symptoms = list(dict.fromkeys(list(symptoms) + comorbidity_hints))

            metadata = dict(metadata or {})
            metadata["comorbidities"] = list(comorbidities)
            if comorbidity_hints:
                notes = list(metadata.get("notes", []))
                notes.append(
                    "Includes comorbidity cues: " + ", ".join(comorbidities)
                )
                metadata["notes"] = notes
                metadata["comorbidity_hints"] = comorbidity_hints

            print(f"   {i+1}. {condition} with {', '.join(comorbidities)}")
            challenge_cases.append({
                "type": "comorbidity",
                "condition": condition,
                "comorbidities": comorbidities,
                "symptoms": augmented_symptoms,
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

            engine = ConversationEngine(
                _build_question_generator(),
                max_questions=4,
                min_symptom_count=3,
                max_no_progress_turns=2,
            )

            symptom_queue = list(case.get("symptoms", []))
            disclosed_symptoms = set(case.get("symptoms", [])[:2])

            def auto_response(prompt: str, _turn_index: int):
                while symptom_queue:
                    symptom = symptom_queue.pop(0)
                    if symptom in disclosed_symptoms:
                        continue
                    disclosed_symptoms.add(symptom)
                    phrase = symptom.replace("_", " ")
                    return f"Patient reports {phrase}.", [symptom]
                return "No additional symptoms reported.", []

            turns, ranked_predictions = _run_triage_dialogue(
                discriminator,
                engine,
                chief_complaint=complaint,
                response_provider=auto_response,
                initial_extracted_symptoms=case.get("symptoms", [])[:2],
            )

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

                info_sheet = format_info_sheet(
                    turns,
                    ranked_predictions,
                    chief_complaint=complaint,
                )
                print("      Info sheet preview:")
                for line in info_sheet.splitlines():
                    print(f"         {line}")

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
#!/usr/bin/env python3
"""
PHAITA CLI - Command line interface for Pre-Hospital AI Triage Algorithm
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from phaita import AdversarialTrainer, Config, RespiratoryConditions
from phaita.models import SymptomGenerator, ComplaintGenerator, DiagnosisDiscriminator


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
        
        # Get diagnosis prediction
        predictions = discriminator.predict_diagnosis([complaint])
        pred_code, confidence = predictions[0]
        
        print(f"\n🩺 Example {i+1}:")
        print(f"   True Condition: {code} - {condition_data['name']}")
        print(f"   Symptoms: {', '.join(symptoms[:3])}...")
        print(f"   Patient Says: \"{complaint}\"")
        print(f"   AI Diagnosis: {pred_code} (confidence: {confidence:.3f})")
        
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
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--count', type=int, default=10,
                          help='Number of examples to generate')
    gen_parser.add_argument('--condition', type=str,
                          help='Specific condition to generate for')
    gen_parser.add_argument('--output', type=str,
                          help='Output file for generated data')
    
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
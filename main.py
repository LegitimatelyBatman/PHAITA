#!/usr/bin/env python3
"""
PHAITA Main Entry Point - Simplified interface for common tasks

This script provides a centralized, intuitive entry point for the most common
PHAITA workflows:
  - Running demos
  - Training models
  - Running triage/diagnosis
  - Interactive patient simulation

For advanced usage, use cli.py or patient_cli.py directly.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_demo(args):
    """Run a simple demonstration of PHAITA."""
    print("🏥 Running PHAITA Demo...\n")
    
    demo_script = Path(__file__).parent / "demos" / "simple_demo.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(demo_script)],
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return 1


def run_train(args):
    """Train the adversarial model."""
    print("🔥 Training PHAITA model...\n")
    
    cli_script = Path(__file__).parent / "cli.py"
    cmd = [sys.executable, str(cli_script), "train"]
    
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


def run_diagnose(args):
    """Run diagnosis/triage workflow."""
    print("🩺 Running PHAITA Diagnosis...\n")
    
    cli_script = Path(__file__).parent / "cli.py"
    cmd = [sys.executable, str(cli_script), "diagnose"]
    
    if args.complaint:
        cmd.extend(["--complaint", args.complaint])
    if args.interactive:
        cmd.append("--interactive")
    if args.detailed:
        cmd.append("--detailed")
    if args.verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Diagnosis failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running diagnosis: {e}")
        return 1


def run_interactive(args):
    """Run interactive patient simulation."""
    print("👤 Running Interactive Patient Simulation...\n")
    
    patient_cli_script = Path(__file__).parent / "patient_cli.py"
    cmd = [sys.executable, str(patient_cli_script)]
    
    if args.condition:
        cmd.extend(["--condition", args.condition])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.max_turns:
        cmd.extend(["--max-turns", str(args.max_turns)])
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Interactive simulation failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running interactive simulation: {e}")
        return 1


def run_generate(args):
    """Generate synthetic patient data."""
    print("📝 Generating synthetic data...\n")
    
    cli_script = Path(__file__).parent / "cli.py"
    cmd = [sys.executable, str(cli_script), "generate"]
    
    if args.count:
        cmd.extend(["--count", str(args.count)])
    if args.condition:
        cmd.extend(["--condition", args.condition])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return 1


def run_advanced(args):
    """Forward to advanced CLI for full feature access."""
    print("⚙️  Forwarding to advanced CLI...\n")
    
    cli_script = Path(__file__).parent / "cli.py"
    cmd = [sys.executable, str(cli_script)] + args.cli_args
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        print(f"❌ Error running CLI: {e}")
        return 1


def main():
    """Main entry point with simplified command structure."""
    parser = argparse.ArgumentParser(
        description="PHAITA - Pre-Hospital AI Triage Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simplified Entry Point for Common Tasks:

Quick Start Examples:
  python main.py demo                              # Run a quick demo
  python main.py train --epochs 50                 # Train the model
  python main.py diagnose --interactive            # Interactive diagnosis
  python main.py diagnose --complaint "I can't breathe"
  python main.py interactive                       # Patient simulation
  python main.py generate --count 10               # Generate synthetic data

For advanced features, use:
  python cli.py --help                             # Full CLI interface
  python patient_cli.py --help                     # Patient simulator
  
Documentation:
  README.md                 - Quick start guide
  docs/guides/SOP.md        - Standard Operating Procedure
  docs/TESTING.md           - Testing guide
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser(
        'demo',
        help='Run a simple demonstration of PHAITA'
    )
    demo_parser.set_defaults(func=run_demo)
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train the adversarial model'
    )
    train_parser.add_argument('--epochs', type=int,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int,
                             help='Training batch size')
    train_parser.add_argument('--lr', type=float,
                             help='Learning rate')
    train_parser.set_defaults(func=run_train)
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        'diagnose',
        help='Run diagnosis/triage on a patient complaint'
    )
    diagnose_parser.add_argument('--complaint', type=str,
                                help='Patient complaint to analyze')
    diagnose_parser.add_argument('--interactive', action='store_true',
                                help='Interactive mode for multiple complaints')
    diagnose_parser.add_argument('--detailed', action='store_true',
                                help='Show detailed analysis')
    diagnose_parser.set_defaults(func=run_diagnose)
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Run interactive patient simulation'
    )
    interactive_parser.add_argument('--condition', type=str,
                                   help='ICD-10 condition code to simulate')
    interactive_parser.add_argument('--seed', type=int,
                                   help='Random seed for reproducibility')
    interactive_parser.add_argument('--max-turns', type=int,
                                   help='Maximum number of conversation turns')
    interactive_parser.set_defaults(func=run_interactive)
    
    # Generate command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate synthetic patient data'
    )
    generate_parser.add_argument('--count', type=int, default=10,
                                help='Number of examples to generate')
    generate_parser.add_argument('--condition', type=str,
                                help='Specific condition to generate for')
    generate_parser.add_argument('--output', type=str,
                                help='Output file for generated data')
    generate_parser.set_defaults(func=run_generate)
    
    # Advanced command (forwards to cli.py)
    advanced_parser = subparsers.add_parser(
        'cli',
        help='Access full CLI with all advanced features'
    )
    advanced_parser.add_argument('cli_args', nargs=argparse.REMAINDER,
                                help='Arguments to forward to cli.py')
    advanced_parser.set_defaults(func=run_advanced)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n💡 Tip: Start with 'python main.py demo' to see PHAITA in action!")
        return 0
    
    # Execute the selected command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

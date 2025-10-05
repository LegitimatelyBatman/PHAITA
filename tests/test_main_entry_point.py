#!/usr/bin/env python3
"""
Test the main.py centralized entry point.
"""

import subprocess
import sys
from pathlib import Path


def test_main_help():
    """Test that main.py --help works."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "main.py --help should exit with code 0"
    assert "PHAITA - Pre-Hospital AI Triage Algorithm" in result.stdout
    assert "demo" in result.stdout
    assert "train" in result.stdout
    assert "diagnose" in result.stdout
    assert "interactive" in result.stdout
    assert "generate" in result.stdout
    print("✅ test_main_help passed")


def test_main_no_command():
    """Test that main.py without command shows help and tip."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "main.py without command should exit with code 0"
    assert "PHAITA - Pre-Hospital AI Triage Algorithm" in result.stdout
    assert "💡 Tip:" in result.stdout or "Tip:" in result.stdout
    print("✅ test_main_no_command passed")


def test_demo_command_help():
    """Test that main.py demo --help works."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "demo", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "main.py demo --help should exit with code 0"
    assert "main.py demo" in result.stdout or "usage:" in result.stdout
    print("✅ test_demo_command_help passed")


def test_train_command_help():
    """Test that main.py train --help works."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "train", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "main.py train --help should exit with code 0"
    assert "--epochs" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--lr" in result.stdout
    print("✅ test_train_command_help passed")


def test_diagnose_command_help():
    """Test that main.py diagnose --help works."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "diagnose", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "main.py diagnose --help should exit with code 0"
    assert "--complaint" in result.stdout
    assert "--interactive" in result.stdout
    assert "--detailed" in result.stdout
    print("✅ test_diagnose_command_help passed")


def test_interactive_command_help():
    """Test that main.py interactive --help works."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "interactive", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "main.py interactive --help should exit with code 0"
    assert "--condition" in result.stdout
    assert "--seed" in result.stdout
    assert "--max-turns" in result.stdout
    print("✅ test_interactive_command_help passed")


def test_generate_command_help():
    """Test that main.py generate --help works."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "generate", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "main.py generate --help should exit with code 0"
    assert "--count" in result.stdout
    assert "--condition" in result.stdout
    assert "--output" in result.stdout
    print("✅ test_generate_command_help passed")


def test_cli_command_help():
    """Test that main.py cli --help works (shows cli subcommand help)."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "cli", "--help"],
        capture_output=True,
        text=True
    )
    
    # Should show help for the cli subcommand
    assert result.returncode == 0, "main.py cli --help should exit with code 0"
    assert "cli_args" in result.stdout or "Arguments to forward" in result.stdout
    print("✅ test_cli_command_help passed")


def test_demo_command_execution():
    """Test that main.py demo actually runs the demo."""
    main_script = Path(__file__).parent.parent / "main.py"
    
    result = subprocess.run(
        [sys.executable, str(main_script), "demo"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Demo should run successfully
    assert result.returncode == 0, f"main.py demo should exit with code 0, got {result.returncode}"
    # Check for expected output from simple_demo.py
    assert "PHAITA" in result.stdout or "PHAITA" in result.stderr
    assert "Respiratory Conditions" in result.stdout or "respiratory" in result.stdout.lower()
    print("✅ test_demo_command_execution passed")


if __name__ == "__main__":
    print("Running main.py entry point tests...\n")
    
    tests = [
        test_main_help,
        test_main_no_command,
        test_demo_command_help,
        test_train_command_help,
        test_diagnose_command_help,
        test_interactive_command_help,
        test_generate_command_help,
        test_cli_command_help,
        test_demo_command_execution,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print('='*60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

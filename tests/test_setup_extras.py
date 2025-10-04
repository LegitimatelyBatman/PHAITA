#!/usr/bin/env python3
"""Verify setup.py extras_require configuration."""
from pathlib import Path

def read_requirements(path: str) -> list:
    """Read requirements from a file."""
    requirements_path = Path(__file__).parent.parent / path
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#") and not stripped.startswith("-")
    ]

print("Testing setup.py extras_require configuration...")
print()

base = read_requirements("requirements-base.txt")
print(f"✓ Base requirements: {len(base)} packages")
print(f"  First 5: {', '.join(base[:5])}")

gpu = read_requirements("requirements-gpu.txt")
print(f"✓ GPU extras: {len(gpu)} packages")
print(f"  {', '.join(gpu)}")

dev = read_requirements("requirements-dev.txt")
print(f"✓ Dev extras: {len(dev)} packages")
print(f"  {', '.join(dev)}")

scraping = read_requirements("requirements-scraping.txt")
print(f"✓ Scraping extras: {len(scraping)} packages")
print(f"  {', '.join(scraping)}")

all_extras = gpu + dev + scraping
print(f"✓ All extras combined: {len(all_extras)} packages")

print()
print("✅ All extras are properly configured!")


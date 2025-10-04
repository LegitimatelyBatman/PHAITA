from pathlib import Path
from typing import List

from setuptools import setup, find_packages


def read_requirements(path: str = "requirements.txt") -> List[str]:
    requirements_path = Path(__file__).parent / path
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#") and not stripped.startswith("-")
    ]

setup(
    name="phaita",
    version="0.1.0",
    description="Pre-Hospital AI Triage Algorithm with Adversarial Training",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
)

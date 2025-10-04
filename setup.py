from pathlib import Path
from typing import List

from setuptools import setup, find_packages


def read_requirements(path: str = "requirements.txt") -> List[str]:
    """Read and parse a requirements file."""
    requirements_path = Path(__file__).parent / path
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#") and not stripped.startswith("-")
    ]

# Core dependencies (from requirements-base.txt)
install_requires = read_requirements("requirements-base.txt")

# Optional extras
extras_require = {
    "gpu": read_requirements("requirements-gpu.txt"),
    "dev": read_requirements("requirements-dev.txt"),
    "scraping": read_requirements("requirements-scraping.txt"),
}

# Convenience option to install everything
extras_require["all"] = (
    extras_require["gpu"] 
    + extras_require["dev"] 
    + extras_require["scraping"]
)

setup(
    name="phaita",
    version="0.1.0",
    description="Pre-Hospital AI Triage Algorithm with Adversarial Training",
    packages=find_packages(),
    py_modules=["cli", "patient_cli"],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
)

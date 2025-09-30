from setuptools import setup, find_packages

setup(
    name="phaita",
    version="0.1.0",
    description="Pre-Hospital AI Triage Algorithm with Adversarial Training",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    ],
    python_requires=">=3.8",
)
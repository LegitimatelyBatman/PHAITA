# Implementation Overview

This document highlights where key functionality lives inside the `phaita` package.

## Data & Language Resources
| Module | Purpose |
|--------|---------|
| `phaita/data/icd_conditions.py` | ICD-10 respiratory condition metadata, symptom priors, and lay-language mappings. |
| `phaita/data/forum_scraper.py` | Mock forum posts plus bidirectional lay/medical terminology mapping utilities. |
| `phaita/data/synthetic_generator.py` | Helpers for batching complaint generation for experiments and datasets. |
| `phaita/data/preprocessing.py` | Text normalisation, persistence helpers, and token preparation. |

## Model Layer
| Module | Purpose |
|--------|---------|
| `phaita/models/bayesian_network.py` | Baseline symptom sampler using conditional probabilities. |
| `phaita/models/enhanced_bayesian_network.py` | Adds age, severity, and rare-presentation modifiers. |
| `phaita/models/generator.py` | Complaint generation via templates or quantised Mistral 7B. |
| `phaita/models/discriminator.py` | DeBERTa + symptom graph fusion for diagnosis and authenticity scoring. |
| `phaita/models/question_generator.py` | Clarifying question prompts for demo interactions. |
| `phaita/models/gnn_module.py` | Graph construction and attention layers that power knowledge integration. |

## Training & Evaluation
| Module | Purpose |
|--------|---------|
| `phaita/training/adversarial_trainer.py` | Alternating generator/discriminator optimisation with curriculum scheduling. |
| `phaita/utils/metrics.py` | Accuracy, diversity, and confidence calculations. |
| `phaita/utils/realism_scorer.py` | Transformer-backed realism scoring of complaints. |
| `phaita/utils/config.py` | YAML configuration loader and dataclass wrappers. |

## Interfaces & Tests
| Module | Purpose |
|--------|---------|
| `cli.py` | Unified entry point for demos, generation, training, and diagnosis tools. |
| `simple_demo.py`, `demo_deep_learning.py` | Walkthrough scripts showcasing the lightweight and full-model flows. |
| `test_*.py` | Regression tests for data utilities, forum scraping, Bayesian sampling, and end-to-end integrations. |

## Notes
- Quantised deep-learning models are optional; the template pipeline remains available for CPU-only environments.
- Documentation files (README, PROJECT_SUMMARY, IMPLEMENTATION_DETAILS, DEEP_LEARNING_GUIDE) explain architecture, training strategy, and historical changes at increasing levels of depth.
